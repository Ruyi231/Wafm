from collections.abc import Sequence

import flax.nnx as nnx
import jax.numpy as jnp
import math
from openpi.shared import array_typing as at


# _SQRT2 = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float32))
_SQRT2 = math.sqrt(2.0)


def _pad_time_to_even(x: at.Array) -> at.Array:
    """单级 DWT 的兜底 padding；多级 DWT 会在入口统一 padding 到 2**levels 的倍数。"""
    time_len = x.shape[1]
    if time_len % 2 == 0:
        return x
    return jnp.pad(x, ((0, 0), (0, 1), (0, 0)), mode="edge")


@at.typecheck
def haar_dwt_1d(
    x: at.Float[at.Array, "b t a"],
) -> tuple[at.Float[at.Array, "b th a"], at.Float[at.Array, "b th a"]]:
    """对 action horizon 维做一级正交 Haar-DWT。"""
    x = _pad_time_to_even(x)
    even = x[:, 0::2, :]
    odd = x[:, 1::2, :]
    approx = (even + odd) / _SQRT2
    detail = (even - odd) / _SQRT2
    return approx, detail


@at.typecheck
def haar_idwt_1d(
    approx: at.Float[at.Array, "b th a"],
    detail: at.Float[at.Array, "b th a"],
) -> at.Float[at.Array, "b t a"]:
    """一级 Haar-IDWT，将 even/odd 交织回时间维。"""
    even = (approx + detail) / _SQRT2
    odd = (approx - detail) / _SQRT2
    return jnp.stack([even, odd], axis=2).reshape(approx.shape[0], approx.shape[1] * 2, approx.shape[2])


def multi_level_haar_dwt(
    x: at.Float[at.Array, "b t a"], levels: int
) -> tuple[at.Float[at.Array, "b tl a"], list[at.Float[at.Array, "b td a"]], int]:
    """多级 Haar-DWT，返回 A_L、[D1, ..., D_L] 和实际层数。"""
    if levels < 0:
        raise ValueError(f"levels must be non-negative, got {levels}")
    if x.shape[1] < 2 or levels == 0:
        return x, [], 0

    actual_levels = min(levels, max(0, (x.shape[1] - 1).bit_length()))
    multiple = 2**actual_levels
    pad_len = (-x.shape[1]) % multiple
    approx = x
    if pad_len:
        # 对完整 action chunk 做一次 edge padding；IDWT 后由 target_length 裁剪回原始 horizon。
        approx = jnp.pad(approx, ((0, 0), (0, pad_len), (0, 0)), mode="edge")

    details = []
    completed_levels = 0
    for _ in range(actual_levels):
        if approx.shape[1] < 2:
            break
        approx, detail = haar_dwt_1d(approx)
        details.append(detail)
        completed_levels += 1
    return approx, details, completed_levels


def multi_level_haar_idwt(
    approx: at.Float[at.Array, "b tl a"],
    details: Sequence[at.Float[at.Array, "b td a"]],
    target_length: int | None = None,
) -> at.Float[at.Array, "b t a"]:
    """多级 Haar-IDWT；details 按 [D1, ..., D_L] 输入，内部从高层向低层重构。"""
    x = approx
    for detail in reversed(details):
        x = haar_idwt_1d(x, detail)
    if target_length is not None:
        x = x[:, :target_length, :]
    return x


class _SubbandFiLMHead(nnx.Module):
    """单个小波子带的 FiLM 调制 head；每个子带实例化一次以保证参数独立。"""

    def __init__(self, action_dim: int, cond_dim: int, bottleneck_dim: int, *, rngs: nnx.Rngs):
        self.state_proj = nnx.Linear(action_dim, bottleneck_dim, rngs=rngs)
        self.film_proj = nnx.Linear(cond_dim, 2 * bottleneck_dim, rngs=rngs)
        self.out_proj = nnx.Linear(bottleneck_dim, action_dim, rngs=rngs)

    def __call__(
        self,
        z_subband: at.Float[at.Array, "b ts a"],
        cond_band: at.Float[at.Array, "b ts h"],
    ) -> at.Float[at.Array, "b ts a"]:
        h_z = self.state_proj(z_subband)
        gamma_beta = self.film_proj(cond_band)
        gamma, beta = jnp.split(gamma_beta, 2, axis=-1)

        # replace 模式下 head 直接决定动作速度场；限制 gamma 幅度可降低训练早期调制过强导致的不稳定风险。
        gamma = 0.1 * jnp.tanh(gamma)
        beta = 0.1 * beta

        h = (1.0 + gamma) * h_z + beta
        h = nnx.gelu(h)
        return self.out_proj(h)


class WaveletSubbandFlowHead(nnx.Module):
    """用 Haar 小波子带速度场替换原 action flow head。"""

    def __init__(
        self,
        *,
        action_dim: int,
        token_dim: int,
        levels: int,
        bottleneck_dim: int,
        use_band_gate: bool,
        rngs: nnx.Rngs,
    ):
        if levels < 1:
            raise ValueError(f"WaveletSubbandFlowHead expects levels >= 1, got {levels}")
        self.levels = levels
        self.use_band_gate = use_band_gate
        self.approx_head = _SubbandFiLMHead(action_dim, token_dim, bottleneck_dim, rngs=rngs)
        self.detail_heads = nnx.Dict(
            **{
                f"D{i}": _SubbandFiLMHead(action_dim, token_dim, bottleneck_dim, rngs=rngs)
                for i in range(1, levels + 1)
            }
        )
        self.gate_proj = nnx.Linear(token_dim, levels + 1, rngs=rngs) if use_band_gate else None

    def __call__(
        self,
        noisy_actions: at.Float[at.Array, "b t a"],
        action_tokens: at.Float[at.Array, "b t h"],
    ) -> tuple[at.Float[at.Array, "b t a"], dict[str, at.Array | list[at.Array] | int | None]]:
        z_approx, z_details, actual_levels = multi_level_haar_dwt(noisy_actions, self.levels)
        h_approx, h_details, _ = multi_level_haar_dwt(action_tokens, actual_levels)

        v_approx = self.approx_head(z_approx, h_approx)
        v_details = [
            self.detail_heads[f"D{i + 1}"](z_detail, h_detail)
            for i, (z_detail, h_detail) in enumerate(zip(z_details, h_details, strict=True))
        ]

        gates = None
        if self.use_band_gate:
            # band gate 是子带级整体开关，仍使用全局 action-token 摘要；FiLM 负责子带内部通道级调制。
            cond_global = jnp.mean(action_tokens, axis=1)
            assert self.gate_proj is not None
            gates = nnx.sigmoid(self.gate_proj(cond_global)[:, : actual_levels + 1])
            v_approx = gates[:, 0, None, None] * v_approx
            v_details = [gates[:, i + 1, None, None] * v_detail for i, v_detail in enumerate(v_details)]

        v_x = multi_level_haar_idwt(v_approx, v_details, target_length=noisy_actions.shape[1])
        info: dict[str, at.Array | list[at.Array] | int | None] = {
            "wavelet_flow_approx": v_approx,
            "wavelet_flow_details": v_details,
            "wavelet_state_approx": z_approx,
            "wavelet_state_details": z_details,
            "wavelet_token_approx": h_approx,
            "wavelet_token_details": h_details,
            "wavelet_levels": actual_levels,
            "wavelet_gates": gates,
        }
        if gates is not None:
            info["wavelet_gate_mean"] = jnp.mean(gates)
            info["wavelet_gate_A_mean"] = jnp.mean(gates[:, 0])
            for i in range(actual_levels):
                info[f"wavelet_gate_D{i + 1}_mean"] = jnp.mean(gates[:, i + 1])
        return v_x, info
