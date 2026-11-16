"""Exercise the real launcher with fake uv jobs: no JAX, data, or GPU required."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

RAW3 = "libero_full_raw_3xfilm_m128_wavloss"
WIDE = "libero_full_raw_film_m384_wavloss"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash is required to test the training launcher")


@pytest.fixture
def launcher(tmp_path):
    source = Path(__file__).resolve().parents[1] / "train_shared_loss.sh"
    (tmp_path / source.name).write_text(source.read_text(), encoding="utf-8", newline="\n")
    # Prove the launcher enforces its GPU selection after sourcing the environment.
    (tmp_path / "env.sh").write_text(
        'export CUDA_VISIBLE_DEVICES=99\ngit() { if [[ "$1" == rev-parse ]]; then echo test-revision; fi; }\n',
        encoding="utf-8",
        newline="\n",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == run && "$2" == python ]]; then echo 'resolved protocol'; exit 0; fi
[[ "$1" == run && "$2" == scripts/train.py ]]
config="$3"
printf 'start|%s|%s|%s\\n' "$config" "$CUDA_VISIBLE_DEVICES" "$*" >> "$TRACE"
if [[ "${FAIL_CONFIG:-}" == "$config" ]]; then exit 7; fi
sleep 0.1
printf 'end|%s\\n' "$config" >> "$TRACE"
""",
        encoding="utf-8",
        newline="\n",
    )
    uv.chmod(0o755)
    env = os.environ.copy()
    for name in ("GPU_ID", "CUDA_VISIBLE_DEVICES", "SEED", "DRY_RUN", "EXP_PREFIX", "LOG_DIR", "FAIL_CONFIG"):
        env.pop(name, None)
    env.update(
        PATH=str(bin_dir) + os.pathsep + env["PATH"],
        TRACE=(tmp_path / "events.txt").as_posix(),
        TRAIN_ENV_FILE=(tmp_path / "env.sh").as_posix(),
    )

    def run(*args, **overrides):
        return subprocess.run(
            [BASH, "train_shared_loss.sh", *args],
            cwd=tmp_path,
            env={**env, **overrides},
            text=True,
            capture_output=True,
            check=False,
        )

    return tmp_path, run


def test_default_queue_is_single_seed_and_serial_on_one_gpu(launcher):
    root, run = launcher
    result = run()
    assert result.returncode == 0, result.stderr
    events = (root / "events.txt").read_text().splitlines()
    assert [e.split("|")[:2] for e in events] == [["start", RAW3], ["end", RAW3], ["start", WIDE], ["end", WIDE]]
    for event in (events[0], events[2]):
        assert event.split("|")[2] == "0"
        assert "--seed=42" in event
    for config in (RAW3, WIDE):
        snapshot = root / "logs/shared_loss" / config / f"{config}_seed42.config.txt"
        assert "training_seed=42" in snapshot.read_text()


def test_dry_run_does_not_start_jobs_or_create_logs(launcher):
    root, run = launcher
    result = run(DRY_RUN="1", GPU_ID="3")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Command:") == 2
    assert result.stdout.count("--seed=42") == 2
    assert "GPU=3" in result.stdout
    assert not (root / "events.txt").exists()
    assert not (root / "logs").exists()


def test_first_failure_stops_second_training(launcher):
    root, run = launcher
    result = run(FAIL_CONFIG=RAW3)
    assert result.returncode == 7
    events = (root / "events.txt").read_text().splitlines()
    assert len(events) == 1
    assert events[0].startswith(f"start|{RAW3}|")
    assert not (root / "logs/shared_loss" / WIDE).exists()


def test_second_run_collision_is_detected_before_first_training(launcher):
    root, run = launcher
    log = root / "logs/shared_loss" / WIDE / f"{WIDE}_seed42.log"
    log.parent.mkdir(parents=True)
    log.write_text("existing run", encoding="utf-8")
    result = run()
    assert result.returncode == 3
    assert not (root / "events.txt").exists()
    assert log.read_text() == "existing run"


def test_explicit_seed_and_gpu_override(launcher):
    root, run = launcher
    result = run("raw_wide_loss", SEED="0", GPU_ID="2", CUDA_VISIBLE_DEVICES="1")
    assert result.returncode == 0, result.stderr
    events = (root / "events.txt").read_text().splitlines()
    assert len(events) == 2
    assert events[0].startswith(f"start|{WIDE}|2|")
    assert "--seed=0" in events[0]


@pytest.mark.parametrize(
    ("args", "env"),
    [
        ([], {"GPU_ID": "0,1"}),
        (["raw_loss", "0", "bad"], {}),
        (["unknown"], {}),
    ],
)
def test_invalid_plan_does_not_start_training(launcher, args, env):
    root, run = launcher
    result = run(*args, **env)
    assert result.returncode == 2
    assert not (root / "events.txt").exists()
