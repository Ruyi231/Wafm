#!/usr/bin/env bash

export PATH=/nfs/lizhenhao/tools/uv/bin:$PATH
export UV_PYTHON_INSTALL_DIR=/nfs/lizhenhao/tools/uv/python
export UV_CACHE_DIR=/nfs/lizhenhao/tools/uv/cache

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Using uv: $(command -v uv)"
echo "Using python: $(command -v python)"