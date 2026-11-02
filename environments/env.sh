# source before installing environments or downloading weights
N=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export HF_HOME=$N/models/huggingface
export TORCH_HOME=$N/models/torch
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1   # hf-mirror rejects Xet transfers (CAS 401)
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export UV_HTTP_TIMEOUT=600
export PATH=$HOME/.local/bin:$PATH
