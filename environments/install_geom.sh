#!/bin/bash
# geom env: Pi3 / Pi3X (multi-frame cameras + point maps) and MoGe-2 (metric point map, intrinsics, normals).
set -e
N=$(cd "$(dirname "$0")/.." && pwd); source $N/environments/env.sh
E=$N/envs/geom; S=$N/models/src
P="uv pip install --python $E/bin/python"
# Proxy settings, if needed, are inherited from the process environment.
mkdir -p $S
[ -d $S/Pi3 ] || git clone --depth 1 https://github.com/yyfz/Pi3.git $S/Pi3
[ -d $S/MoGe ] || git clone --depth 1 https://github.com/microsoft/MoGe.git $S/MoGe
[ -x $E/bin/python ] || uv venv --python 3.11 $E
$P "torch==2.7.1" "torchvision==0.22.1"
$P "numpy>=1.26,<2" "opencv-python<4.12" pillow scipy trimesh einops safetensors huggingface_hub click pyyaml imageio imageio-ffmpeg \
   "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38" \
   --extra-index-url https://pypi.org/simple || echo WARN_deps
$P -e $S/MoGe || echo WARN_moge
$P -r $S/Pi3/requirements.txt || echo WARN_pi3_reqs
$P "numpy>=1.26,<2" "opencv-python<4.12"
cd $N && $E/bin/python - <<'EOF'
import sys, torch
sys.path.insert(0, "models/src/Pi3")
from pi3.models.pi3x import Pi3X
from moge.model.v2 import MoGeModel
print("GEOM_IMPORTS_OK", torch.__version__, torch.cuda.is_available())
EOF
# weights: parallel ranged downloads from hf-mirror (single connections run at ~2 MB/s)
cd $N
envs/base/bin/python environments/fetch.py https://hf-mirror.com/Ruicheng/moge-2-vitl-normal/resolve/main/model.pt models/checkpoints/moge-2-vitl-normal/model.pt 16
envs/base/bin/python environments/fetch.py https://hf-mirror.com/yyfz233/Pi3X/resolve/main/model.safetensors models/checkpoints/pi3x/model.safetensors 24
echo INSTALL_GEOM_DONE
