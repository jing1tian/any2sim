#!/bin/bash
# Additions to the copied envs: LPIPS (sam3d env, used by evaluate_render) + its AlexNet backbone weights.
set -e
N=$(cd "$(dirname "$0")/.." && pwd); source $N/environments/env.sh
uv pip install --python $N/envs/sam3d/bin/python --no-deps "lpips==0.1.4"
mkdir -p $N/models/torch/hub/checkpoints
[ -f $N/models/torch/hub/checkpoints/alexnet-owt-7be5be79.pth ] || \
  curl -sSL -o $N/models/torch/hub/checkpoints/alexnet-owt-7be5be79.pth https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
TORCH_HOME=$N/models/torch $N/envs/sam3d/bin/python -c "import lpips; lpips.LPIPS(net='alex'); print('LPIPS_OK')"

# agent tooling: mesh booleans / UV unwrap for agent-written geometry (base + sam3d, where fits render), MCP server
uv pip install --python $N/envs/base/bin/python manifold3d xatlas "mcp>=1.2,<2"
uv pip install --python $N/envs/sam3d/bin/python --no-deps manifold3d
$N/envs/base/bin/python -c "import manifold3d, xatlas, mcp; print('AGENT_TOOLING_OK')"
$N/envs/sam3d/bin/python -c "import manifold3d, xatlas; print('SAM3D_MESH_OK')"
echo INSTALL_EXTRAS_DONE
