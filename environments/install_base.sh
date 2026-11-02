#!/bin/bash
# base env: any2sim package (CLI, Scene IR, tools that need no deep-learning model), mujoco checks, CoACD.
set -x
N=$(cd "$(dirname "$0")/.." && pwd); source $N/environments/env.sh
E=$N/envs/base
[ -x $E/bin/python ] || uv venv --python 3.11 $E
uv pip install --python $E/bin/python -e "$N[dev]"
$E/bin/python -c "import any2sim, mujoco, coacd, trimesh; print('BASE_OK', mujoco.__version__)"
echo INSTALL_BASE_DONE
