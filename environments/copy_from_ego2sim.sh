#!/bin/bash
# Copy the environments, model weights/sources and robot assets that any2sim needs from the ego2sim
# workspace into any2sim/{envs,models,assets}, then rewrite absolute paths inside the copied venvs.
# Idempotent: rsync skips files already copied. Usage: bash environments/copy_from_ego2sim.sh [part...]
# parts: assets models envs patch (default: all, in that order)
set -u
SRC=${EGO2SIM_WORKSPACE:?Set EGO2SIM_WORKSPACE to the source workspace directory}
DST=$(cd "$(dirname "$0")/.." && pwd)
PARTS=${*:-assets models envs patch}
R="rsync -a --info=stats1"

for part in $PARTS; do
case $part in
assets)
  mkdir -p $DST/assets/robots
  $R "$SRC/robot_assets/" "$DST/assets/robots/"
  echo ASSETS_DONE ;;
models)
  mkdir -p $DST/models/src $DST/models/huggingface $DST/models/torch
  # model checkouts referenced by editable installs (sam-3d-objects keeps its hydra patch)
  for m in sam2 sam3 sam-3d-objects void-model diff-gaussian-rasterization; do
    $R $SRC/model_src/$m $DST/models/src/
  done
  # HF cache keeps its blobs/snapshot symlink layout (relative links); token file excluded
  $R --exclude token --exclude stored_tokens $SRC/weights/huggingface/ $DST/models/huggingface/
  $R $SRC/weights/torch/ $DST/models/torch/
  # sam-3d-objects/checkpoints/hf points at the old cache: repoint to the copy
  ln -sfn $DST/models/huggingface/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints \
    $DST/models/src/sam-3d-objects/checkpoints/hf
  echo MODELS_DONE ;;
envs)
  mkdir -p $DST/envs
  $R $SRC/envs/egl $SRC/envs/cuda-12.1 $DST/envs/
  $R $SRC/envs/vision/ $DST/envs/vision/
  $R $SRC/envs/sam3d/ $DST/envs/sam3d/
  $R $SRC/envs/stage4/ $DST/envs/render/
  echo ENVS_DONE ;;
patch)
  # old absolute prefixes -> new ones, in text files that carry paths (scripts, .pth, editable finders,
  # direct_url.json, activate scripts). Interpreter symlinks point at the shared uv python and stay.
  for pair in vision:vision sam3d:sam3d stage4:render; do
    old=${pair%%:*}; new=${pair##*:}; E=$DST/envs/$new
    find $E/bin $E/lib/python3.11/site-packages -maxdepth 2 -type f \
      \( -path "$E/bin/*" -o -name "*.pth" -o -name "__editable__*.py" -o -name direct_url.json \) -print0 |
      xargs -0 grep -lI -e "$SRC/" 2>/dev/null |
      xargs -r sed -i -e "s#$SRC/envs/$old#$DST/envs/$new#g" -e "s#$SRC/envs/cuda-12.1#$DST/envs/cuda-12.1#g" \
                      -e "s#$SRC/model_src#$DST/models/src#g" -e "s#$SRC/weights#$DST/models#g"
    find $E/lib/python3.11/site-packages -maxdepth 2 -name "__editable__*.pyc" -delete
    left=$(grep -rlI "$SRC/" $E/bin $E/lib/python3.11/site-packages/*.pth $E/lib/python3.11/site-packages/__editable__*.py 2>/dev/null | wc -l)
    echo "PATCHED $new (remaining old refs: $left)"
  done
  echo PATCH_DONE ;;
esac
done
