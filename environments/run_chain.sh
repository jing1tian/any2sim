#!/bin/bash
# run_chain.sh <project> <name> <tool>=<json-file-or-'{}'> ... : run tools in order, detached; stop at the first failure.
# Output: logs/chain_<name>.out ; last line "CHAIN_EXIT <code> <tool>".
N=$(cd "$(dirname "$0")/.." && pwd); cd $N; mkdir -p logs
proj=$1; name=$2; shift 2
script=logs/chain_$name.sh
echo "cd $N" > $script
for spec in "$@"; do
  tool=${spec%%=*}; js=${spec#*=}
  printf '%s' "$js" > logs/chain_${name}_$tool.json
  echo "envs/base/bin/any2sim tool '$proj' '$tool' \"\$(cat logs/chain_${name}_$tool.json)\" > logs/chain_${name}_$tool.out 2>&1 || { echo CHAIN_EXIT 1 $tool; exit 1; }" >> $script
  echo "echo done $tool" >> $script
done
echo "echo CHAIN_EXIT 0 all" >> $script
(setsid nohup bash $script > logs/chain_$name.out 2>&1 < /dev/null &)
echo "started chain $name"
