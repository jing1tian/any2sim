#!/bin/bash
# run_detached.sh <project> <tool> '<json>' : run an any2sim tool detached from the shell (survives ssh drops).
# Output: logs/run_<tool>.out ; the last line is "EXIT <code>".
N=$(cd "$(dirname "$0")/.." && pwd); cd $N
mkdir -p logs
printf '%s' "${3:-{\}}" > logs/run_$2.json
cat > logs/run_$2.sh <<EOS
cd $N
envs/base/bin/any2sim tool '$1' '$2' "\$(cat logs/run_$2.json)"
echo EXIT \$?
EOS
(setsid nohup bash logs/run_$2.sh > logs/run_$2.out 2>&1 < /dev/null &)
echo "started $2 -> logs/run_$2.out"
