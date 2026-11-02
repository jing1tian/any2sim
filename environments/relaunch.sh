#!/bin/bash
# relaunch.sh <script-under-environments> : stop a running copy of an install script and start it again detached
N=$(cd "$(dirname "$0")/.." && pwd); name=$1
for p in $(pgrep -f "bash environments/$name"); do pkill -P $p; kill $p; done
sleep 1
cd $N && (setsid nohup bash environments/$name > logs/${name%.sh}.log 2>&1 < /dev/null &)
echo "relaunched $name"
