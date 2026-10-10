#!/usr/bin/env bash
# (Re)start the game server in the background. Usage: bash scripts/serve.sh [stop]
cd "$(dirname "$0")/.."
PIDFILE=.server.pid
if [ -f "$PIDFILE" ]; then kill "$(cat "$PIDFILE")" 2>/dev/null; rm -f "$PIDFILE"; sleep 1; fi
[ "${1:-}" = "stop" ] && exit 0
nohup python3 server.py > .server.log 2>&1 &
echo $! > "$PIDFILE"
for _ in $(seq 1 40); do curl -s localhost:${PORT:-8000}/api/config > /dev/null && { echo "server up on http://localhost:${PORT:-8000}"; exit 0; }; sleep 0.5; done
echo "server failed to start; see .server.log"; exit 1
