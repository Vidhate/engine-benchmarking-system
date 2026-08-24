#!/usr/bin/env bash
# Start `langgraph dev` in the background and wait until it is healthy.
#   scripts/serve.sh start   -> writes .server.pid, blocks until /ok responds
#   scripts/serve.sh stop    -> kills it
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$APP_DIR/scripts/.server.pid"
LOG_FILE="$APP_DIR/scripts/.server.log"
PORT="${TARGET_APP_PORT:-2024}"

start() {
  stop || true
  cd "$APP_DIR"
  uv run langgraph dev --port "$PORT" --no-browser --no-reload >"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  for _ in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:$PORT/ok" >/dev/null 2>&1; then
      echo "langgraph dev healthy on port $PORT (pid $(cat "$PID_FILE"))"
      return 0
    fi
    sleep 1
  done
  echo "server did not become healthy; last log lines:" >&2
  tail -30 "$LOG_FILE" >&2
  return 1
}

stop() {
  if [ -f "$PID_FILE" ]; then
    pkill -P "$(cat "$PID_FILE")" 2>/dev/null || true
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    rm -f "$PID_FILE"
  fi
  pkill -f "langgraph dev --port $PORT" 2>/dev/null || true
}

case "${1:-start}" in
  start) start ;;
  stop) stop; echo "stopped" ;;
  *) echo "usage: $0 {start|stop}" >&2; exit 2 ;;
esac
