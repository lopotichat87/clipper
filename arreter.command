#!/bin/bash
# Arrête le serveur Clipper (les générations en cours seront interrompues).
cd "$(dirname "$0")"
PID=$(cat config/server.pid 2>/dev/null)
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
  kill "$PID" && echo "Clipper arrêté."
else
  PIDS=$(lsof -tiTCP:7860 -sTCP:LISTEN 2>/dev/null)
  if [ -n "$PIDS" ]; then kill $PIDS && echo "Clipper arrêté."; else echo "Clipper n'était pas lancé."; fi
fi
