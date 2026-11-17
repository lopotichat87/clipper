#!/bin/bash
# Démarre Clipper en arrière-plan (tu peux fermer cette fenêtre) puis ouvre l'interface.
cd "$(dirname "$0")"
URL="http://127.0.0.1:7860"
if curl -s -m 2 "$URL/api/health" >/dev/null 2>&1; then
  echo "Clipper tourne déjà."
else
  echo "Démarrage de Clipper en arrière-plan…"
  mkdir -p logs
  nohup ./.venv/bin/python -u server.py --no-browser >> logs/server.log 2>&1 < /dev/null &
  disown
  for i in $(seq 1 40); do
    curl -s -m 1 "$URL/api/health" >/dev/null 2>&1 && break
    sleep 0.5
  done
fi
[ "$1" != "--no-open" ] && open "$URL"
echo "Interface ouverte. Tu peux fermer cette fenêtre : Clipper continue de tourner."
echo "Pour l'arrêter : double-clic sur arreter.command"
