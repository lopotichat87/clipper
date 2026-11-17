#!/bin/bash
# Lance Clipper automatiquement à chaque ouverture de session (nécessaire pour publier chaque jour
# sans y penser). Relance ce script pour désactiver.
cd "$(dirname "$0")"
DIR="$(pwd)"
PLIST="$HOME/Library/LaunchAgents/com.clipper.server.plist"
if [ -f "$PLIST" ]; then
  launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null
  rm -f "$PLIST"
  echo "Démarrage automatique désactivé."
  exit 0
fi
mkdir -p "$HOME/Library/LaunchAgents" logs
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.clipper.server</string>
  <key>ProgramArguments</key><array>
    <string>$DIR/.venv/bin/python</string><string>-u</string><string>$DIR/server.py</string><string>--no-browser</string>
  </array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>StandardOutPath</key><string>$DIR/logs/server.log</string>
  <key>StandardErrorPath</key><string>$DIR/logs/server.log</string>
</dict></plist>
PL
"$DIR/arreter.command" >/dev/null 2>&1
launchctl bootstrap "gui/$(id -u)" "$PLIST" && echo "Démarrage automatique activé : Clipper se lance à chaque ouverture de session."
