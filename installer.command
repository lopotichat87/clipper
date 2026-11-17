#!/bin/bash
# Installation (une seule fois) : crée l'environnement Python et installe les dépendances.
cd "$(dirname "$0")"
PY=$(command -v python3.11 || command -v python3.12 || command -v python3)
echo "Python utilisé : $PY"
"$PY" -m venv .venv && ./.venv/bin/pip install --upgrade pip && ./.venv/bin/pip install -r requirements.txt \
  && echo "Installation terminée. Lance Clipper avec lancer.command."
