#!/usr/bin/env bash
set -euo pipefail

# Avvia Vite. Se lo shim è rotto, esegue l’entrypoint JS direttamente.
if [ -x node_modules/.bin/vite ]; then
  exec npm run dev -- --host
else
  echo "[WARN] vite shim non eseguibile, uso node entrypoint"
  exec node node_modules/vite/bin/vite.js --host
fi

