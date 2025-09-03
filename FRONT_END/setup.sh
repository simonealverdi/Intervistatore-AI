#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Setup AI Interview Frontend"
echo "=============================="

# colori
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err() { echo -e "${RED}[ERR ]${NC}  $*"; }

# 0) Verifica di essere nella cartella FRONT_END
if [ ! -f "package.json" ]; then
  err "package.json non trovato nella dir corrente. Vai in FRONT_END/ e rilancia."
  exit 1
fi

# 1) Node.js: usa quello del Codespace (nvm). NON fare apt upgrade/install.
if ! command -v node >/dev/null 2>&1; then
  err "Node non trovato. In Codespaces dovrebbe esserci nvm. Apri un nuovo terminale o esegui: 'source ~/.nvm/nvm.sh && nvm install --lts'."
  exit 1
else
  log "Node: $(node -v)"
  log "npm:  $(npm -v)"
fi

# 2) Installa deps del frontend
if [ -f package-lock.json ]; then
  log "Installazione dipendenze con npm ci..."
  npm ci
else
  log "Installazione dipendenze con npm install..."
  npm install
fi

# 3) Hardening: assicurati che gli eseguibili in .bin siano eseguibili
chmod +x node_modules/.bin/* 2>/dev/null || true

# 4) Fallback per Vite (se lo shim .bin fosse anomalo)
if [ ! -x node_modules/.bin/vite ]; then
  warn "Shim Vite non eseguibile. Creo symlink di fallback..."
  rm -f node_modules/.bin/vite
  ln -s ../vite/bin/vite.js node_modules/.bin/vite || true
  chmod +x node_modules/vite/bin/vite.js || true
fi

# 5) Consiglio per PostCSS warning (non modifico io i file)
if [ -f postcss.config.js ]; then
  warn "Se vedi warning MODULE_TYPELESS_PACKAGE_JSON, rinomina 'postcss.config.js' in 'postcss.config.cjs' e usa module.exports."
fi

log "Setup completato! 🎉"
