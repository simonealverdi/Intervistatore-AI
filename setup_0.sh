#!/bin/bash

set -e  # esce al primo errore

echo "🚀 Setup FRONT_END"
cd FRONT_END

echo "📥 Installazione nvm e Node.js 20"
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

export NVM_DIR="$HOME/.nvm"
source "$NVM_DIR/nvm.sh"

nvm install 20
nvm use 20

echo "🧹 Pulizia frontend"
rm -rf node_modules package-lock.json

echo "📦 Installazione dipendenze frontend"
npm install

echo "🔄 Avvio frontend in dev mode"
npm run dev &  # lancia in background

cd ..

echo "🚀 Setup BACK_END"
cd BACK_END

echo "📦 Installazione dipendenze backend"
pip install -r requirements.txt

echo "🔄 Avvio backend (uvicorn)"
python -m uvicorn Main.main:app --reload

