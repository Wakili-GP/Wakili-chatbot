#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/wakili-backend"
SERVICE_NAME="wakili-api"
PYTHON_BIN="python3"

if [[ ! -d "$APP_DIR" ]]; then
  echo "Expected app at $APP_DIR"
  exit 1
fi

sudo apt-get update
sudo apt-get install -y python3-pip python3-venv nginx git certbot python3-certbot-nginx

cd "$APP_DIR"
$PYTHON_BIN -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

sudo cp deployment/digitalocean/wakili-api.service /etc/systemd/system/wakili-api.service
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

sudo cp deployment/digitalocean/nginx-api.conf /etc/nginx/sites-available/wakili-api
sudo ln -sf /etc/nginx/sites-available/wakili-api /etc/nginx/sites-enabled/wakili-api
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

echo "Deployment complete. Next steps:"
echo "1) Point DNS A record: api.wakili.me -> your Droplet IP"
echo "2) Issue SSL cert: sudo certbot --nginx -d api.wakili.me"
echo "3) Verify: curl https://api.wakili.me/health"