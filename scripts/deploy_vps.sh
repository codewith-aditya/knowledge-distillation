#!/usr/bin/env bash
#
# One-shot setup for the Autonomous Knowledge Distillation system on a fresh VPS.
#
#   bash deploy_vps.sh [TARGET_DIR]
#
# Idempotent: safe to re-run. Creates venv, installs deps, writes config,
# installs the systemd unit, and (re)starts the service.
#
# Requirements on the target: Ubuntu/Debian, python3, redis-server.
#
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-/root/distillation}"
VENV="$TARGET/venv"

if [ ! -f "$SRC_DIR/requirements.txt" ]; then
  echo "Source directory does not look like the distillation repo: $SRC_DIR" >&2
  exit 1
fi

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ── 1. System packages ───────────────────────────────────────
log "Checking system dependencies"
export DEBIAN_FRONTEND=noninteractive
MISSING=""
command -v python3 >/dev/null || MISSING="$MISSING python3"
dpkg -s python3-venv >/dev/null 2>&1 || MISSING="$MISSING python3-venv"
dpkg -s python3-pip  >/dev/null 2>&1 || MISSING="$MISSING python3-pip"
if [ -n "$MISSING" ]; then
  apt-get update -qq
  apt-get install -y -qq $MISSING
fi

if ! command -v redis-server >/dev/null; then
  log "Installing redis-server"
  apt-get update -qq
  apt-get install -y -qq redis-server
fi
systemctl enable --now redis-server >/dev/null 2>&1 || systemctl enable --now redis >/dev/null 2>&1
redis-cli ping >/dev/null || { echo "Redis is not responding"; exit 1; }
echo "redis: $(redis-cli ping)"

# ── 2. Copy application ──────────────────────────────────────
log "Syncing application to $TARGET"
mkdir -p "$TARGET"
if [ "$SRC_DIR" != "$TARGET" ]; then
  rsync -a --delete \
    --exclude 'venv' --exclude '.env' --exclude '__pycache__' \
    --exclude '*.pyc' --exclude 'data' \
    "$SRC_DIR/" "$TARGET/"
fi
mkdir -p "$TARGET"/{config,logs,data/checkpoints}

# ── 3. Virtualenv + dependencies ─────────────────────────────
log "Building virtualenv"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$TARGET/requirements.txt"
"$VENV/bin/python" -c "import yaml, httpx, redis, rich; print('deps ok')"

# ── 4. Env file ──────────────────────────────────────────────
if [ ! -f "$TARGET/.env" ]; then
  log "Writing .env (endpoints use no API key)"
  cat > "$TARGET/.env" <<'EOF'
# Both model endpoints are keyless OpenAI-compatible servers.
DISTILL_RESEARCH_API_KEY=
DISTILL_TEACHER_API_KEY=
EOF
  chmod 600 "$TARGET/.env"
fi

# ── 5. Output paths ──────────────────────────────────────────
log "Preparing output directories"
mkdir -p /root/datasets/jsonl /root/logs /root/checkpoints

# ── 6. systemd unit ──────────────────────────────────────────
log "Installing systemd unit"
cat > /etc/systemd/system/distillation.service <<EOF
[Unit]
Description=Autonomous AI Knowledge Distillation System
After=network-online.target redis-server.service
Wants=network-online.target
Requires=redis-server.service

[Service]
Type=simple
User=root
WorkingDirectory=$TARGET
Environment="PATH=$VENV/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=$TARGET/.env
ExecStart=$VENV/bin/python $TARGET/main.py $TARGET/config/production.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=distillation
# A single run holds one 12-worker asyncio loop; never let it get OOM-killed
# silently — the checkpoint loop needs a chance to flush.
OOMPolicy=stop
TimeoutStopSec=60
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable distillation >/dev/null

# ── 7. Verify endpoints before starting ──────────────────────
log "Verifying model endpoints"
RESEARCH_URL=$("$VENV/bin/python" - <<PY
import yaml
c = yaml.safe_load(open("$TARGET/config/production.yaml"))
print(c["research_base_url"], c["teacher_base_url"])
PY
)
set -- $RESEARCH_URL
for url in "$1" "$2"; do
  code=$(curl -s -m 15 -o /dev/null -w '%{http_code}' "$url/models" || echo 000)
  printf '  %-40s %s\n' "$url/models" "$code"
  if [ "$code" != "200" ]; then
    echo "  !! endpoint unreachable — service will start but generate nothing"
  fi
done

# ── 8. Start ─────────────────────────────────────────────────
log "Starting service"
systemctl restart distillation
sleep 12
systemctl is-active distillation
journalctl -u distillation -n 15 --no-pager | tail -15

log "Done. Useful commands"
cat <<'EOF'
  systemctl status distillation
  journalctl -u distillation -f
  tail -f /root/logs/distill_$(date +%Y%m%d).log
  grep -c '^TOPIC:' /root/datasets/raw_001.txt      # sample count
  redis-cli llen distill:queue:answer              # backlog
  bash /root/bin/qna.sh -s                          # latest Q&A
EOF
