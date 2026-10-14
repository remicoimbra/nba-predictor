#!/usr/bin/env bash
# deploy_remote.sh
#
# Partie VPS du déploiement, lancée par infra/deploy_to_vps.ps1 (utilisateur
# ubuntu) après l'envoi de bundle.tar.gz. Voir docs/deploiement.md.
#
#   bash deploy_remote.sh <dossier contenant bundle.tar.gz>
#
# 1. dépendances Python (seulement si requirements.txt a changé)
# 2. code de l'API (l'ancien reste dans api/app.prev pour un retour arrière)
# 3. modèle entraîné
# 4. front : nouvelle version dans releases/, bascule atomique du lien current
# 5. redémarrage du service + contrôle de santé (échec -> code de sortie 1)
#
# Les données du jour (team_state.json, schedule.json) ne sont pas touchées :
# c'est le rôle de sync_to_vps.ps1.

set -euo pipefail

INCOMING="${1:?usage : deploy_remote.sh <dossier contenant bundle.tar.gz>}"
APP_ROOT=/home/ubuntu/nba-predictor
WEB_ROOT=/var/www/nba-predictor
SERVICE=nba-predictor-api
KEEP_RELEASES=3
RELEASE=$(date +%Y%m%d-%H%M%S)
MODEL_FILES=(xgboost_v1.pkl feature_columns.json model_metrics.json)

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

STAGING="$INCOMING/staging-$RELEASE"
trap 'rm -rf "$STAGING"' EXIT
mkdir -p "$STAGING"
tar -xzf "$INCOMING/bundle.tar.gz" -C "$STAGING" code models VERSION

for f in code/api/app/main.py code/api/requirements.txt code/data/preprocessing/features_lib.py VERSION; do
    [ -e "$STAGING/$f" ] || { echo "Archive incomplète : $f manquant" >&2; exit 1; }
done
for f in "${MODEL_FILES[@]}"; do
    [ -e "$STAGING/models/$f" ] || { echo "Archive incomplète : models/$f manquant" >&2; exit 1; }
done

mkdir -p "$APP_ROOT/api" "$APP_ROOT/data/preprocessing" "$APP_ROOT/data/processed" \
    "$APP_ROOT/data/models/saved_models"

# 1. Dépendances Python, AVANT de toucher au code : si pip échoue, la version
# en service reste intacte.
VENV="$APP_ROOT/api/venv"
if [ ! -x "$VENV/bin/python" ] || ! cmp -s "$STAGING/code/api/requirements.txt" "$APP_ROOT/api/requirements.txt"; then
    log "Installation des dépendances Python"
    [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
    # Wheels uniquement : aucun setup.py de paquet exécuté à l'installation.
    "$VENV/bin/pip" install --quiet --disable-pip-version-check --no-input \
        --only-binary=:all: -r "$STAGING/code/api/requirements.txt"
else
    log "Dépendances Python inchangées"
fi

# 2. Code de l'API
log "Code de l'API"
rm -rf "$APP_ROOT/api/app.prev"
if [ -d "$APP_ROOT/api/app" ]; then
    mv "$APP_ROOT/api/app" "$APP_ROOT/api/app.prev"
fi
mv "$STAGING/code/api/app" "$APP_ROOT/api/app"
install -m 644 "$STAGING/code/api/requirements.txt" "$APP_ROOT/api/requirements.txt"
if [ -f "$APP_ROOT/data/preprocessing/features_lib.py" ]; then
    cp -p "$APP_ROOT/data/preprocessing/features_lib.py" "$APP_ROOT/data/preprocessing/features_lib.py.prev"
fi
install -m 644 "$STAGING/code/data/preprocessing/features_lib.py" "$APP_ROOT/data/preprocessing/features_lib.py"

# 3. Modèle (.tmp puis mv : jamais de fichier à moitié copié)
log "Modèle"
for f in "${MODEL_FILES[@]}"; do
    install -m 644 "$STAGING/models/$f" "$APP_ROOT/data/models/saved_models/$f.tmp"
    mv -f "$APP_ROOT/data/models/saved_models/$f.tmp" "$APP_ROOT/data/models/saved_models/$f"
done

# 4. Front. Extrait directement dans releases/ : les fichiers héritent du
# groupe caddy (bit setgid du dossier), seul autorisé à les lire avec ubuntu.
log "Front (version $RELEASE)"
mkdir "$WEB_ROOT/releases/$RELEASE"
tar -xzf "$INCOMING/bundle.tar.gz" -C "$WEB_ROOT/releases/$RELEASE" --strip-components=1 front
[ -f "$WEB_ROOT/releases/$RELEASE/index.html" ] || { echo "Front incomplet : index.html manquant" >&2; exit 1; }
# Bascule atomique : Caddy sert l'ancienne version ou la nouvelle, jamais un mélange.
ln -sfn "releases/$RELEASE" "$WEB_ROOT/current.new"
mv -Tf "$WEB_ROOT/current.new" "$WEB_ROOT/current"
# Garde les KEEP_RELEASES versions les plus récentes (noms = horodatages).
find "$WEB_ROOT/releases" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort \
    | head -n -"$KEEP_RELEASES" | while read -r old; do rm -rf "$WEB_ROOT/releases/$old"; done

install -m 644 "$STAGING/VERSION" "$APP_ROOT/VERSION"

# 5. Redémarrage et contrôle. Le service n'accepte que les connexions venant
# de 127.0.0.5 (filtre IP systemd), d'où --interface.
log "Redémarrage de $SERVICE"
sudo systemctl restart "$SERVICE"
check() { curl -fsS -m 3 --interface 127.0.0.5 -o /dev/null "http://127.0.0.4:8000$1" 2>/dev/null; }
for _ in $(seq 1 30); do
    if check /; then
        check /model/metrics || { echo "API démarrée mais /model/metrics en échec" >&2; exit 1; }
        rm -f "$INCOMING/bundle.tar.gz"
        log "API OK ($(cat "$APP_ROOT/VERSION"))"
        exit 0
    fi
    sleep 1
done
echo "L'API ne répond pas après 30 s : sudo journalctl -u $SERVICE -n 50" >&2
exit 1
