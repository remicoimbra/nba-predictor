# Déploiement

Le site tourne sur un VPS (Ubuntu) à l'adresse **https://nba.remicoimbra.fr**.

```
PC ── deploy_to_vps.ps1 (code, modèle, front)  ──scp/ssh──┐
   └─ sync_to_vps.ps1   (données du jour)      ──scp/ssh──┤
                                                          ▼
Internet ──HTTPS──> Caddy ── /api/* ──> nba-predictor-api (systemd, 127.0.0.4:8000)
                          └─ le reste ─> /var/www/nba-predictor/current (fichiers statiques)
```

- **Front** : export statique de Next.js (`output: "export"`), construit sur le PC et servi tel quel par Caddy. Aucun serveur Node en production. Toutes les données viennent de l'API, appelée depuis le navigateur sur `/api` (même domaine, pas de CORS).
- **API** : FastAPI sous uvicorn, service systemd `nba-predictor-api` ([infra/vps/nba-predictor-api.service](../infra/vps/nba-predictor-api.service)).
- **Données** : `team_state.json` et `schedule.json`, poussés chaque jour depuis le PC (`stats.nba.com` bloque les IP de serveurs, voir [architecture.md](architecture.md)). L'API les relit d'elle-même quand ils changent.

## Sécurité

- L'API tourne sous un utilisateur jetable (`DynamicUser`), voit son dossier en lecture seule et rien d'autre du disque. Elle n'a ni capacité ni moyen d'élever ses privilèges, et ses ressources sont bornées (mémoire, un cœur au maximum).
- Réseau de l'API : elle écoute sur `127.0.0.4` et n'accepte que les connexions venant de `127.0.0.5` (l'adresse source de Caddy). Toute autre adresse est refusée par systemd, en entrée comme en sortie : l'API ne peut joindre ni Internet, ni les autres services de la machine. Score `systemd-analyze security` : 1,4 (« OK »).
- Pas de `/docs` ni d'`openapi.json` en production (`NBA_API_DOCS=0`), pas d'appel à la NBA (`NBA_LIVE_SCHEDULE_FALLBACK=0`).
- Caddy : HTTPS automatique, HSTS, Content-Security-Policy, `nosniff`, `X-Frame-Options`, corps de requête limité à 16 Ko ([infra/vps/nba.remicoimbra.fr.caddy](../infra/vps/nba.remicoimbra.fr.caddy)).
- Rien n'est construit ni installé avec npm sur le serveur. Les dépendances Python sont installées en wheels uniquement (`--only-binary=:all:`) : aucun script d'installation de paquet n'y est exécuté.

## Disposition sur le serveur

| Chemin | Contenu |
| --- | --- |
| `/home/ubuntu/nba-predictor/api/app` | Code de l'API (`app.prev` : version précédente) |
| `/home/ubuntu/nba-predictor/api/venv` | Environnement Python (Python 3.14 du système) |
| `/home/ubuntu/nba-predictor/data/preprocessing/features_lib.py` | Calcul des variables, partagé avec l'entraînement |
| `/home/ubuntu/nba-predictor/data/models/saved_models/` | Modèle entraîné |
| `/home/ubuntu/nba-predictor/data/processed/` | `team_state.json`, `schedule.json` (synchro quotidienne) |
| `/home/ubuntu/nba-predictor/VERSION` | Commit déployé et date |
| `/var/www/nba-predictor/releases/` | Les 3 dernières versions du front ; `current` pointe sur l'active |

Même arborescence qu'en local : l'API trouve `data/` sans configuration.

## Redéployer

Après une modification du code, du front ou un réentraînement du modèle :

```powershell
.\infra\deploy_to_vps.ps1
```

Le script construit le front (`NEXT_PUBLIC_API_URL=/api`), envoie le code de l'API (fichiers suivis par git, dans leur état local), `features_lib.py`, le modèle et le front, puis lance [infra/vps/deploy_remote.sh](../infra/vps/deploy_remote.sh) sur le serveur. Celui-ci :

1. met à jour les dépendances Python, seulement si `api/requirements.txt` a changé ;
2. remplace le code de l'API et le modèle ;
3. installe le front dans une nouvelle version et bascule `current` d'un coup ;
4. redémarre le service et vérifie qu'il répond.

Le script contrôle ensuite le site depuis le PC. Journal : `infra/logs/deploy_AAAA-MM-JJ.log`.

Les données du jour ne passent pas par là : c'est `sync_to_vps.ps1`.

## Synchro quotidienne

```powershell
.\infra\sync_to_vps.ps1 -VpsHost ubuntu@remicoimbra.fr -RemoteDir /home/ubuntu/nba-predictor/data/processed
```

À automatiser avec le Planificateur de tâches Windows (une fois par jour, après la fin des matchs).

## Retour arrière

Front (version précédente) :

```bash
cd /var/www/nba-predictor
ls releases/                                    # versions disponibles
ln -sfn releases/<version> current.new && mv -Tf current.new current
```

API (code précédent) :

```bash
cd ~/nba-predictor/api
mv app app.bad && mv app.prev app && sudo systemctl restart nba-predictor-api
```

## Surveiller

```bash
systemctl status nba-predictor-api                   # état, mémoire utilisée
sudo journalctl -u nba-predictor-api -n 100          # journaux de l'API
cat ~/nba-predictor/VERSION                          # version déployée
ls -l ~/nba-predictor/data/processed                 # date de la dernière synchro
curl -s --interface 127.0.0.5 http://127.0.0.4:8000/ # l'API, sans passer par Caddy
```

## Mises à jour

Les paquets du système (Python, Caddy) sont mis à jour automatiquement (unattended-upgrades). Les dépendances Python de l'API, elles, sont figées dans `api/requirements.txt` : pour les mettre à jour, modifier ce fichier (en gardant scikit-learn, xgboost et joblib alignés sur `data/requirements.txt`, sinon le modèle peut ne plus se charger), réentraîner si besoin, puis redéployer.

## Installation initiale (déjà faite, pour mémoire)

```bash
sudo apt-get install -y python3.14-venv
mkdir -p ~/nba-predictor/{api,data/preprocessing,data/processed,data/models/saved_models}
sudo install -d -o ubuntu -g caddy -m 2750 /var/www/nba-predictor /var/www/nba-predictor/releases
sudo install -m 644 nba-predictor-api.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable nba-predictor-api
```

Puis, depuis le PC, `deploy_to_vps.ps1` et `sync_to_vps.ps1`. Pour Caddy : sauvegarder `/etc/caddy/Caddyfile`, y ajouter le bloc de `nba.remicoimbra.fr.caddy` sur une copie, valider la copie (`sudo -u caddy caddy validate --config <copie>`), puis l'installer et lancer `sudo systemctl reload caddy`. Le certificat HTTPS est obtenu automatiquement.
