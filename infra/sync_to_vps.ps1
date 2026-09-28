<#
sync_to_vps.ps1

Synchro quotidienne des données NBA, à lancer depuis un PC à IP résidentielle
(stats.nba.com et cdn.nba.com bloquent l'IP du VPS, cf. CLAUDE.md "Déploiement") :

  1. fetch_games.py --refresh  : nouveaux matchs joués de la saison en cours
  2. clean_data.py             : -> games_clean.csv
  3. team_state.py             : -> team_state.json (Elo, forme, win%)
  4. fetch_schedule.py         : -> schedule.json (calendrier J-3 -> J+14)
  5. envoi de team_state.json + schedule.json sur le VPS (scp puis mv atomique)

L'API sur le VPS détecte les nouveaux fichiers toute seule (date de
modification), sans redémarrage. Pas de réentraînement du modèle ici : il
n'est pas nécessaire au quotidien.

Usage :
  .\infra\sync_to_vps.ps1 -VpsHost ubuntu@<ip-du-vps> -RemoteDir /chemin/vers/data/processed
  .\infra\sync_to_vps.ps1 -NoPush        # pipeline local uniquement, sans envoi

Prérequis pour l'envoi : connexion SSH par clé (sans mot de passe) vers le
VPS, sinon la tâche planifiée échoue (BatchMode) au lieu d'attendre une saisie.
Journal : infra/logs/sync_AAAA-MM-JJ.log
#>

param(
    [string]$VpsHost = $env:NBA_VPS_HOST,
    [string]$RemoteDir = $env:NBA_VPS_DATA_DIR,
    [string]$Python = $env:NBA_SYNC_PYTHON,
    [switch]$NoPush
)

# "Continue" et pas "Stop" : sous PowerShell 5.1, la moindre ligne écrite par
# Python sur stderr (un simple warning pandas) deviendrait une erreur fatale.
# Les échecs sont détectés via le code de sortie de chaque étape.
$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"
$ProcessedDir = Join-Path $DataDir "processed"
if (-not $Python) { $Python = Join-Path $ProjectRoot "api\venv\Scripts\python.exe" }

$LogDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("sync_{0}.log" -f (Get-Date -Format "yyyy-MM-dd"))

# Sortie Python en UTF-8 (accents, flèches) quel que soit le contexte d'exécution.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

function Write-Log([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

function Invoke-Step([string]$Label, [string]$Exe, [string[]]$Arguments) {
    Write-Log "--- $Label"
    & $Exe @Arguments 2>&1 | ForEach-Object { Write-Log "$_" }
    if ($LASTEXITCODE -ne 0) {
        throw "$Label a échoué (code de sortie $LASTEXITCODE)"
    }
}

try {
    Write-Log "=== Synchro NBA Predictor ==="
    if (-not (Test-Path $Python)) { throw "Python introuvable : $Python" }

    Push-Location $DataDir
    try {
        Invoke-Step "Historique (saison en cours)" $Python @("collectors\fetch_games.py", "--refresh")
        Invoke-Step "Nettoyage" $Python @("preprocessing\clean_data.py")
        Invoke-Step "État des équipes" $Python @("preprocessing\team_state.py")
        Invoke-Step "Calendrier" $Python @("collectors\fetch_schedule.py")
    }
    finally {
        Pop-Location
    }

    if ($NoPush) {
        Write-Log "Envoi vers le VPS ignoré (-NoPush)"
    }
    else {
        if (-not $VpsHost -or -not $RemoteDir) {
            throw "VpsHost et RemoteDir sont requis pour l'envoi (paramètres ou variables NBA_VPS_HOST / NBA_VPS_DATA_DIR)"
        }
        $sshOpts = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=20")
        foreach ($name in @("team_state.json", "schedule.json")) {
            Invoke-Step "Envoi $name" "scp" ($sshOpts + @((Join-Path $ProcessedDir $name), "${VpsHost}:$RemoteDir/$name.tmp"))
        }
        # Remplacement atomique : l'API ne lit jamais un fichier à moitié copié.
        $remoteCmd = "mv -f '$RemoteDir/team_state.json.tmp' '$RemoteDir/team_state.json' && mv -f '$RemoteDir/schedule.json.tmp' '$RemoteDir/schedule.json'"
        Invoke-Step "Activation sur le VPS" "ssh" ($sshOpts + @($VpsHost, $remoteCmd))
    }

    Write-Log "=== Synchro terminée ==="
    exit 0
}
catch {
    Write-Log "ÉCHEC : $_"
    exit 1
}
