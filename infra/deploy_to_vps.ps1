<#
deploy_to_vps.ps1

Déploie NBA Predictor sur le VPS depuis le PC (voir docs/deploiement.md) :

  1. build du front en export statique (front/out/), API appelée sur /api
  2. archive : code de l'API (fichiers suivis par git, dans leur état local),
     features_lib.py, modèle entraîné (data/models/saved_models/), front/out/
  3. envoi sur le VPS (scp), puis infra/vps/deploy_remote.sh (ssh) : installe
     le tout et redémarre le service nba-predictor-api
  4. contrôle de https://nba.remicoimbra.fr

Rien n'est construit ni installé avec npm sur le VPS. Les données du jour
(team_state.json, schedule.json) ne sont PAS envoyées ici : c'est le rôle de
sync_to_vps.ps1.

Usage :
  .\infra\deploy_to_vps.ps1                          # vers ubuntu@remicoimbra.fr
  .\infra\deploy_to_vps.ps1 -VpsHost ubuntu@autre    # (ou variable NBA_VPS_HOST)

Prérequis : connexion SSH par clé vers le VPS, front/node_modules installé,
modèle entraîné (python models\train_model.py dans data/).
Journal : infra/logs/deploy_AAAA-MM-JJ.log
#>

param(
    [string]$VpsHost = $env:NBA_VPS_HOST,
    [string]$SiteUrl = "https://nba.remicoimbra.fr"
)

# "Continue" et pas "Stop" : sous PowerShell 5.1, la moindre ligne écrite sur
# stderr par un outil externe deviendrait une erreur fatale. Les échecs sont
# détectés via le code de sortie de chaque étape.
$ErrorActionPreference = "Continue"

if (-not $VpsHost) { $VpsHost = "ubuntu@remicoimbra.fr" }
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$FrontDir = Join-Path $ProjectRoot "front"
$ModelDir = Join-Path $ProjectRoot "data\models\saved_models"
$ModelFiles = @("xgboost_v1.pkl", "feature_columns.json", "model_metrics.json")
# Fichiers de code envoyés : uniquement ceux suivis par git (pas de .env,
# __pycache__ ou fichier de test égaré).
$CodePaths = @("api/app", "api/requirements.txt", "data/preprocessing/features_lib.py")
# Dossier d'arrivée sur le VPS, relatif au home d'ubuntu.
$RemoteIncoming = "nba-predictor-incoming"
# tar de Windows (bsdtar), pas celui de Git Bash s'il est dans le PATH.
$Tar = Join-Path $env:SystemRoot "System32\tar.exe"

$LogDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("deploy_{0}.log" -f (Get-Date -Format "yyyy-MM-dd"))

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$Utf8NoBom = New-Object System.Text.UTF8Encoding $false

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

$Staging = Join-Path ([System.IO.Path]::GetTempPath()) ("nba-deploy-" + [guid]::NewGuid().ToString("N"))

try {
    Write-Log "=== Déploiement NBA Predictor -> $VpsHost ==="
    foreach ($name in $ModelFiles) {
        if (-not (Test-Path (Join-Path $ModelDir $name))) {
            throw "Modèle incomplet : $name absent de $ModelDir (lancer models\train_model.py)"
        }
    }

    # 1. Front : export statique, l'API est sur le même domaine (/api).
    Push-Location $FrontDir
    try {
        $env:NEXT_PUBLIC_API_URL = "/api"
        Invoke-Step "Build du front" "npm.cmd" @("run", "build")
    }
    finally {
        Remove-Item Env:NEXT_PUBLIC_API_URL -ErrorAction SilentlyContinue
        Pop-Location
    }
    $OutDir = Join-Path $FrontDir "out"
    if (-not (Test-Path (Join-Path $OutDir "index.html"))) { throw "Build du front incomplet : out\index.html absent" }
    # Si NEXT_PUBLIC_API_URL n'a pas été pris en compte, l'URL de dev serait
    # figée dans le bundle et le site appellerait l'API... du visiteur.
    if (Select-String -Path (Join-Path $OutDir "_next\static\chunks\*.js") -Pattern "127.0.0.1:8000" -SimpleMatch -Quiet) {
        throw "Le bundle contient encore l'URL de dev (127.0.0.1:8000) au lieu de /api"
    }

    # 2. Archive
    Write-Log "--- Préparation de l'archive"
    New-Item -ItemType Directory -Path $Staging | Out-Null
    $tracked = & git -C $ProjectRoot ls-files -- @CodePaths
    if ($LASTEXITCODE -ne 0 -or -not $tracked) { throw "git ls-files a échoué" }
    foreach ($rel in $tracked) {
        $dst = Join-Path $Staging ("code/" + $rel)
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
        Copy-Item -Path (Join-Path $ProjectRoot $rel) -Destination $dst -ErrorAction Stop
    }
    New-Item -ItemType Directory -Path (Join-Path $Staging "models") | Out-Null
    foreach ($name in $ModelFiles) {
        Copy-Item -Path (Join-Path $ModelDir $name) -Destination (Join-Path $Staging "models") -ErrorAction Stop
    }
    Copy-Item -Path $OutDir -Destination (Join-Path $Staging "front") -Recurse -ErrorAction Stop

    $commit = (& git -C $ProjectRoot rev-parse --short HEAD).Trim()
    $dirty = & git -C $ProjectRoot status --porcelain -- api data/preprocessing front/src front/public front/package.json front/next.config.mjs
    $version = "commit $commit"
    if ($dirty) { $version += " + modifications non commitées" }
    $version += ", déployé le " + (Get-Date -Format "yyyy-MM-dd HH:mm")
    [System.IO.File]::WriteAllText((Join-Path $Staging "VERSION"), "$version`n", $Utf8NoBom)

    # Script distant en LF quoi qu'il arrive (bash refuse les CRLF).
    $remoteScript = Join-Path $Staging "deploy_remote.sh"
    $scriptText = [System.IO.File]::ReadAllText((Join-Path $PSScriptRoot "vps\deploy_remote.sh")) -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText($remoteScript, $scriptText, $Utf8NoBom)

    $Bundle = Join-Path $Staging "bundle.tar.gz"
    Invoke-Step "Compression ($version)" $Tar @("-czf", $Bundle, "-C", $Staging, "code", "models", "front", "VERSION")

    # 3. Envoi et installation
    $sshOpts = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=20")
    Invoke-Step "Dossier d'arrivée sur le VPS" "ssh" ($sshOpts + @($VpsHost, "mkdir -p ~/$RemoteIncoming && chmod 700 ~/$RemoteIncoming"))
    Invoke-Step "Envoi de l'archive" "scp" ($sshOpts + @($Bundle, $remoteScript, "${VpsHost}:$RemoteIncoming/"))
    Invoke-Step "Installation sur le VPS" "ssh" ($sshOpts + @($VpsHost, "bash ~/$RemoteIncoming/deploy_remote.sh ~/$RemoteIncoming"))

    # 4. Contrôle public (Caddy -> front et API)
    foreach ($path in @("/", "/api/")) {
        $code = & curl.exe -s -o NUL -m 20 -w "%{http_code}" "$SiteUrl$path"
        Write-Log "Contrôle $SiteUrl$path -> $code"
        if ($code -ne "200") { throw "Contrôle public en échec : $SiteUrl$path -> $code" }
    }

    Write-Log "=== Déploiement terminé ==="
    exit 0
}
catch {
    Write-Log "ÉCHEC : $_"
    exit 1
}
finally {
    Remove-Item -Recurse -Force -Path $Staging -ErrorAction SilentlyContinue
}
