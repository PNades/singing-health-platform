<#
.SYNOPSIS
  Create the private bucket and deploy the collection site as Cloud Run service
  strainless-collect (spec 2.8). Safe to re-run: existing resources are kept.

.EXAMPLE
  ./scripts/deploy_collect.ps1 -Project my-project -BillingAccount 0X0X0X-0X0X0X-0X0X0X
  ./scripts/deploy_collect.ps1 -SkipSetup          # redeploy code only

.NOTES
  Needs the gcloud CLI, logged in (gcloud auth login) as a project owner.
  -Bucket and -PaymentFormUrl default to COLLECT_BUCKET and PAYMENT_FORM_URL in .env.
  The image is built from a staged copy of only the files the service needs,
  so data/ and .env can never be uploaded to Cloud Build.
#>
param(
    [string]$Project,
    [string]$Region = "us-central1",
    [string]$Bucket,
    [string]$PaymentFormUrl,
    [string]$BillingAccount,   # set to create the $10/month budget alert (spec 2.8 "Cost")
    [switch]$SkipSetup
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Service = "strainless-collect"

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud CLI not found. Install the Google Cloud SDK and run 'gcloud auth login'."
}

function Invoke-Gcloud {
    & gcloud @args
    if ($LASTEXITCODE -ne 0) { throw "gcloud $($args -join ' ') failed (exit $LASTEXITCODE)" }
}

function Test-Gcloud {
    # True if the gcloud command succeeds; its output and errors are discarded.
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & gcloud @args *> $null; return ($LASTEXITCODE -eq 0) }
    finally { $ErrorActionPreference = $old }
}

function Read-DotEnv([string]$Key) {
    $envFile = Join-Path $Repo ".env"
    if (-not (Test-Path $envFile)) { return $null }
    foreach ($line in Get-Content $envFile) {
        if ($line -match "^\s*$Key\s*=\s*(.*)\s*$") { return $Matches[1].Trim().Trim('"').Trim("'") }
    }
    return $null
}

if (-not $Project) { $Project = (& gcloud config get-value project 2>$null) }
if (-not $Bucket) { $Bucket = Read-DotEnv "COLLECT_BUCKET" }
if (-not $PaymentFormUrl) { $PaymentFormUrl = Read-DotEnv "PAYMENT_FORM_URL" }
if (-not $Project) { throw "No project: pass -Project or run 'gcloud config set project <id>'." }
if (-not $Bucket) { throw "No bucket: pass -Bucket or set COLLECT_BUCKET in .env." }
if (-not $PaymentFormUrl) { Write-Warning "PAYMENT_FORM_URL not set: the Done screen will show no payment button." }

$SA = "$Service@$Project.iam.gserviceaccount.com"
Write-Host "Project $Project, region $Region, bucket gs://$Bucket, service $Service"

if (-not $SkipSetup) {
    Write-Host "`n== APIs"
    $apis = @("run.googleapis.com", "cloudbuild.googleapis.com", "artifactregistry.googleapis.com",
              "storage.googleapis.com", "iam.googleapis.com")
    if ($BillingAccount) { $apis += "billingbudgets.googleapis.com" }
    Invoke-Gcloud services enable @apis --project $Project

    if ($BillingAccount) {
        Write-Host "`n== Budget alert (`$10/month)"
        $existing = & gcloud billing budgets list --billing-account $BillingAccount --format "value(displayName)" 2>$null
        if ($existing -contains "strainless-collect") {
            Write-Host "budget exists"
        } else {
            Invoke-Gcloud billing budgets create --billing-account $BillingAccount `
                --display-name "strainless-collect" --budget-amount 10USD `
                --filter-projects "projects/$Project" `
                --threshold-rule percent=0.5 --threshold-rule percent=0.9 --threshold-rule percent=1.0
        }
    } else {
        Write-Warning "No -BillingAccount: create the `$10/month budget alert in the console before inviting singers."
    }

    Write-Host "`n== Private bucket"
    if (Test-Gcloud storage buckets describe "gs://$Bucket" --project $Project) {
        Write-Host "bucket exists"
    } else {
        Invoke-Gcloud storage buckets create "gs://$Bucket" --project $Project --location $Region `
            --uniform-bucket-level-access --public-access-prevention
    }

    Write-Host "`n== Service account (writes the bucket, nothing else)"
    if (Test-Gcloud iam service-accounts describe $SA --project $Project) {
        Write-Host "service account exists"
    } else {
        Invoke-Gcloud iam service-accounts create $Service --project $Project `
            --display-name "StrainlessAI collection site"
    }
    Invoke-Gcloud storage buckets add-iam-policy-binding "gs://$Bucket" `
        --member "serviceAccount:$SA" --role roles/storage.objectAdmin --quiet | Out-Null
}

Write-Host "`n== Staging build context"
$Stage = Join-Path $env:TEMP "strainless-collect-build"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
$files = @(
    "pyproject.toml", "uv.lock",
    "api/__init__.py", "api/quality.py",
    "api/static/audio/tenor.wav", "api/static/audio/baritone.wav",
    "api/static/audio/alto.wav", "api/static/audio/soprano.wav",
    "ml/__init__.py", "ml/arpeggios.py",
    "ml/features/__init__.py", "ml/features/segment.py",
    "docs/data_protocol.md"
)
$files += Get-ChildItem (Join-Path $Repo "ml/collect") -File -Filter *.py |
    ForEach-Object { "ml/collect/$($_.Name)" }
$files += Get-ChildItem (Join-Path $Repo "ml/collect/static") -File |
    ForEach-Object { "ml/collect/static/$($_.Name)" }
foreach ($f in $files) {
    $src = Join-Path $Repo $f
    if (-not (Test-Path $src)) { throw "missing $f" }
    $dst = Join-Path $Stage $f
    New-Item -ItemType Directory -Force (Split-Path -Parent $dst) | Out-Null
    Copy-Item $src $dst
}
Copy-Item (Join-Path $Repo "ml/collect/Dockerfile") (Join-Path $Stage "Dockerfile")
Set-Content -Path (Join-Path $Stage ".gcloudignore") -Value "__pycache__/" -Encoding ascii
Write-Host "$($files.Count + 1) files staged in $Stage"

Write-Host "`n== Deploy"
# ^@^ makes '@' the separator, so a URL can't break the env-var list.
Invoke-Gcloud run deploy $Service --source $Stage --project $Project --region $Region `
    --service-account $SA --allow-unauthenticated `
    --set-env-vars "^@^COLLECT_BUCKET=$Bucket@PAYMENT_FORM_URL=$PaymentFormUrl" `
    --memory 1Gi --cpu 1 --cpu-boost --min-instances 0 --max-instances 3 `
    --concurrency 4 --timeout 60 --quiet

$Url = (& gcloud run services describe $Service --project $Project --region $Region --format "value(status.url)")
Remove-Item -Recurse -Force $Stage
Write-Host "`nDeployed: $Url"
Write-Host "Make invites: uv run python -m ml.collect.make_invites 3 --url $Url"
