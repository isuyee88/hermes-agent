param(
    [string]$AccountId = "d1215a30b84b673ef0367010b0e78c10",
    [string]$WorkerName = "hermes-feishu-gateway",
    [string]$ModalInternalBaseUrl = "https://isuyee88--hermes-agent-web-app.modal.run",
    [string]$ProxyUrl = "",
    [string]$CompatibilityDate = "2026-04-14"
)

$ErrorActionPreference = "Stop"

function Get-EnvValue {
    param([string]$Name)
    $value = [Environment]::GetEnvironmentVariable($Name, "Process")
    if (-not $value) {
        $value = [Environment]::GetEnvironmentVariable($Name, "User")
    }
    if (-not $value) {
        $value = [Environment]::GetEnvironmentVariable($Name, "Machine")
    }
    return $value
}

function Get-DerivedInternalBearer {
    param(
        [string]$AppId,
        [string]$AppSecret
    )
    if (-not $AppId -or -not $AppSecret) {
        return $null
    }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes("hermes-feishu-internal:$AppId`:$AppSecret")
        $hash = $sha.ComputeHash($bytes)
        $hex = -join ($hash | ForEach-Object { $_.ToString("x2") })
        return "fi_$hex"
    }
    finally {
        $sha.Dispose()
    }
}

if ($ProxyUrl) {
    $env:HTTP_PROXY = $ProxyUrl
    $env:HTTPS_PROXY = $ProxyUrl
    $env:ALL_PROXY = $ProxyUrl
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$bundleDir = Join-Path $repoRoot ".tmp-cf-bundle"
$bundlePath = Join-Path $bundleDir "index.js"
$metadataPath = Join-Path $repoRoot ".tmp-cf-upload-metadata.json"

$token = Get-EnvValue "CLOUDFLARE_API_TOKEN"
$appId = Get-EnvValue "FEISHU_APP_ID"
$appSecret = Get-EnvValue "FEISHU_APP_SECRET"
$internalBearer = Get-EnvValue "HERMES_FEISHU_INTERNAL_BEARER_TOKEN"

if (-not $token) {
    throw "CLOUDFLARE_API_TOKEN is required."
}
if (-not $appId) {
    throw "FEISHU_APP_ID is required."
}
if (-not $appSecret) {
    throw "FEISHU_APP_SECRET is required."
}
if (-not $internalBearer) {
    $internalBearer = Get-DerivedInternalBearer -AppId $appId -AppSecret $appSecret
}
if (-not $internalBearer) {
    throw "Unable to derive HERMES_FEISHU_INTERNAL_BEARER_TOKEN."
}

Push-Location $repoRoot
try {
    wrangler deploy --dry-run --outdir $bundleDir | Out-Null
    if (-not (Test-Path $bundlePath)) {
        throw "Bundled worker not found at $bundlePath"
    }

    $metadata = @{
        main_module = "index.js"
        compatibility_date = $CompatibilityDate
        compatibility_flags = @("nodejs_compat")
        observability = @{
            enabled = $true
            head_sampling_rate = 1
            logs = @{
                enabled = $true
                invocation_logs = $true
                head_sampling_rate = 1
                persist = $true
            }
            traces = @{
                enabled = $true
                head_sampling_rate = 1
                persist = $true
            }
        }
        bindings = @(
            @{
                name = "FEISHU_AGENT_WORKFLOW"
                type = "workflow"
                workflow_name = "hermes-feishu-agent-workflow"
                class_name = "FeishuAgentWorkflow"
            },
            @{
                name = "FEISHU_API_BASE"
                type = "plain_text"
                text = "https://open.feishu.cn"
            },
            @{
                name = "FEISHU_ACK_REACTION_EMOJI"
                type = "plain_text"
                text = "OK"
            },
            @{
                name = "MODAL_INTERNAL_BASE_URL"
                type = "plain_text"
                text = $ModalInternalBaseUrl
            },
            @{
                name = "FEISHU_APP_ID"
                type = "secret_text"
                text = $appId
            },
            @{
                name = "FEISHU_APP_SECRET"
                type = "secret_text"
                text = $appSecret
            },
            @{
                name = "MODAL_INTERNAL_BEARER_TOKEN"
                type = "secret_text"
                text = $internalBearer
            }
        )
    } | ConvertTo-Json -Depth 12

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($metadataPath, $metadata, $utf8NoBom)

    $uploadRaw = curl.exe -sS -X PUT `
        -H "Authorization: Bearer $token" `
        -F "metadata=@$metadataPath;type=application/json" `
        -F "index.js=@$bundlePath;type=application/javascript+module" `
        "https://api.cloudflare.com/client/v4/accounts/$AccountId/workers/scripts/$WorkerName"
    $upload = $uploadRaw | ConvertFrom-Json
    if (-not $upload.success) {
        throw "Worker upload failed: $uploadRaw"
    }

    $subdomainResponse = Invoke-RestMethod `
        -Headers @{ Authorization = "Bearer $token"; "Content-Type" = "application/json" } `
        -Uri "https://api.cloudflare.com/client/v4/accounts/$AccountId/workers/scripts/$WorkerName/subdomain" `
        -Method Post `
        -Body (@{ enabled = $true } | ConvertTo-Json)

    $workerUrl = "https://$WorkerName.suyee88.workers.dev"
    [pscustomobject]@{
        worker = $WorkerName
        worker_url = $workerUrl
        upload_success = $upload.success
        subdomain_enabled = $subdomainResponse.result.enabled
        previews_enabled = $subdomainResponse.result.previews_enabled
    } | ConvertTo-Json -Compress
}
finally {
    Remove-Item -LiteralPath $metadataPath -ErrorAction SilentlyContinue
    Pop-Location
}
