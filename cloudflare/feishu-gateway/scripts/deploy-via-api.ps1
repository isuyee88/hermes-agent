param(
    [string]$AccountId = "d1215a30b84b673ef0367010b0e78c10",
    [string]$WorkerName = "hermes-feishu-gateway",
    [string]$ModalInternalBaseUrl = "https://isuyee88--hermes-agent-web-handler.modal.run",
    [string]$ProxyUrl = "",
    [string]$CompatibilityDate = "2026-04-14",
    [string]$FeishuAppSuffix = ""
)

$ErrorActionPreference = "Stop"

function Get-EnvValue {
    param([string]$Name)
    foreach ($scope in @("Process", "User", "Machine")) {
        $value = [Environment]::GetEnvironmentVariable($Name, $scope)
        if ($value) {
            return $value
        }
    }
    return $null
}

function Get-EnvValueWithOrder {
    param(
        [string]$Name,
        [string[]]$Scopes = @("Process", "User", "Machine")
    )
    foreach ($scope in $Scopes) {
        $value = [Environment]::GetEnvironmentVariable($Name, $scope)
        if ($value) {
            return $value
        }
    }
    return $null
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

function Normalize-FeishuAppSuffix {
    param([string]$Value)
    $normalized = [string]$Value
    if ($null -eq $normalized) {
        $normalized = ""
    }
    $normalized = $normalized.Trim().ToLowerInvariant()
    switch ($normalized) {
        "" { return "" }
        "1" { return "" }
        "default" { return "" }
        "primary" { return "" }
        "2" { return "2" }
        "3" { return "3" }
        default { return "" }
    }
}

function Get-AvailableFeishuCredentialPairs {
    $pairs = [System.Collections.Generic.List[hashtable]]::new()

    foreach ($suffix in @("", "2", "3")) {
        $scopes = if ($suffix) { @("User", "Process", "Machine") } else { @("Process", "User", "Machine") }
        $credentialNames = @(
            @{
                app_id_name = "FEISHU_APP_ID$suffix"
                app_secret_name = "FEISHU_APP_SECRET$suffix"
            }
        )
        if ($suffix) {
            $credentialNames += @{
                app_id_name = "ID$suffix"
                app_secret_name = "token$suffix"
            }
        }

        foreach ($names in $credentialNames) {
            $appIdValue = Get-EnvValueWithOrder -Name $names.app_id_name -Scopes $scopes
            $appSecretValue = Get-EnvValueWithOrder -Name $names.app_secret_name -Scopes $scopes
            if (-not [string]::IsNullOrWhiteSpace($appIdValue) -and -not [string]::IsNullOrWhiteSpace($appSecretValue)) {
                $pairs.Add(@{
                    suffix = $suffix
                    app_id_name = $names.app_id_name
                    app_secret_name = $names.app_secret_name
                    app_id = $appIdValue
                    app_secret = $appSecretValue
                })
                break
            }
        }
    }

    return $pairs
}

function Resolve-FeishuCredentialPair {
    param([string]$PreferredSuffix = "")

    $explicitPreferredSuffix = [string]$PreferredSuffix
    $explicitEnvPreferredSuffix = [string](Get-EnvValue "HERMES_FEISHU_APP_SUFFIX")
    if ([string]::IsNullOrWhiteSpace($explicitEnvPreferredSuffix)) {
        $explicitEnvPreferredSuffix = [string](Get-EnvValue "FEISHU_APP_SUFFIX")
    }
    $hasExplicitPreference = -not [string]::IsNullOrWhiteSpace($explicitPreferredSuffix) -or -not [string]::IsNullOrWhiteSpace($explicitEnvPreferredSuffix)

    if (-not $hasExplicitPreference) {
        $availablePairs = Get-AvailableFeishuCredentialPairs
        if ($availablePairs.Count -gt 1) {
            $choices = $availablePairs | ForEach-Object {
                $label = if ([string]::IsNullOrWhiteSpace([string]$_.suffix)) { "default" } else { [string]$_.suffix }
                "$label => $($_.app_id_name)"
            }
            throw "Multiple Feishu credential pairs detected ($($choices -join ', ')). Set -FeishuAppSuffix, HERMES_FEISHU_APP_SUFFIX, or FEISHU_APP_SUFFIX explicitly."
        }
    }

    $orderedSuffixes = [System.Collections.Generic.List[string]]::new()
    foreach ($candidate in @(
        $PreferredSuffix,
        (Get-EnvValue "HERMES_FEISHU_APP_SUFFIX"),
        (Get-EnvValue "FEISHU_APP_SUFFIX"),
        "",
        "2",
        "3"
    )) {
        $suffix = Normalize-FeishuAppSuffix $candidate
        if (-not $orderedSuffixes.Contains($suffix)) {
            $orderedSuffixes.Add($suffix)
        }
    }

    foreach ($suffix in $orderedSuffixes) {
        $scopes = if ($suffix) { @("User", "Process", "Machine") } else { @("Process", "User", "Machine") }
        $credentialNames = @(
            @{
                app_id_name = "FEISHU_APP_ID$suffix"
                app_secret_name = "FEISHU_APP_SECRET$suffix"
            }
        )
        if ($suffix) {
            $credentialNames += @{
                app_id_name = "ID$suffix"
                app_secret_name = "token$suffix"
            }
        }

        foreach ($names in $credentialNames) {
            $appIdValue = Get-EnvValueWithOrder -Name $names.app_id_name -Scopes $scopes
            $appSecretValue = Get-EnvValueWithOrder -Name $names.app_secret_name -Scopes $scopes
            if (-not [string]::IsNullOrWhiteSpace($appIdValue) -and -not [string]::IsNullOrWhiteSpace($appSecretValue)) {
                return @{
                    suffix = $suffix
                    app_id_name = $names.app_id_name
                    app_secret_name = $names.app_secret_name
                    app_id = $appIdValue
                    app_secret = $appSecretValue
                }
            }
        }
    }

    return $null
}

function Resolve-OptionalFeishuSecret {
    param(
        [string]$BaseName,
        [string]$Suffix = ""
    )
    $scopes = if ($Suffix) { @("User", "Process", "Machine") } else { @("Process", "User", "Machine") }
    $candidates = @()
    if ($Suffix) {
        $candidates += "$BaseName$Suffix"
    }
    $candidates += $BaseName

    foreach ($name in $candidates) {
        $value = Get-EnvValueWithOrder -Name $name -Scopes $scopes
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            return $value
        }
    }

    return $null
}

function Add-OptionalSecretBinding {
    param(
        [System.Collections.ArrayList]$Bindings,
        [string]$Name,
        [string]$Value
    )
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return
    }
    [void]$Bindings.Add(@{
        name = $Name
        type = "secret_text"
        text = $Value
    })
}

function Add-OptionalPlainBinding {
    param(
        [System.Collections.ArrayList]$Bindings,
        [string]$Name,
        [string]$Value
    )
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return
    }
    [void]$Bindings.Add(@{
        name = $Name
        type = "plain_text"
        text = $Value
    })
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
$feishuPair = Resolve-FeishuCredentialPair -PreferredSuffix $FeishuAppSuffix
$appId = if ($feishuPair) { [string]$feishuPair.app_id } else { "" }
$appSecret = if ($feishuPair) { [string]$feishuPair.app_secret } else { "" }
$internalBearer = Get-EnvValue "HERMES_FEISHU_INTERNAL_BEARER_TOKEN"
$verificationToken = Resolve-OptionalFeishuSecret -BaseName "FEISHU_VERIFICATION_TOKEN" -Suffix $(if ($feishuPair) { [string]$feishuPair.suffix } else { "" })
$encryptKey = Resolve-OptionalFeishuSecret -BaseName "FEISHU_ENCRYPT_KEY" -Suffix $(if ($feishuPair) { [string]$feishuPair.suffix } else { "" })
$gatewayApiKey = Get-EnvValue "CLOUDFLARE_AI_GATEWAY_API_KEY"
$textPlainRouteName = Get-EnvValue "HERMES_CF_TEXT_PLAIN_ROUTE_NAME"
$textCodingRouteName = Get-EnvValue "HERMES_CF_TEXT_CODING_ROUTE_NAME"
$imageRouteName = Get-EnvValue "HERMES_CF_IMAGE_ROUTE_NAME"
$externalWaitEnabled = Get-EnvValue "HERMES_FEISHU_CF_EXTERNAL_WAIT_ENABLED"
$externalWaitMaxMs = Get-EnvValue "HERMES_FEISHU_CF_EXTERNAL_WAIT_MAX_MS"
$externalWaitPollMs = Get-EnvValue "HERMES_FEISHU_CF_EXTERNAL_WAIT_POLL_MS"
$requireObservedAsync = Get-EnvValue "HERMES_FEISHU_REQUIRE_OBSERVED_ASYNC_CAPABILITY"
$classifierEnabled = Get-EnvValue "HERMES_CF_REQUEST_CLASSIFIER_ENABLED"
$strictClassRouting = Get-EnvValue "HERMES_CF_STRICT_CLASS_ROUTE_ENFORCEMENT"
$imageGatewayEnabled = Get-EnvValue "HERMES_CF_IMAGE_GATEWAY_ENABLED"
$textRoutePolicyMode = Get-EnvValue "HERMES_CF_TEXT_ROUTE_POLICY_MODE"
$modelCatalogSyncEnabled = Get-EnvValue "HERMES_MODEL_CATALOG_SYNC_ENABLED"
$modelCatalogSyncProviders = Get-EnvValue "HERMES_MODEL_CATALOG_SYNC_PROVIDERS"
$modelCatalogRoutePublishEnabled = Get-EnvValue "HERMES_MODEL_CATALOG_ROUTE_PUBLISH_ENABLED"
$modelCatalogRouteMaxModels = Get-EnvValue "HERMES_MODEL_CATALOG_ROUTE_MAX_MODELS"
$modelCatalogMinHealthScore = Get-EnvValue "HERMES_MODEL_CATALOG_MIN_HEALTH_SCORE"
$modelCatalogMaxNegativeFeedbackCount = Get-EnvValue "HERMES_MODEL_CATALOG_MAX_NEGATIVE_FEEDBACK_COUNT"
$modelCatalogMaxAuthErrorCount = Get-EnvValue "HERMES_MODEL_CATALOG_MAX_AUTH_ERROR_COUNT"
$modelCatalogMaxModelNotFoundCount = Get-EnvValue "HERMES_MODEL_CATALOG_MAX_MODEL_NOT_FOUND_COUNT"
$modelCatalogMaxRateLimitCount = Get-EnvValue "HERMES_MODEL_CATALOG_MAX_RATE_LIMIT_COUNT"
$modelCatalogRateLimitCooldownSeconds = Get-EnvValue "HERMES_MODEL_CATALOG_RATE_LIMIT_COOLDOWN_SECONDS"
$dynamicRouteExecutionEnabled = Get-EnvValue "HERMES_CF_DYNAMIC_ROUTE_EXECUTION_ENABLED"
$providerRouteLimitsJson = Get-EnvValue "HERMES_CF_PROVIDER_ROUTE_LIMITS_JSON"
$gatewayProviderSlugsJson = Get-EnvValue "HERMES_CF_GATEWAY_PROVIDER_SLUGS_JSON"
$modelCatalogQueueEnabled = Get-EnvValue "HERMES_MODEL_CATALOG_QUEUE_ENABLED"
$modelCatalogQueueDelaySeconds = Get-EnvValue "HERMES_MODEL_CATALOG_QUEUE_DELAY_SECONDS"
$modelCatalogQueueRetryDelaySeconds = Get-EnvValue "HERMES_MODEL_CATALOG_QUEUE_RETRY_DELAY_SECONDS"
$modelCatalogQueueHardRetryDelaySeconds = Get-EnvValue "HERMES_MODEL_CATALOG_QUEUE_HARD_RETRY_DELAY_SECONDS"
$modelCatalogQueueLeaseSeconds = Get-EnvValue "HERMES_MODEL_CATALOG_QUEUE_LEASE_SECONDS"
$feishuKpiQueueEnabled = Get-EnvValue "HERMES_FEISHU_KPI_QUEUE_ENABLED"
$feishuKpiQueueDelaySeconds = Get-EnvValue "HERMES_FEISHU_KPI_QUEUE_DELAY_SECONDS"
$feishuKpiQueueRetryDelaySeconds = Get-EnvValue "HERMES_FEISHU_KPI_QUEUE_RETRY_DELAY_SECONDS"
$feishuKpiQueueHardRetryDelaySeconds = Get-EnvValue "HERMES_FEISHU_KPI_QUEUE_HARD_RETRY_DELAY_SECONDS"
$feishuKpiQueueLeaseSeconds = Get-EnvValue "HERMES_FEISHU_KPI_QUEUE_LEASE_SECONDS"
$feishuModelRegistryMirrorEnabled = Get-EnvValue "FEISHU_MODEL_REGISTRY_MIRROR_ENABLED"
$feishuRegistryMaxMutations = Get-EnvValue "FEISHU_MODEL_REGISTRY_MAX_MUTATIONS_PER_RUN"
$feishuRegistryStaleDeleteAfterSeconds = Get-EnvValue "FEISHU_MODEL_REGISTRY_STALE_DELETE_AFTER_SECONDS"
$feishuRegistryContinueDelaySeconds = Get-EnvValue "FEISHU_MODEL_REGISTRY_CONTINUE_DELAY_SECONDS"
$feishuBitableTableName = Get-EnvValue "FEISHU_BITABLE_TABLE_NAME"

if (-not $token) {
    throw "CLOUDFLARE_API_TOKEN is required."
}
if (-not $appId) {
    throw "Feishu app credentials are required. Set FEISHU_APP_ID/SECRET or FEISHU_APP_ID2/3 + FEISHU_APP_SECRET2/3."
}
if (-not $appSecret) {
    throw "Feishu app credentials are required. Set FEISHU_APP_ID/SECRET or FEISHU_APP_ID2/3 + FEISHU_APP_SECRET2/3."
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

    $bindings = [System.Collections.ArrayList]::new()
    [void]$bindings.Add(@{
        name = "FEISHU_AGENT_WORKFLOW"
        type = "workflow"
        workflow_name = "hermes-feishu-agent-workflow"
        class_name = "FeishuAgentWorkflow"
    })
    [void]$bindings.Add(@{
        name = "FEISHU_RECONCILE_QUEUE"
        type = "durable_object_namespace"
        class_name = "FeishuReconcileQueue"
    })
    [void]$bindings.Add(@{
        name = "SITE_PREFETCH_CACHE"
        type = "durable_object_namespace"
        class_name = "SitePrefetchCache"
    })
    [void]$bindings.Add(@{
        name = "BROWSER"
        type = "browser"
    })
    [void]$bindings.Add(@{
        name = "MODEL_CATALOG_DB"
        type = "d1"
        database_id = "783a29b3-47e5-41f7-aadd-3f284765ac79"
    })
    [void]$bindings.Add(@{
        name = "MODEL_CATALOG_QUEUE"
        type = "queue"
        queue_name = "hermes-model-catalog-heartbeat"
    })
    [void]$bindings.Add(@{
        name = "FEISHU_GATEWAY_ANALYTICS"
        type = "analytics_engine"
        dataset = "hermes_feishu_gateway_events"
    })
    [void]$bindings.Add(@{
        name = "FEISHU_GATEWAY_KPI_ROLLUPS"
        type = "analytics_engine"
        dataset = "hermes_feishu_kpi_rollups"
    })
    [void]$bindings.Add(@{
        name = "FEISHU_API_BASE"
        type = "plain_text"
        text = "https://open.feishu.cn"
    })
    [void]$bindings.Add(@{
        name = "FEISHU_ACK_REACTION_EMOJI"
        type = "plain_text"
        text = "OK"
    })
    [void]$bindings.Add(@{
        name = "MODAL_INTERNAL_BASE_URL"
        type = "plain_text"
        text = $ModalInternalBaseUrl
    })
    [void]$bindings.Add(@{
        name = "FEISHU_APP_ID"
        type = "secret_text"
        text = $appId
    })
    [void]$bindings.Add(@{
        name = "FEISHU_APP_SECRET"
        type = "secret_text"
        text = $appSecret
    })
    [void]$bindings.Add(@{
        name = "MODAL_INTERNAL_BEARER_TOKEN"
        type = "secret_text"
        text = $internalBearer
    })
    Add-OptionalSecretBinding -Bindings $bindings -Name "FEISHU_VERIFICATION_TOKEN" -Value $verificationToken
    Add-OptionalSecretBinding -Bindings $bindings -Name "FEISHU_ENCRYPT_KEY" -Value $encryptKey
    Add-OptionalSecretBinding -Bindings $bindings -Name "CLOUDFLARE_AI_GATEWAY_API_KEY" -Value $gatewayApiKey
    Add-OptionalSecretBinding -Bindings $bindings -Name "CLOUDFLARE_API_TOKEN" -Value $token
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_FEISHU_CF_EXTERNAL_WAIT_ENABLED" -Value $externalWaitEnabled
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_FEISHU_CF_EXTERNAL_WAIT_MAX_MS" -Value $externalWaitMaxMs
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_FEISHU_CF_EXTERNAL_WAIT_POLL_MS" -Value $externalWaitPollMs
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_FEISHU_REQUIRE_OBSERVED_ASYNC_CAPABILITY" -Value $requireObservedAsync
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_CF_REQUEST_CLASSIFIER_ENABLED" -Value $classifierEnabled
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_CF_STRICT_CLASS_ROUTE_ENFORCEMENT" -Value $strictClassRouting
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_CF_IMAGE_GATEWAY_ENABLED" -Value $imageGatewayEnabled
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_CF_TEXT_ROUTE_POLICY_MODE" -Value $textRoutePolicyMode
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_MODEL_CATALOG_SYNC_ENABLED" -Value $modelCatalogSyncEnabled
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_MODEL_CATALOG_SYNC_PROVIDERS" -Value $modelCatalogSyncProviders
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_MODEL_CATALOG_ROUTE_PUBLISH_ENABLED" -Value $modelCatalogRoutePublishEnabled
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_MODEL_CATALOG_ROUTE_MAX_MODELS" -Value $modelCatalogRouteMaxModels
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_MODEL_CATALOG_MIN_HEALTH_SCORE" -Value $modelCatalogMinHealthScore
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_MODEL_CATALOG_MAX_NEGATIVE_FEEDBACK_COUNT" -Value $modelCatalogMaxNegativeFeedbackCount
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_MODEL_CATALOG_MAX_AUTH_ERROR_COUNT" -Value $modelCatalogMaxAuthErrorCount
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_MODEL_CATALOG_MAX_MODEL_NOT_FOUND_COUNT" -Value $modelCatalogMaxModelNotFoundCount
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_MODEL_CATALOG_MAX_RATE_LIMIT_COUNT" -Value $modelCatalogMaxRateLimitCount
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_MODEL_CATALOG_RATE_LIMIT_COOLDOWN_SECONDS" -Value $modelCatalogRateLimitCooldownSeconds
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_CF_DYNAMIC_ROUTE_EXECUTION_ENABLED" -Value $dynamicRouteExecutionEnabled
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_CF_PROVIDER_ROUTE_LIMITS_JSON" -Value $providerRouteLimitsJson
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_CF_GATEWAY_PROVIDER_SLUGS_JSON" -Value $gatewayProviderSlugsJson
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_MODEL_CATALOG_QUEUE_ENABLED" -Value $modelCatalogQueueEnabled
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_MODEL_CATALOG_QUEUE_DELAY_SECONDS" -Value $modelCatalogQueueDelaySeconds
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_MODEL_CATALOG_QUEUE_RETRY_DELAY_SECONDS" -Value $modelCatalogQueueRetryDelaySeconds
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_MODEL_CATALOG_QUEUE_HARD_RETRY_DELAY_SECONDS" -Value $modelCatalogQueueHardRetryDelaySeconds
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_MODEL_CATALOG_QUEUE_LEASE_SECONDS" -Value $modelCatalogQueueLeaseSeconds
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_FEISHU_KPI_QUEUE_ENABLED" -Value $feishuKpiQueueEnabled
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_FEISHU_KPI_QUEUE_DELAY_SECONDS" -Value $feishuKpiQueueDelaySeconds
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_FEISHU_KPI_QUEUE_RETRY_DELAY_SECONDS" -Value $feishuKpiQueueRetryDelaySeconds
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_FEISHU_KPI_QUEUE_HARD_RETRY_DELAY_SECONDS" -Value $feishuKpiQueueHardRetryDelaySeconds
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_FEISHU_KPI_QUEUE_LEASE_SECONDS" -Value $feishuKpiQueueLeaseSeconds
    Add-OptionalPlainBinding -Bindings $bindings -Name "FEISHU_MODEL_REGISTRY_MIRROR_ENABLED" -Value $feishuModelRegistryMirrorEnabled
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_CF_TEXT_PLAIN_ROUTE_NAME" -Value $textPlainRouteName
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_CF_TEXT_CODING_ROUTE_NAME" -Value $textCodingRouteName
    Add-OptionalPlainBinding -Bindings $bindings -Name "HERMES_CF_IMAGE_ROUTE_NAME" -Value $imageRouteName
    Add-OptionalPlainBinding -Bindings $bindings -Name "FEISHU_BITABLE_TABLE_NAME" -Value $feishuBitableTableName
    Add-OptionalPlainBinding -Bindings $bindings -Name "FEISHU_MODEL_REGISTRY_MAX_MUTATIONS_PER_RUN" -Value $feishuRegistryMaxMutations
    Add-OptionalPlainBinding -Bindings $bindings -Name "FEISHU_MODEL_REGISTRY_STALE_DELETE_AFTER_SECONDS" -Value $feishuRegistryStaleDeleteAfterSeconds
    Add-OptionalPlainBinding -Bindings $bindings -Name "FEISHU_MODEL_REGISTRY_CONTINUE_DELAY_SECONDS" -Value $feishuRegistryContinueDelaySeconds

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
        bindings = $bindings
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
        feishu_app_id_env = if ($feishuPair) { $feishuPair.app_id_name } else { "" }
        feishu_app_secret_env = if ($feishuPair) { $feishuPair.app_secret_name } else { "" }
    } | ConvertTo-Json -Compress
}
finally {
    Remove-Item -LiteralPath $metadataPath -ErrorAction SilentlyContinue
    Pop-Location
}
