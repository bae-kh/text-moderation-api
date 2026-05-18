$ErrorActionPreference = "Stop"

$HostUrl = "http://localhost:8000"
$LocustFile = "load_tests/locustfile.py"
$ResultDir = "load_test_results"

if (!(Test-Path $ResultDir)) {
    New-Item -ItemType Directory -Path $ResultDir | Out-Null
}

$Scenarios = @(
    @{ Users = 1; SpawnRate = 1; RunTime = "1m"; Name = "users_1" },
    @{ Users = 5; SpawnRate = 1; RunTime = "1m"; Name = "users_5" },
    @{ Users = 10; SpawnRate = 1; RunTime = "1m"; Name = "users_10" },
    @{ Users = 20; SpawnRate = 2; RunTime = "1m"; Name = "users_20" }
)

foreach ($Scenario in $Scenarios) {
    Write-Host "========================================"
    Write-Host "Running Locust scenario: $($Scenario.Name)"
    Write-Host "Users: $($Scenario.Users)"
    Write-Host "SpawnRate: $($Scenario.SpawnRate)"
    Write-Host "RunTime: $($Scenario.RunTime)"
    Write-Host "========================================"

    python -m locust `
        -f $LocustFile `
        --headless `
        --users $Scenario.Users `
        --spawn-rate $Scenario.SpawnRate `
        --run-time $Scenario.RunTime `
        -H $HostUrl `
        --csv "$ResultDir/$($Scenario.Name)"
}