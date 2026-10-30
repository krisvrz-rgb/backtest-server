# Watchdog voor de servers op CrushMini.
# Kijkt of elke server antwoordt op zijn poort (TCP-verbinding naar 127.0.0.1). Antwoordt er een niet, dan wordt
# hij verborgen gestart met zijn eigen, bestaande startcommando. Een server die antwoordt wordt nooit gestopt of herstart.
# Na een start wacht de watchdog 3 minuten voor hij dezelfde server opnieuw probeert (de server moet kunnen opstarten).
# Draait elke 2 minuten via de geplande taak "Server Watchdog" (zie install-autostart.cmd).

$ErrorActionPreference = 'Continue'
$dir = 'C:\watchdog'
$log = Join-Path $dir 'watchdog.log'
$stateFile = Join-Path $dir 'watchdog-state.json'
$cooldownMinutes = 3

# poort, naam, programma, argumenten
$servers = @(
    @{ Port = 5000; Name = 'Portfolio server';    Exe = 'wscript.exe';  Args = '"C:\Users\krisv\HA\tools\pf-stil.vbs" -Actie start -Stil' },
    @{ Port = 5055; Name = 'AI usage server';     Exe = 'schtasks.exe'; Args = '/run /tn "AI usage server"' },
    @{ Port = 5056; Name = 'Gemini usage server'; Exe = 'schtasks.exe'; Args = '/run /tn "Gemini usage server"' },
    @{ Port = 8000; Name = 'AI server';           Exe = 'schtasks.exe'; Args = '/run /tn "AI server"' },
    @{ Port = 9000; Name = 'Backtest server';     Exe = 'cmd.exe';      Args = '/c "C:\backtest-server\start-backtest-server.cmd"' }
)

New-Item -ItemType Directory -Force -Path $dir | Out-Null

function Write-Log($text) {
    Add-Content -Path $log -Value ("{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $text)
}

function Test-Port($port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync('127.0.0.1', $port)
        return ($task.Wait(2000) -and $client.Connected)
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

$state = @{}
if (Test-Path $stateFile) {
    try { (Get-Content $stateFile -Raw | ConvertFrom-Json).PSObject.Properties | ForEach-Object { $state[$_.Name] = $_.Value } } catch { }
}

foreach ($s in $servers) {
    $key = [string]$s.Port
    if (Test-Port $s.Port) { continue }
    if ($state.ContainsKey($key)) {
        $last = [datetime]::Parse($state[$key])
        if ((Get-Date) - $last -lt [timespan]::FromMinutes($cooldownMinutes)) { continue }
    }
    try {
        Start-Process -FilePath $s.Exe -ArgumentList $s.Args -WindowStyle Hidden
        Write-Log ("{0} (poort {1}) antwoordde niet: gestart" -f $s.Name, $s.Port)
    } catch {
        Write-Log ("{0} (poort {1}) antwoordde niet, starten mislukt: {2}" -f $s.Name, $s.Port, $_.Exception.Message)
    }
    $state[$key] = (Get-Date).ToString('o')
}

$state | ConvertTo-Json | Set-Content -Path $stateFile

# Log klein houden: alleen de laatste 1000 regels.
if ((Test-Path $log) -and (Get-Item $log).Length -gt 200KB) {
    (Get-Content $log -Tail 1000) | Set-Content $log
}
