# Avvio di Forex AI Trader su Windows con MetaTrader 5.
#
#   .\avvia.ps1             prima installazione + controllo + trading SIMULATO (nessun ordine)
#   .\avvia.ps1 -Ordini     invia ordini veri sul conto collegato a MT5 (il conto reale resta bloccato)
#   .\avvia.ps1 -Evolvi     esegue prima una sessione di evoluzione delle strategie
#   .\avvia.ps1 -Analisi    solo analisi immediata degli strumenti, poi esce

param(
    [switch]$Ordini,
    [switch]$Evolvi,
    [switch]$Analisi,
    [switch]$Aggiorna
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Passo($testo) { Write-Host "`n==> $testo" -ForegroundColor Cyan }

# 1. Python 3.11+
Passo "Controllo di Python"
$python = $null
foreach ($cand in @("py -3", "python")) {
    try {
        $ver = & ([scriptblock]::Create("$cand -c `"import sys; print(sys.version_info >= (3, 11))`"")) 2>$null
        if ($ver -eq "True") { $python = $cand; break }
    } catch { }
}
if (-not $python) {
    Write-Host "Serve Python 3.11 o successivo: installalo da https://www.python.org/downloads/ (spunta 'Add to PATH')." -ForegroundColor Red
    exit 1
}

# 2. Ambiente virtuale e dipendenze
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Passo "Creo l'ambiente virtuale (.venv)"
    & ([scriptblock]::Create("$python -m venv .venv"))
    $Aggiorna = $true
}
if ($Aggiorna) {
    Passo "Installo le dipendenze (MetaTrader5, Claude, pandas...)"
    & $venvPython -m pip install --upgrade pip | Out-Null
    & $venvPython -m pip install -e ".[mt5]"
    if ($LASTEXITCODE -ne 0) { Write-Host "Installazione fallita." -ForegroundColor Red; exit 1 }
}

# 3. Configurazione
if (-not (Test-Path "config.toml")) {
    Copy-Item "config.example.toml" "config.toml"
    Write-Host "Creato config.toml. Controlla in [mt5] il suffisso dei simboli del tuo broker (es. '.m')." -ForegroundColor Yellow
}

# 4. Credenziali Claude (facoltative)
if (-not $env:ANTHROPIC_API_KEY) {
    $chiave = Read-Host "Chiave API di Claude (INVIO per usare solo le regole, senza ragionamento LLM)"
    if ($chiave) { $env:ANTHROPIC_API_KEY = $chiave }
}

$trader = @("-m", "forex_trader", "--config", "config.toml")

# 5. Controllo della connessione a MT5 con un'analisi
Passo "Controllo della connessione a MetaTrader 5 (deve essere aperto, con Algo Trading attivo)"
& $venvPython @trader analyze --instrument EUR_USD
if ($LASTEXITCODE -ne 0) {
    Write-Host "MT5 non raggiungibile: apri MetaTrader 5, accedi al conto e attiva 'Algo Trading'." -ForegroundColor Red
    exit 1
}
if ($Analisi) { exit 0 }

# 6. Evoluzione facoltativa
if ($Evolvi) {
    Passo "Evoluzione delle strategie sullo storico del broker"
    & $venvPython @trader evolve
}

# 7. Trading continuo
if ($Ordini) {
    Passo "Avvio con ORDINI VERI sul conto collegato (Ctrl+C per fermare)"
    & $venvPython @trader run
} else {
    Passo "Avvio in modalità SIMULATA: ragiona e registra, nessun ordine (Ctrl+C per fermare)"
    Write-Host "Per inviare ordini sul conto demo: .\avvia.ps1 -Ordini" -ForegroundColor Yellow
    & $venvPython @trader run --dry-run
}
