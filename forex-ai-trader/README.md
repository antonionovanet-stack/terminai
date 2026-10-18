# Forex AI Trader

Software di trading automatico sul Forex che **ragiona sui trend di mercato** prima di operare.
Unisce tre livelli:

1. **Analisi quantitativa**: trend multi-timeframe (H4 per la direzione, H1 per l'ingresso), regime di
   mercato (trend / laterale / volatile / transizione), momentum, volatilità, supporti e resistenze.
2. **Ragionamento**: un ragionatore a regole che espone argomenti pro e contro con un punteggio, e
   **Claude** (Anthropic), che riceve l'analisi, il parere quantitativo, le posizioni aperte e l'esito dei
   trade recenti, e restituisce una decisione strutturata (tesi, fattori, rischi, condizione di
   invalidazione, stop e target).
3. **Gestione del rischio**: ha sempre l'ultima parola. Nessuna decisione, nemmeno di Claude, può
   aggirarla.

> ⚠️ **Avvertenza.** Il trading sul Forex con leva comporta un rischio elevato di perdere il capitale.
> Questo è un software sperimentale: non è una consulenza finanziaria e non garantisce profitti. I
> risultati su dati sintetici non dicono nulla sul mercato reale. Fai backtest su dati storici veri,
> poi usa a lungo il conto **demo**, e solo dopo valuta con prudenza il denaro reale.

Broker supportati: **MetaTrader 5** (predefinito), **OANDA** e un simulatore per i backtest.

## Architettura

```
 Broker (MetaTrader 5 / OANDA / simulatore)
        │ candele H1, prezzi, conto
        ▼
 analysis.py ──► MarketSnapshot (H1 + H4 ricampionato, regime, livelli)
        │
        ▼
 Ragionatore ── rules   : strategy.py      (punteggio con argomenti espliciti)
              ├ claude  : llm_reasoner.py  (Claude decide, con ripiego sulle regole)
              └ hybrid  : regole come filtro + Claude come giudice (predefinito)
        │ Decision (BUY/SELL/HOLD, confidenza, stop in ATR, R/R, tesi)
        ▼
 risk.py ──► OrderPlan oppure Rejection(motivi)
        │
        ▼
 engine.py ──► ordine con stop loss e take profit, stop a pareggio a +1R
        │
        ▼
 journal.py ──► diario JSONL con ogni ragionamento, ordine ed esito
```

### La strategia di base

- **In trend** (ADX ≥ 25 o in transizione, con trend H4 definito): cerca *pullback* verso la EMA20 H1
  nella direzione del trend H4, con H1 allineato, momentum MACD a favore ed RSI non in eccesso. Penalizza
  ingressi troppo estesi dalla media.
- **In laterale** (ADX < 20): ritorno alla media sulle bande di Bollinger con RSI in ipercomprato o
  ipervenduto, penalizzato se va contro il trend H4.
- **In regime volatile** (ATR oltre il 97° percentile, tipico delle notizie): resta fuori.

### Modalità di ragionamento

| `reasoner` | Comportamento |
| --- | --- |
| `rules` | Solo regole deterministiche. Gratuito e veloce, ideale per i backtest. |
| `claude` | Claude decide a ogni candela. Il più "pensante" e il più costoso. |
| `hybrid` (predefinito) | Le regole filtrano: Claude viene consultato solo quando c'è un possibile setup. In modalità `consensus` si opera solo se Claude conferma la stessa direzione; in `llm_final` decide Claude. |

Se Claude non è raggiungibile, va in rate limit o rifiuta la richiesta, il sistema ripiega sulle regole e
lo annota nel diario (`source: "rules(fallback: ...)"`).

### Controlli di rischio

- Rischio fisso per trade (predefinito 1% del capitale), con dimensione calcolata dalla distanza dello
  stop in ATR e convertita nella valuta del conto (gestisce anche le coppie JPY e i cross).
- Tetto di leva sul valore nozionale.
- Massimo di posizioni aperte, massimo per singola valuta (evita l'esposizione correlata), una posizione
  per strumento.
- Confidenza minima, rapporto rendimento/rischio minimo, filtro sullo spread (assoluto e rispetto allo
  stop).
- **Stop di giornata** (predefinito -3%): blocca nuovi ingressi fino al giorno dopo.
- **Kill switch** sul drawdown dal picco (predefinito -15%): blocca il trading fino al riavvio manuale.

## Installazione

Serve Python 3.11 o successivo.

### Con MetaTrader 5 (Windows)

Il pacchetto ufficiale `MetaTrader5` dialoga con il terminale MT5 aperto **sullo stesso PC Windows**,
quindi il programma va eseguito lì (non su un server Linux o su un Mac).

1. Installa e apri MetaTrader 5 ed entra nel tuo conto, **demo** per iniziare
   (File → Apri un conto → conto demo).
2. Attiva il pulsante **Algo Trading** nella barra degli strumenti (deve diventare verde) e, in
   Strumenti → Opzioni → Expert Advisor, consenti il trading algoritmico.
3. In "Market Watch" controlla come il tuo broker chiama le coppie: se sono `EURUSD.m` o `EURUSDm`,
   imposta `symbol_suffix` nella sezione `[mt5]` della configurazione.
4. Installa il programma in PowerShell:

```powershell
git clone https://github.com/<tuo-utente>/forex-ai-trader.git
cd forex-ai-trader
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,mt5]"
copy config.example.toml config.toml   # poi adatta i parametri
```

5. Credenziali:

```powershell
$env:ANTHROPIC_API_KEY = "..."     # per i ragionatori claude/hybrid
# Facoltative: se il terminale è già collegato al conto, non servono.
$env:MT5_LOGIN = "12345678"
$env:MT5_PASSWORD = "..."
$env:MT5_SERVER = "NomeBroker-Demo"
```

Il bot gestisce **solo le posizioni aperte da lui** (riconosciute dal `magic` number), così i trade
manuali sullo stesso conto restano intatti. Le unità calcolate dal gestore del rischio vengono convertite
in lotti arrotondando sempre per difetto, quindi non si rischia mai più del previsto. Se il conto
collegato è **reale**, il programma si rifiuta di partire senza `--live` e `FOREX_ALLOW_LIVE=yes`.

Gli orari delle candele MT5 sono quelli del server del broker (spesso UTC+2 o UTC+3). Imposta
`server_utc_offset_hours` perché le candele H4 e il limite di perdita giornaliero siano allineati a UTC.

### Con OANDA (qualsiasi sistema)

```bash
pip install -e ".[dev]"
export OANDA_API_TOKEN=...       # token API OANDA (conto demo gratuito su oanda.com)
export OANDA_ACCOUNT_ID=...      # es. 101-004-1234567-001
```

Poi usa `--broker oanda` oppure `broker = "oanda"` nella configurazione.

## Uso

### 1. Backtest

```bash
# Su dati sintetici (nessuna credenziale richiesta)
forex-trader backtest --bars 4000

# Su dati storici reali: CSV H1 con colonne time,open,high,low,close
forex-trader backtest --csv dati/EURUSD_H1.csv --instrument EUR_USD --spread 1.2 --journal journal/bt.jsonl

# Con Claude nel ciclo (consuma API: --max-llm-calls limita la spesa)
forex-trader --config config.toml backtest --csv dati/EURUSD_H1.csv --reasoner hybrid --max-llm-calls 30
```

Il backtest usa lo stesso motore, la stessa analisi e lo stesso gestore del rischio dell'operatività dal
vivo. Gli ordini vengono eseguiti all'apertura della barra successiva (nessun look-ahead), con spread
bid/ask; se in una barra vengono toccati sia lo stop sia il target si assume lo stop (ipotesi prudente).
Su un computer tipico servono circa 15 ms per barra.

### 2. Analisi istantanea, con il ragionamento in chiaro

```bash
forex-trader --config config.toml analyze --instrument EUR_USD   # dati dal tuo MT5 + ragionatore ibrido
forex-trader analyze --synthetic --reasoner rules       # prova offline
```

Esempio di output:

```
EUR_USD @ 1.08412 (2026-10-18 14:00 UTC)
  Regime: trending | Trend H4: up | Trend H1: up
  ...
Decisione [hybrid(claude:claude-opus-5-5)]: BUY (confidenza 0.64)
  Tesi: Pullback ordinato sulla EMA20 H1 in un trend H4 rialzista ...
  + Trend H4 rialzista ...
  - Resistenza a 1.0870 vicina al target ...
  Invalidazione: chiusura H1 sotto 1.0815 ...
```

### 3. Trading continuo

```bash
# Prima: solo ragionamento, nessun ordine (consigliato per qualche settimana)
forex-trader --config config.toml run --dry-run

# Conto DEMO MT5 con ordini veri
forex-trader --config config.toml run

# Conto REALE: richiede una doppia conferma esplicita (PowerShell)
$env:FOREX_ALLOW_LIVE = "yes"; forex-trader --config config.toml run --live
```

Il ciclo valuta ogni strumento una volta per candela H1 chiusa. Ogni ordine parte con stop loss e take
profit già agganciati lato broker, così le posizioni restano protette anche se il programma si ferma.

## Auto-evoluzione delle strategie

Il bot migliora da solo le sue regole di ingresso a mercato. Ogni strategia è descritta da un "DNA" di
26 parametri: pesi e soglie di trend, pullback, momentum, RSI, ritorno alla media, stop e target
(l'elenco con i limiti è `GENOME` in `forex_trader/evolution.py`).

Una sessione di evoluzione:

1. **Parte dal campione attuale** e crea una popolazione di varianti: mutazioni, incroci tra le migliori,
   individui casuali per esplorare e, se abiliti `llm_proposals`, **proposte di Claude**. Claude legge i
   risultati per periodo, fa una diagnosi (es. "troppi falsi segnali in laterale") e propone varianti
   guidate da ipotesi esplicite.
2. **Valuta ogni candidato** con un backtest walk-forward su più periodi consecutivi e su tutti gli
   strumenti. Il punteggio misura la *robustezza statistica* del vantaggio (t-statistic dei rendimenti per
   trade, media tra periodi meno la dispersione), con penalità per drawdown oltre il 10% e per pochi trade.
   Premia la costanza, non il colpo di fortuna.
3. **Seleziona e ripete** per più generazioni (selezione a torneo ed élite).
4. **Verifica sui dati mai visti.** L'ultimo 30% dello storico non viene mai usato durante la ricerca. Il
   migliore candidato diventa il nuovo campione solo se, su quei dati, insieme:
   - ha un vantaggio statistico positivo ed è migliore del campione di almeno `promote_margin`;
   - è in guadagno;
   - ha almeno `min_holdout_trades` trade;
   - ha un drawdown entro `max_holdout_drawdown_pct`.

   Altrimenti resta il campione attuale.

Ogni sessione, candidato, ipotesi di Claude e promozione resta nel registro `strategies/registry.json`.

**Cosa l'evoluzione non può toccare:** i limiti di rischio (`[risk]`): rischio per trade, leva, stop di
giornata, kill switch. Cambia *quando e come* si entra, mai *quanto* si rischia.

```bash
# Evoluzione manuale su dati storici (uno o più strumenti)
forex-trader evolve --csv EUR_USD=dati/eurusd_h1.csv --csv GBP_USD=dati/gbpusd_h1.csv

# Storico scaricato dal tuo MetaTrader 5, con 5 proposte ragionate di Claude
forex-trader --config config.toml evolve --claude 5

# Solo valutazione, senza cambiare il campione
forex-trader evolve --csv EUR_USD=dati/eurusd_h1.csv --no-promote

# Campione, genealogia, ipotesi e promozioni
forex-trader strategies --show
```

**In automatico:** con `run` e `auto = true`, il bot esegue l'evoluzione una volta a settimana nel
weekend, a mercato chiuso. Usa lo storico più recente del tuo broker e, se nasce un campione migliore,
lo adotta subito, senza riavvio. `backtest`, `analyze` e `run` usano sempre il campione del registro; con
`--default-strategy` usano invece i parametri della configurazione.

> **Limiti onesti.** Ottimizzare sul passato rischia sempre di "imparare il rumore" (overfitting). I
> controlli qui sopra riducono molto il rischio, ma non lo eliminano: un mercato che cambia regime può
> rendere obsoleta anche una strategia che ha superato la verifica. Per questo i limiti di rischio
> restano fissi e vale sempre la regola: prima demo, poi denaro reale.

## Il diario delle decisioni

`journal/decisions.jsonl` contiene una riga JSON per ogni evento: `decision` (con tesi, fattori, rischi e
fonte), `risk_rejected` (con i motivi), `order_placed`, `order_dry_run`, `stop_to_break_even`,
`trade_closed`. Le ultime decisioni e gli esiti dei trade sullo strumento vengono ripassati a Claude,
che può così tenere conto di cosa ha funzionato e cosa no.

## Test

```bash
pytest
```

I test coprono l'evoluzione (operatori genetici entro i limiti, backtest con cache identico a quello
normale, promozione e registro, pianificazione settimanale, proposte di Claude), indicatori, analisi, regole, dimensionamento e limiti di rischio, esecuzione simulata
(gap, spread, stop e target nella stessa barra), il connettore MetaTrader 5 con un terminale simulato
(simboli, lotti, ordini, blocco del conto reale, filtro per magic number, storico dei deal), il ragionatore Claude con un client simulato (mappatura,
limiti, rifiuti, errori di rete), la modalità ibrida, il motore in dry-run e la configurazione.

## Estendere

- **Esecuzione continua su Windows**: il PC deve restare acceso con MT5 aperto. Per un'operatività
  24/5 conviene un VPS Windows con il terminale MT5 installato.
- **Altro broker**: implementa il protocollo `Broker` in `forex_trader/brokers/base.py`.
- **Altra strategia**: qualsiasi oggetto con `decide(snapshot, context) -> Decision` è un ragionatore.
- **Più contesto per Claude**: aggiungi campi in `ClaudeReasoner.build_prompt` (calendario macro,
  notizie, correlazioni tra coppie).
