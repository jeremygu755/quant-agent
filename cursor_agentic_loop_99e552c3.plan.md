---
name: Cursor Agentic Loop
overview: Standalone Python agent that loops every 30 minutes. Two-model architecture (GPT-4.1 Nano for cheap filtering, GPT-5.1 for deep decisions). Dynamic TP/SL set by the model per trade. Sell path runs before buy path every cycle.
todos:
  - id: scaffold
    content: Create project folder structure, requirements.txt, .env template
    status: pending
  - id: config
    content: Create config.py — load .env, define event_map, sector ETF map
    status: pending
  - id: news
    content: Build news.py — fetch and parse Yahoo Finance RSS with feedparser
    status: pending
  - id: llm
    content: "Build llm.py — two-model setup: GPT-4.1 Nano for classify/filter, GPT-5.1 for deep analysis + dynamic TP/SL"
    status: pending
  - id: quant
    content: Build quant.py — pull price/volume/volatility signals via yfinance, assemble full market context blob for LLM
    status: pending
  - id: state
    content: Build state.py — manage state.json for open positions, after_snapshots, and model-generated TP/SL/reason
    status: pending
  - id: sell
    content: Build sell.py — sell path (check TP/SL hit, ask LLM if reason still valid)
    status: pending
  - id: buy
    content: Build buy.py — buy path (Nano filters, GPT-5.1 decides, model sets TP/SL/reason)
    status: pending
  - id: alpaca
    content: Build alpaca_client.py — paper trade execution via Alpaca API
    status: pending
  - id: logger
    content: Build logger.py — write structured JSON log per cycle (decisions.jsonl)
    status: pending
  - id: main
    content: Build main.py — while True loop with time.sleep(1800), sell path first then buy path
    status: pending
isProject: false
---

# Cursor-Based Agentic Quant Loop (v2)

## Architecture

```mermaid
flowchart TD
    wake["main.py wakes up (every 30 min)"]

    subgraph sellPath [Sell Path — runs FIRST]
        loadState["Load open positions from state.json"]
        updateSnaps["Update after_snapshots via yfinance"]
        checkTP["Price >= model take_profit?"]
        checkSL["Price <= model stop_loss?"]
        checkReason["GPT-5.1: Is buy reason still valid?"]
        execSell["Execute SELL via Alpaca"]
    end

    subgraph buyPath [Buy Path — runs SECOND]
        rss["news.py: Fetch Yahoo Finance RSS"]
        nano["GPT-4.1 Nano: classify headline, extract ticker, sentiment"]
        actionable{"Actionable?"}
        quant["quant.py: full market snapshot"]
        gpt51["GPT-5.1: deep analysis + set dynamic TP/SL/reason"]
        execBuy["Execute BUY via Alpaca"]
        noop["No trade"]
    end

    logAll["logger.py: log everything to decisions.jsonl"]
    sleep["time.sleep(1800)"]

    wake --> loadState
    loadState --> updateSnaps
    updateSnaps --> checkTP
    checkTP -->|"yes"| execSell
    checkTP -->|"no"| checkSL
    checkSL -->|"yes"| execSell
    checkSL -->|"no"| checkReason
    checkReason -->|"reason dead"| execSell
    checkReason -->|"still valid"| rss

    wake --> rss
    rss --> nano
    nano --> actionable
    actionable -->|"yes"| quant
    actionable -->|"no"| noop
    quant --> gpt51
    gpt51 -->|"buy"| execBuy
    gpt51 -->|"no trade"| noop

    execSell --> logAll
    execBuy --> logAll
    noop --> logAll
    logAll --> sleep
    sleep --> wake
```

## File Structure

```
quant-agent/
├── main.py             # while True loop, sell-first then buy
├── news.py             # RSS fetch + parse headlines
├── llm.py              # Two-model: GPT-4.1 Nano (filter) + GPT-5.1 (decide)
├── quant.py            # price/volume/volatility signals via yfinance
├── state.py            # state.json management (positions, snapshots)
├── sell.py             # Sell path: TP/SL check + reason validation
├── buy.py              # Buy path: Nano filter → GPT-5.1 decision
├── alpaca_client.py    # Alpaca paper trade execution
├── logger.py           # Structured JSON log (decisions.jsonl)
├── config.py           # Load .env, event_map, sector ETF map
├── .env                # OPENAI_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY
└── requirements.txt    # openai, alpaca-trade-api, yfinance, feedparser
```

## Two-Model Strategy (~$1-3/month)

- **GPT-4.1 Nano** ($0.10/$0.40 per 1M tokens) — runs every cycle on every headline. Classifies event type, extracts ticker, rates sentiment. Cheap enough to call 48x/day on dozens of headlines.
- **GPT-5.1** ($1.25/$10.00 per 1M tokens) — only called when Nano says news is actionable, or to validate reasons for open positions. Maybe 2-5 calls/day.

## Dynamic TP/SL (Model-Generated, Not Hardcoded)

When GPT-5.1 recommends a buy, it also outputs:

```json
{
  "action": "buy",
  "ticker": "XOM",
  "reason": "OPEC supply shock, energy majors historically move 7-8%",
  "take_profit": 127.50,
  "take_profit_reason": "High vol environment, supply shocks sustain 7-8% moves",
  "stop_loss": 114.20,
  "stop_loss_reason": "Below pre-news price means thesis failed",
  "confidence": 0.75,
  "expected_timeframe": "1-3 days"
}
```

All values stored in `state.json` per position. No hardcoded percentages anywhere.

## Quant Signals Fed to LLM (assembled by `quant.py`)

For each candidate ticker, `yfinance` pulls and computes:
- Price at t-30min, t-1h, t-1d vs current price
- % return change at each interval
- Volume vs 20-day average (spike detection)
- 5-day vs 30-day volatility ratio (volatility spike)
- SPY % change (broad market context)
- Sector ETF % change (e.g., XLE for energy, XLK for tech)
- Momentum: is the move accelerating or slowing?
- Reversal signs: has direction changed?

## API Keys Required (.env)

```
OPENAI_API_KEY=sk-...
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
```

No other keys needed. Yahoo Finance RSS and yfinance are free/keyless.

## Dependencies

- `openai` — GPT-4.1 Nano + GPT-5.1
- `yfinance` — price/volume data
- `feedparser` — Yahoo Finance RSS
- `alpaca-trade-api` — paper trading
- `python-dotenv` — .env management


