# MT5 LSTM Trading Bot — GBPUSD

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![MetaTrader5](https://img.shields.io/badge/MetaTrader-5-blueviolet?style=flat)](https://www.metatrader5.com/)
[![License](https://img.shields.io/github/license/NadirAliOfficial/-MT5-LSTM-Trading-Bot-GBPUSD-)](LICENSE)

A complete trading automation framework combining MetaTrader 5 live data, LSTM neural network predictions, and a full backtesting engine. The bot detects engulfing candlestick patterns, combines volume analysis with pivot-based signals, and executes GBPUSD trades with trailing stop-loss, partial take-profit, and real-time MT5 order management.

---

## Architecture

```
MT5 Terminal (Live Data)
        │
        ▼
Feature Engineering
  ├── Engulfing pattern detection
  ├── Volume multiplier signals
  └── Pivot level calculation
        │
        ▼
LSTM Model (TensorFlow/Keras)
  ├── Sequence length: 60 bars
  ├── Hidden layers: 2x LSTM + Dense
  └── Output: BUY / SELL / HOLD
        │
        ▼
Risk Manager
  ├── Position sizing
  ├── Trailing stop-loss
  └── Partial take-profit (scale-out)
        │
        ▼
MT5 Execution API
  └── Live orders / position management
```

---

## Features

- **Real-time data feed** from MetaTrader 5 via the official Python API
- **LSTM neural network** trained on engineered features — not raw OHLCV
- **Feature engineering** — engulfing patterns, volume multipliers, pivot high/low levels
- **Full backtesting engine** — replays historical bars with the same signal logic
- **Risk-managed execution** — trailing SL, partial TP (scale out at 1:1), max drawdown limit
- **Equity curve visualization** — plots training loss and live P&L after each session
- **Regime-aware** — `LIVE_TRADING` flag toggles between backtest and live execution

---

## Requirements

```
MetaTrader5
pandas
numpy
scikit-learn
matplotlib
tensorflow>=2.10
pytz
```

Install:

```bash
pip install -r requirements.txt
```

MetaTrader 5 must be installed, logged into a broker account, and have **DLL imports** and **Automated trading** enabled in Tools → Options.

---

## Quickstart

```bash
# 1. Set LIVE_TRADING = False to run in backtest mode first
# 2. Train the model and review equity curve
python first_Milestone.py

# 3. Run feature engineering and signal validation
python second_Milestone.py

# 4. Launch full pipeline (backtest → train → optionally live trade)
python third_final_Milestone.py
```

Set `LIVE_TRADING = True` in `third_final_Milestone.py` to switch to live execution.

---

## File Structure

| File | Purpose |
|---|---|
| `first_Milestone.py` | Data fetching, feature engineering, initial LSTM training |
| `second_Milestone.py` | Signal validation, pivot calculation, volume analysis |
| `third_final_Milestone.py` | Full pipeline — backtest engine + live trading loop |
| `requirements.txt` | Python dependencies |

---

## Strategy Logic

**Entry Conditions (BUY example):**
1. Bullish engulfing candle on M15/H1
2. Volume at least 1.5x 20-bar average
3. Price above nearest pivot support
4. LSTM prediction score > 0.65 threshold

**Exit Logic:**
- Partial close at 1:1 R/R (50% of position)
- Trailing stop activates after 1:1, trails at 1.5x ATR
- Hard SL at swing low below entry

---

## Notes

- Run in backtest mode first to validate performance on your broker's data before going live
- MT5 spread on GBPUSD varies by broker — factor this into your backtesting spread setting
- Recommended broker: one with tight spreads on GBPUSD (< 1.5 pips average)

---

## Developer

Built by **Nadir Ali Khan** — [TEAM NAK](https://github.com/NadirAliOfficial) | [Telegram](https://t.me/NAKBlockDev)
