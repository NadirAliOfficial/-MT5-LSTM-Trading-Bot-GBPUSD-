# MT5 LSTM Trading Bot (GBPUSD)

A complete trading automation framework using MetaTrader 5, LSTM Neural Networks, and Python. The bot detects engulfing candlestick patterns, combines volume and pivot-based signals, and trades GBPUSD using a risk-managed strategy with trailing stop-loss, partial take-profit, and live execution support.

## Features

- 📊 Real-time data feed from MetaTrader 5
- 🤖 LSTM-based machine learning model for trade predictions
- 🧠 Feature engineering: Engulfing patterns, volume multipliers, pivot levels
- 🧪 Backtesting engine with R/R, trailing SL, partial TP
- 🔄 Live trading with MT5 execution API
- 📈 Equity curve and training visualization

## Requirements

- MetaTrader 5 (installed and connected to a broker)
- Python 3.8+
- Required Python packages:

```bash
pip install -r requirements.txt
```

**requirements.txt** example:
```
MetaTrader5
pandas
numpy
scikit-learn
matplotlib
tensorflow
pytz
```

## How to Run

1. **Ensure MetaTrader 5 is installed** and logged in.
2. Launch a **Python environment** (like Jupyter or terminal).
3. Run the script:
```bash
python trading_bot.py
```

## File Structure

- `trading_bot.py` – Main file containing:
  - Data fetching
  - Feature engineering
  - LSTM training
  - Backtesting
  - Live trading loop

## Notes

- Live trading mode can be toggled with `LIVE_TRADING = True`
- You must allow DLL imports and automated trading in MT5 settings.

## Contact

For help or customization, reach out on Telegram: [@NAKBlockDev](https://t.me/NAKBlockDev)
<!-- updated: 2026-05-13-03 -->
