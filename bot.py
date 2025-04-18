import MetaTrader5 as mt5
import math
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# USER CONFIG
# -------------------------------------------------------------------------
SYMBOL = "GBPUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
DATE_DEBUT = datetime(2022, 4, 30)
DATE_FIN = datetime(2025, 2, 13)

INITIAL_CAPITAL = 100000
RISK_PERCENT = 0.01        # 1% risk per trade
RISK_REWARD_RATIO = 3.0    # for final TP after partial

# 1. Trailing Stop in Pips:
TRAILING_STOP_PIPS = 20    # how many pips behind price to keep SL
PIP_VALUE = 0.0001         # for GBPUSD, 1 pip = 0.0001

# 2. Partial Close at 2R + Move SL to Break Even
USE_PARTIAL_CLOSE = True
PARTIAL_CLOSE_R = 2.0      # close half at 2R
PARTIAL_CLOSE_FRACTION = 0.5  # close 50% of position

# LIVE Trading Toggle
LIVE_TRADING = True  # Set to True to enable live trading logic

# -------------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# -------------------------------------------------------------------------
def initialize_mt5(max_attempts=3):
    for attempt in range(max_attempts):
        if mt5.initialize():
            print("MetaTrader 5 initialized successfully.")
            return True
        else:
            print(f"Retrying MT5 initialization... Attempt {attempt + 1}/{max_attempts}")
            time.sleep(2)
    raise ConnectionError("Failed to initialize MT5 after multiple attempts.")

def shutdown_mt5():
    mt5.shutdown()
    print("MetaTrader 5 shutdown completed.")

def fetch_mt5_data(symbol, timeframe, start, end):
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        print(f"No data retrieved for {symbol} in the given date range.")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['Datetime'] = pd.to_datetime(df['time'], unit='s')
    df.rename(
        columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                 'close': 'Close', 'tick_volume': 'Volume'},
        inplace=True)
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df

# -------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------------------------------------------------
def add_engulfing_signals(df):
    df['Bullish_Engulfing'] = (
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Open'] < df['Close'].shift(1)) &
        (df['Close'] > df['Open'].shift(1))
    )
    df['Bearish_Engulfing'] = (
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Open'] > df['Close'].shift(1)) &
        (df['Close'] < df['Open'].shift(1))
    )
    return df

def add_pivot(df):
    df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    return df

def add_volume_multiplier(df, window=10):
    df['Avg_Volume'] = df['Volume'].shift(1).rolling(window).mean()
    df['Vol_Multiplier'] = df['Volume'] / df['Avg_Volume']
    return df

def define_signals(df):
    df['Buy_Signal'] = (
        df['Bullish_Engulfing'] &
        (df['Vol_Multiplier'] >= 1.48) &
        (df['Close'] > df['Pivot'])
    )
    df['Sell_Signal'] = (
        df['Bearish_Engulfing'] &
        (df['Vol_Multiplier'] >= 1.60) &
        (df['Close'] < df['Pivot'])
    )
    return df

def create_target(df):
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    return df

# -------------------------------------------------------------------------
# 3. LSTM MODEL
# -------------------------------------------------------------------------
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(units=30, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------------------------------------------------
# 4. BACKTEST ROUTINE
# -------------------------------------------------------------------------
def backtest_strategy(df_test, y_test, y_pred_class):
    """
    1) 1% risk per trade
    2) Hard TP at RISK_REWARD_RATIO
    3) Trailing Stop in pips from the H/L
    4) Partial close at 2R => close half position, SL to break even
    """

    capital = INITIAL_CAPITAL
    position = 0     # 0 = flat, 1 = long, -1 = short
    equity_curve = []

    df_test = df_test.copy().reset_index(drop=True)
    df_test['Actual'] = y_test.values
    df_test['Prediction'] = y_pred_class

    # Variables to track open trades
    entry_price = None
    sl = None
    tp = None
    position_size = 0.0
    initial_risk = 0.0

    # For partial close
    partial_close_done = False  # once we do partial, set True

    # Track highest/lowest prices reached while in position for trailing
    highest_price = None
    lowest_price = None

    for i in range(len(df_test)):

        # =========== NO POSITION ==============
        if position == 0:
            partial_close_done = False
            # Check BUY
            if df_test.loc[i, 'Buy_Signal'] and (df_test.loc[i, 'Prediction'] == 1):
                sl = df_test['Low'].iloc[i]
                entry_price = df_test['Close'].iloc[i]
                if sl >= entry_price:
                    continue  # invalid

                risk_amount = capital * RISK_PERCENT
                initial_risk = entry_price - sl
                position_size = risk_amount / initial_risk

                # final take profit at R:R ratio
                tp = entry_price + RISK_REWARD_RATIO * initial_risk

                highest_price = entry_price  # start tracking for trailing
                position = 1
                # Debug
                print(f"BUY at {entry_price:.5f}, SL={sl:.5f}, TP={tp:.5f}, time={df_test['Datetime'].iloc[i]}")

            # Check SELL
            elif df_test.loc[i, 'Sell_Signal'] and (df_test.loc[i, 'Prediction'] == 0):
                sl = df_test['High'].iloc[i]
                entry_price = df_test['Close'].iloc[i]
                if sl <= entry_price:
                    continue

                risk_amount = capital * RISK_PERCENT
                initial_risk = sl - entry_price
                position_size = risk_amount / initial_risk

                tp = entry_price - RISK_REWARD_RATIO * initial_risk

                lowest_price = entry_price
                position = -1
                # Debug
                print(f"SELL at {entry_price:.5f}, SL={sl:.5f}, TP={tp:.5f}, time={df_test['Datetime'].iloc[i]}")

        # =========== LONG POSITION ==============
        elif position == 1:
            current_price = df_test['Close'].iloc[i]
            current_high = df_test['High'].iloc[i]
            current_low = df_test['Low'].iloc[i]

            # Update highest price
            if highest_price is None:
                highest_price = current_high
            else:
                highest_price = max(highest_price, current_high)

            # 1) Trailing Stop in pips from the highest price
            # If highest_price - TRAILING_STOP_PIPS * PIP_VALUE is above the current SL => move SL
            trailing_candidate = highest_price - (TRAILING_STOP_PIPS * PIP_VALUE)
            if trailing_candidate > sl:
                sl = trailing_candidate
                # print(f"[Long] Updated trailing SL to {sl:.5f}")

            # 2) Check partial close at 2R
            if USE_PARTIAL_CLOSE and not partial_close_done:
                # If the current price is >= entry + 2 * initial_risk => partial close
                if (current_price - entry_price) >= (PARTIAL_CLOSE_R * initial_risk):
                    # Realize half the position
                    half_position_size = position_size * PARTIAL_CLOSE_FRACTION
                    # The profit on that portion = (current_price - entry_price)* half_position_size
                    partial_profit = (current_price - entry_price) * half_position_size
                    capital += partial_profit
                    # Reduce the position size
                    position_size -= half_position_size
                    # Move SL to break even
                    sl = entry_price
                    partial_close_done = True

                    print(f"[Long] Partial Close at 2R. Profit={partial_profit:.2f}, Remaining Size={position_size:.4f}, SL->BE")

            # 3) Check Stop Loss or Final Take Profit
            # if Low < SL => exit
            if current_low <= sl:
                exit_price = sl
                profit = (exit_price - entry_price) * position_size
                capital += profit
                position = 0
                print(f"STOP LOSS LONG at {exit_price:.5f}, P/L={profit:.2f}, time={df_test['Datetime'].iloc[i]}")

            elif current_high >= tp:
                exit_price = tp
                profit = (exit_price - entry_price) * position_size
                capital += profit
                position = 0
                print(f"TAKE PROFIT LONG at {exit_price:.5f}, P/L={profit:.2f}, time={df_test['Datetime'].iloc[i]}")

        # =========== SHORT POSITION ==============
        elif position == -1:
            current_price = df_test['Close'].iloc[i]
            current_high = df_test['High'].iloc[i]
            current_low = df_test['Low'].iloc[i]

            # Update lowest price
            if lowest_price is None:
                lowest_price = current_low
            else:
                lowest_price = min(lowest_price, current_low)

            # 1) Trailing Stop in pips from the lowest price
            trailing_candidate = lowest_price + (TRAILING_STOP_PIPS * PIP_VALUE)
            if trailing_candidate < sl:
                sl = trailing_candidate
                # print(f"[Short] Updated trailing SL to {sl:.5f}")

            # 2) Check partial close at 2R
            if USE_PARTIAL_CLOSE and not partial_close_done:
                # If entry_price - current_price >= 2* initial_risk
                if (entry_price - current_price) >= (PARTIAL_CLOSE_R * initial_risk):
                    half_position_size = position_size * PARTIAL_CLOSE_FRACTION
                    partial_profit = (entry_price - current_price) * half_position_size
                    capital += partial_profit
                    position_size -= half_position_size
                    # Move SL to break even
                    sl = entry_price
                    partial_close_done = True

                    print(f"[Short] Partial Close at 2R. Profit={partial_profit:.2f}, Remaining Size={position_size:.4f}, SL->BE")

            # 3) Check Stop Loss or Final Take Profit
            if current_high >= sl:
                exit_price = sl
                profit = (entry_price - exit_price) * position_size
                capital += profit
                position = 0
                print(f"STOP LOSS SHORT at {exit_price:.5f}, P/L={profit:.2f}, time={df_test['Datetime'].iloc[i]}")

            elif current_low <= tp:
                exit_price = tp
                profit = (entry_price - exit_price) * position_size
                capital += profit
                position = 0
                print(f"TAKE PROFIT SHORT at {exit_price:.5f}, P/L={profit:.2f}, time={df_test['Datetime'].iloc[i]}")

        equity_curve.append(capital)

    # Final stats
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_drawdown = np.max(drawdown) * 100

    print(f"\nBacktest Final Capital: {capital:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    return capital, equity_curve

# -------------------------------------------------------------------------
# 5. LIVE TRADING (TRAILING STOP + PARTIAL AT 2R)
# -------------------------------------------------------------------------
def live_trading_loop(symbol, model, scaler, features):
    """
    Demonstrates:
    1) Trailing stop in pips
    2) Partial close at 2R => close half, move SL to BE

    In practice, you need to store more data externally (like entry_price, initial_risk, etc.)
    so you can track partial closes accurately. This code uses a simple approach:
    - We track if we have an open position (1 or -1).
    - We handle partial close logic if not yet done.
    - We update SL by sending modify orders.
    """
    print("\nStarting LIVE trading loop (Press CTRL + C to exit).")

    open_position_type = 0    # 0 = no position, 1 = long, -1 = short
    position_ticket = None
    partial_close_done = False

    # We'll store some references from the moment we open a trade
    live_entry_price = None
    live_sl = None
    live_tp = None
    lot_size = 0.0
    initial_risk = 0.0

    while True:
        now = datetime.utcnow()
        df_live = fetch_mt5_data(symbol, TIMEFRAME, now - pd.Timedelta(days=3), now)
        if df_live.empty or len(df_live) < 30:
            print("No/insufficient data. Retrying in 60s...")
            time.sleep(60)
            continue

        df_live.sort_values('Datetime', inplace=True)
        df_live.reset_index(drop=True, inplace=True)

        # Feature engineering
        df_live = add_engulfing_signals(df_live)
        df_live = add_pivot(df_live)
        df_live = add_volume_multiplier(df_live)
        df_live = define_signals(df_live)
        df_live.dropna(inplace=True)

        # Last row
        last_row = df_live.iloc[-1]
        X_live = last_row[features].values.reshape(1, -1)
        X_live_scaled = scaler.transform(X_live)
        X_live_scaled = X_live_scaled.reshape((1, 1, len(features)))

        pred_proba = model.predict(X_live_scaled)
        pred_class = (pred_proba > 0.5).astype(int)[0][0]

        buy_signal = bool(last_row['Buy_Signal'])
        sell_signal = bool(last_row['Sell_Signal'])
        current_close = last_row['Close']
        current_high = last_row['High']
        current_low = last_row['Low']

        if open_position_type == 0:
            partial_close_done = False

            # Potential LONG
            if buy_signal and pred_class == 1:
                stop_loss_price = current_low
                if stop_loss_price >= current_close:
                    print("Skipping LONG due to invalid SL >= entry.")
                else:
                    # Calculate risk
                    risk_amount = INITIAL_CAPITAL * RISK_PERCENT
                    initial_risk = current_close - stop_loss_price
                    lot_size = risk_amount / initial_risk  # Must adapt to broker's real lot calculation

                    # Hard TP after partial => let's say 3R for final
                    take_profit_price = current_close + RISK_REWARD_RATIO * initial_risk

                    print(f"\n[LIVE] Sending BUY order on {symbol}")
                    ticket = send_market_order(
                        symbol=symbol,
                        lot=lot_size,
                        order_type=mt5.ORDER_TYPE_BUY,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                        comment="LSTM Live Buy"
                    )
                    if ticket:
                        open_position_type = 1
                        position_ticket = ticket
                        live_entry_price = current_close
                        live_sl = stop_loss_price
                        live_tp = take_profit_price
                        print(f"BUY Opened: Entry={live_entry_price}, SL={live_sl}, TP={live_tp}")

            # Potential SHORT
            elif sell_signal and pred_class == 0:
                stop_loss_price = current_high
                if stop_loss_price <= current_close:
                    print("Skipping SHORT due to invalid SL <= entry.")
                else:
                    risk_amount = INITIAL_CAPITAL * RISK_PERCENT
                    initial_risk = stop_loss_price - current_close
                    lot_size = risk_amount / initial_risk

                    take_profit_price = current_close - RISK_REWARD_RATIO * initial_risk

                    print(f"\n[LIVE] Sending SELL order on {symbol}")
                    ticket = send_market_order(
                        symbol=symbol,
                        lot=lot_size,
                        order_type=mt5.ORDER_TYPE_SELL,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                        comment="LSTM Live Sell"
                    )
                    if ticket:
                        open_position_type = -1
                        position_ticket = ticket
                        live_entry_price = current_close
                        live_sl = stop_loss_price
                        live_tp = take_profit_price
                        print(f"SELL Opened: Entry={live_entry_price}, SL={live_sl}, TP={live_tp}")

        else:
            # We have an open position. Let's check partial close + trailing
            positions = mt5.positions_get(symbol=symbol)
            # If the position closed or doesn't exist, reset
            if not positions:
                open_position_type = 0
                position_ticket = None
            else:
                # We'll assume we only have 1 position in this script
                pos = positions[0]
                current_price = (mt5.symbol_info_tick(symbol).bid
                                 if open_position_type == 1 else
                                 mt5.symbol_info_tick(symbol).ask)

                # 1) PARTIAL CLOSE at 2R
                if not partial_close_done:
                    # For a BUY, if price >= entry + 2*initial_risk => partial
                    # For a SELL, if entry - price >= 2*initial_risk => partial
                    if open_position_type == 1:
                        if (current_price - live_entry_price) >= 2.0 * initial_risk:
                            # Close half
                            volume_to_close = pos.volume * PARTIAL_CLOSE_FRACTION
                            partial_close_result = close_position_partially(pos.ticket, volume_to_close)
                            if partial_close_result:
                                # Move SL to break even
                                modify_request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "position": pos.ticket,
                                    "sl": live_entry_price,
                                    "tp": pos.tp
                                }
                                res_mod = mt5.order_send(modify_request)
                                if res_mod.retcode == mt5.TRADE_RETCODE_DONE:
                                    live_sl = live_entry_price
                                    partial_close_done = True
                                    print(f"[LIVE Long] Partial close done at 2R, SL => BE.")
                    else:
                        # SHORT
                        if (live_entry_price - current_price) >= 2.0 * initial_risk:
                            volume_to_close = pos.volume * PARTIAL_CLOSE_FRACTION
                            partial_close_result = close_position_partially(pos.ticket, volume_to_close)
                            if partial_close_result:
                                # Move SL to break even
                                modify_request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "position": pos.ticket,
                                    "sl": live_entry_price,
                                    "tp": pos.tp
                                }
                                res_mod = mt5.order_send(modify_request)
                                if res_mod.retcode == mt5.TRADE_RETCODE_DONE:
                                    live_sl = live_entry_price
                                    partial_close_done = True
                                    print(f"[LIVE Short] Partial close done at 2R, SL => BE.")

                # 2) Trailing Stop in pips from highest/lowest price
                # For a real trailing approach, you'd track “highest price so far” or “lowest price so far”.
                # As a simplified approach, we can keep trailing from the current price.
                # E.g., if open_position_type == 1 (buy), newSL = current_price - TRAILING_STOP_PIPS * PIP_VALUE
                # We only move it up if it is > live_sl.

                if open_position_type == 1:
                    new_sl_candidate = current_price - (TRAILING_STOP_PIPS * PIP_VALUE)
                    if new_sl_candidate > live_sl:
                        # Modify
                        modify_request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": pos.ticket,
                            "sl": new_sl_candidate,
                            "tp": pos.tp
                        }
                        mod_res = mt5.order_send(modify_request)
                        if mod_res.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"[LIVE Long] Trailing SL updated from {live_sl:.5f} to {new_sl_candidate:.5f}")
                            live_sl = new_sl_candidate

                else:
                    # open_position_type == -1
                    new_sl_candidate = current_price + (TRAILING_STOP_PIPS * PIP_VALUE)
                    if new_sl_candidate < live_sl:
                        modify_request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": pos.ticket,
                            "sl": new_sl_candidate,
                            "tp": pos.tp
                        }
                        mod_res = mt5.order_send(modify_request)
                        if mod_res.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"[LIVE Short] Trailing SL updated from {live_sl:.5f} to {new_sl_candidate:.5f}")
                            live_sl = new_sl_candidate

        print("Waiting 60s before next check...\n")
        time.sleep(60)

def send_market_order(symbol, lot, order_type, stop_loss=None, take_profit=None, comment=""):
    # Ensure symbol is available
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found!")
        return None
    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)

    # Fetch broker constraints
    min_vol  = symbol_info.volume_min
    max_vol  = symbol_info.volume_max
    step     = symbol_info.volume_step

    # Round down to nearest step and clamp
    lot = math.floor(lot / step) * step
    if lot < min_vol:
        lot = min_vol
    elif lot > max_vol:
        lot = max_vol

    # Get current price
    tick  = mt5.symbol_info_tick(symbol)
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

    # Round SL/TP to proper digits
    sl = round(stop_loss, symbol_info.digits) if stop_loss else None
    tp = round(take_profit, symbol_info.digits) if take_profit else None

    request = {
        "action":     mt5.TRADE_ACTION_DEAL,
        "symbol":     symbol,
        "volume":     float(lot),
        "type":       order_type,
        "price":      price,
        "sl":         sl,
        "tp":         tp,
        "deviation":  20,
        "magic":      123456,
        "comment":    comment,
        "type_time":  mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order Send Failed: retcode={result.retcode}, comment={result.comment}")
        return None

    print(f"Order placed successfully! Ticket: {result.order}, Volume: {lot}")
    return result.order


# -------------------------------------------------------------------------
# 6. MAIN EXECUTION
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # A) Initialize MT5
    initialize_mt5()

    # B) Fetch historical data
    print(f"\nFetching historical data for {SYMBOL}...")
    data = fetch_mt5_data(SYMBOL, TIMEFRAME, DATE_DEBUT, DATE_FIN)
    if data.empty:
        shutdown_mt5()
        raise ValueError("No historical data fetched. Exiting.")
    print(f"Data fetched: {len(data)} rows")

    # C) Feature engineering
    data = add_engulfing_signals(data)
    data = add_pivot(data)
    data = add_volume_multiplier(data)
    data = define_signals(data)
    data = create_target(data)
    data.dropna(inplace=True)
    print(f"Data after feature engineering & dropna: {len(data)} rows")

    # D) Train/Test Split
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Pivot', 'Vol_Multiplier']
    X = data[features]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    # E) Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Reshape for LSTM
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # F) Build & Train Model
    model = build_lstm_model(input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]))
    model.summary()

    history = model.fit(
        X_train_scaled, y_train,
        epochs=10,
        batch_size=200,
        validation_data=(X_test_scaled, y_test),
        verbose=1
    )

    # Plot training history (optional)
    plt.figure(figsize=(12, 5))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    plt.tight_layout()
    plt.show()

    # G) Evaluate
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    y_pred_proba = model.predict(X_test_scaled)
    threshold = 0.5
    y_pred_class = (y_pred_proba > threshold).astype(int).flatten()
    cm = confusion_matrix(y_test, y_pred_class)
    print("Confusion Matrix:\n", cm)
    report = classification_report(y_test, y_pred_class)
    print("Classification Report:\n", report)

    # H) BACKTEST
    print("\n----- BACKTEST RESULTS -----")
    df_test = data.iloc[-len(y_test):].copy()
    final_cap, eq_curve = backtest_strategy(df_test, y_test, y_pred_class)

    # Plot equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(eq_curve, label='Equity')
    plt.title('Backtest Equity Curve')
    plt.xlabel('Trade Steps')
    plt.ylabel('Capital')
    plt.legend()
    plt.show()

    # I) LIVE TRADING if desired
    if LIVE_TRADING:
        try:
            live_trading_loop(SYMBOL, model, scaler, features)
        except KeyboardInterrupt:
            print("Live trading loop interrupted.")
        finally:
            shutdown_mt5()
    else:
        shutdown_mt5()
