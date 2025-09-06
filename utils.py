"""
Utility functions for the trading platform
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def format_currency(value, currency='₹'):
    """Format currency values"""
    return f"{currency}{value:,.2f}"

def format_percentage(value):
    """Format percentage values"""
    return f"{value:+.2f}%"

def calculate_returns(prices):
    """Calculate returns from price series"""
    return ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]) * 100

def get_market_status():
    """Check if market is open"""
    now = datetime.now()
    # NSE trading hours: 9:15 AM to 3:30 PM IST, Monday to Friday
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return "Market Closed (Weekend)"
    
    market_open = now.replace(hour=9, minute=15, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)
    
    if market_open <= now <= market_close:
        return "Market Open 🟢"
    else:
        return "Market Closed 🔴"

def create_progress_bar(current, total):
    """Create a progress bar for loading"""
    progress = current / total
    return st.progress(progress)

def validate_stock_symbol(symbol):
    """Validate if stock symbol exists"""
    from config import config
    return symbol in config.INDIAN_STOCKS

def generate_trading_signals(data, rsi_col='RSI', macd_col='MACD'):
    """Generate buy/sell/hold signals"""
    signals = []
    
    for i in range(len(data)):
        rsi = data.iloc[i][rsi_col]
        macd = data.iloc[i][macd_col]
        
        # Simple signal logic
        if rsi < 30 and macd > 0:
            signals.append('BUY')
        elif rsi > 70 and macd < 0:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    
    return signals

def calculate_risk_metrics(returns):
    """Calculate risk metrics"""
    return {
        'volatility': np.std(returns) * np.sqrt(252),  # Annualized
        'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252),
        'max_drawdown': np.min(returns),
        'var_95': np.percentile(returns, 5)
    }
