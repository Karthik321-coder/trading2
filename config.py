"""
Advanced Trading Platform Configuration
All settings and constants in one place
"""
import os
from datetime import datetime, timedelta

class Config:
    # App Settings
    APP_TITLE = "🚀 Advanced AI Trading Prediction Platform"
    APP_ICON = "📈"
    VERSION = "1.0.0"
    
    # Indian Stock Symbols (Top 50 stocks)
    INDIAN_STOCKS = {
        'RELIANCE': 'RELIANCE.NS', 'TCS': 'TCS.NS', 'HDFCBANK': 'HDFCBANK.NS',
        'INFY': 'INFY.NS', 'ICICIBANK': 'ICICIBANK.NS', 'KOTAKBANK': 'KOTAKBANK.NS',
        'HINDUNILVR': 'HINDUNILVR.NS', 'LT': 'LT.NS', 'ITC': 'ITC.NS',
        'SBIN': 'SBIN.NS', 'BHARTIARTL': 'BHARTIARTL.NS', 'ASIANPAINT': 'ASIANPAINT.NS',
        'MARUTI': 'MARUTI.NS', 'BAJFINANCE': 'BAJFINANCE.NS', 'M&M': 'M&M.NS',
        'NESTLEIND': 'NESTLEIND.NS', 'WIPRO': 'WIPRO.NS', 'TATAMOTORS': 'TATAMOTORS.NS',
        'HCLTECH': 'HCLTECH.NS', 'TECHM': 'TECHM.NS', 'POWERGRID': 'POWERGRID.NS',
        'NTPC': 'NTPC.NS', 'JSWSTEEL': 'JSWSTEEL.NS', 'TATASTEEL': 'TATASTEEL.NS',
        'ADANIPORTS': 'ADANIPORTS.NS', 'COALINDIA': 'COALINDIA.NS', 'ONGC': 'ONGC.NS',
        'IOC': 'IOC.NS', 'GRASIM': 'GRASIM.NS', 'SUNPHARMA': 'SUNPHARMA.NS',
        'DRREDDY': 'DRREDDY.NS', 'CIPLA': 'CIPLA.NS', 'DIVISLAB': 'DIVISLAB.NS',
        'APOLLOHOSP': 'APOLLOHOSP.NS', 'BAJAJFINSV': 'BAJAJFINSV.NS', 'HDFCLIFE': 'HDFCLIFE.NS',
        'SBILIFE': 'SBILIFE.NS', 'BRITANNIA': 'BRITANNIA.NS', 'DABUR': 'DABUR.NS',
        'GODREJCP': 'GODREJCP.NS', 'MARICO': 'MARICO.NS', 'COLPAL': 'COLPAL.NS',
        'PIDILITIND': 'PIDILITIND.NS', 'BERGEPAINT': 'BERGEPAINT.NS', 'AKZONOBEL': 'AKZONOBEL.NS'
    }
    
    # Model Parameters
    LSTM_CONFIG = {
        'sequence_length': 60,
        'epochs': 50,
        'batch_size': 32,
        'validation_split': 0.2,
        'learning_rate': 0.001,
        'dropout_rate': 0.2
    }
    
    # Timeframes
    TIMEFRAMES = {
        '1 Hour': '1h', '1 Day': '1d', 
        '1 Week': '1wk', '1 Month': '1mo'
    }
    
    # Colors (Advanced Theme)
    COLORS = {
        'bullish': '#00ff88', 'bearish': '#ff4444', 'neutral': '#ffaa00',
        'buy_signal': '#00ff00', 'sell_signal': '#ff0000', 'hold_signal': '#ffff00',
        'support': '#4444ff', 'resistance': '#ff44ff', 'prediction': '#00ffff',
        'volume_high': '#ffffff', 'rsi_oversold': '#ff6666', 'rsi_overbought': '#66ff66'
    }
    
    # API Keys (Add your keys here)
    API_KEYS = {
        'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY', ''),
        'news_api': os.getenv('NEWS_API_KEY', ''),
        'twitter_api': os.getenv('TWITTER_API_KEY', '')
    }

# Global config instance
config = Config()
