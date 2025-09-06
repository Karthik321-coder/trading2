"""
📊 Ultra-Advanced Visualization Engine - Professional Charts
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class AdvancedVisualizationEngine:
    """Ultra-advanced visualization engine for financial charts"""
    
    def __init__(self):
        self.color_scheme = {
            'bull': '#00ff88',
            'bear': '#ff4444',
            'neutral': '#ffaa00',
            'volume': 'rgba(100, 149, 237, 0.4)',
            'ma_short': '#00aaff',
            'ma_long': '#ff8800',
            'background': '#0d1117'
        }
    
    def create_ultra_advanced_chart(self, data, predictions=None, 
                                   include_volume=True, include_indicators=True, 
                                   include_patterns=True, include_support_resistance=True, 
                                   title="Advanced Stock Analysis"):
        """Create ultra-advanced financial chart with all features"""
        
        try:
            if data is None or data.empty:
                return self._create_empty_chart("No data available")
            
            # Use last 200 points for performance
            df = data.tail(200).copy()
            
            # Create subplots
            if include_volume:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(title, 'Volume'),
                    row_width=[0.5, 0.5],
                    specs=[[{"secondary_y": False}],
                           [{"secondary_y": False}]]
                )
            else:
                fig = go.Figure()
            
            # Main candlestick chart
            candlestick = go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing=dict(line=dict(color=self.color_scheme['bull'], width=1)),
                decreasing=dict(line=dict(color=self.color_scheme['bear'], width=1)),
                showlegend=True
            )
            
            if include_volume:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)
            
            # Technical indicators
            if include_indicators:
                self._add_technical_indicators(fig, df, include_volume)
            
            # Support and resistance levels
            if include_support_resistance:
                self._add_support_resistance(fig, df)
            
            # Volume bars
            if include_volume and 'Volume' in df.columns:
                colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
                         for i in range(len(df))]
                
                volume_bars = go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.6,
                    showlegend=True
                )
                
                fig.add_trace(volume_bars, row=2, col=1)
            
            # Predictions overlay
            if predictions and len(predictions) > 0:
                self._add_predictions(fig, df, predictions)
            
            # Chart styling
            self._style_chart(fig, include_volume)
            
            return fig
            
        except Exception as e:
            print(f"Chart creation error: {e}")
            return self._create_empty_chart(f"Chart error: {str(e)}")
    
    def _add_technical_indicators(self, fig, df, include_volume=True):
        """Add technical indicators to chart"""
        try:
            row = 1 if include_volume else None
            col = 1 if include_volume else None
            
            # Moving Averages
            if 'SMA_20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['SMA_20'],
                    name='SMA 20',
                    line=dict(color=self.color_scheme['ma_short'], width=1),
                    opacity=0.8
                ), row=row, col=col)
            
            if 'SMA_50' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['SMA_50'],
                    name='SMA 50',
                    line=dict(color=self.color_scheme['ma_long'], width=1),
                    opacity=0.8
                ), row=row, col=col)
            
            # Bollinger Bands
            if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BB_Upper'],
                    name='BB Upper',
                    line=dict(color='rgba(173, 204, 255, 0.5)', width=1, dash='dash'),
                    showlegend=False
                ), row=row, col=col)
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BB_Lower'],
                    name='BB Lower',
                    line=dict(color='rgba(173, 204, 255, 0.5)', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(173, 204, 255, 0.1)',
                    showlegend=True
                ), row=row, col=col)
            
        except Exception as e:
            print(f"Indicators error: {e}")
    
    def _add_support_resistance(self, fig, df):
        """Add support and resistance levels"""
        try:
            # Calculate support and resistance
            recent_high = df['High'].tail(50).max()
            recent_low = df['Low'].tail(50).min()
            
            resistance_levels = [
                recent_high * 0.98,
                recent_high * 1.02
            ]
            
            support_levels = [
                recent_low * 0.98,
                recent_low * 1.02
            ]
            
            # Add resistance lines
            for i, level in enumerate(resistance_levels):
                fig.add_hline(
                    y=level,
                    line_dash='dot',
                    line_color='red',
                    opacity=0.6,
                    annotation_text=f'Resistance {i+1}',
                    annotation_position='top left'
                )
            
            # Add support lines
            for i, level in enumerate(support_levels):
                fig.add_hline(
                    y=level,
                    line_dash='dot',
                    line_color='green',
                    opacity=0.6,
                    annotation_text=f'Support {i+1}',
                    annotation_position='bottom left'
                )
                
        except Exception as e:
            print(f"Support/Resistance error: {e}")
    
    def _add_predictions(self, fig, df, predictions):
        """Add AI predictions to chart"""
        try:
            if not predictions:
                return
            
            # Create future dates
            last_date = df.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=len(predictions),
                freq='D'
            )
            
            # Extract prices from predictions
            if isinstance(predictions, list) and len(predictions) > 0:
                if isinstance(predictions[0], dict):
                    prices = [p.get('price', 0) for p in predictions]
                else:
                    prices = predictions
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=prices,
                    name='AI Prediction',
                    line=dict(color='purple', width=2, dash='dash'),
                    mode='lines+markers',
                    marker=dict(size=4, color='purple')
                ))
                
                # Add confidence intervals if available
                if isinstance(predictions[0], dict) and 'confidence_interval' in predictions[0]:
                    upper_bounds = [p['confidence_interval'].get('upper', p['price']) for p in predictions]
                    lower_bounds = [p['confidence_interval'].get('lower', p['price']) for p in predictions]
                    
                    # Upper bound
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=upper_bounds,
                        name='Confidence Upper',
                        line=dict(color='rgba(128, 0, 128, 0.3)', width=1),
                        showlegend=False
                    ))
                    
                    # Lower bound with fill
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=lower_bounds,
                        name='Confidence Band',
                        line=dict(color='rgba(128, 0, 128, 0.3)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(128, 0, 128, 0.1)',
                        showlegend=True
                    ))
        
        except Exception as e:
            print(f"Predictions error: {e}")
    
    def _style_chart(self, fig, include_volume=True):
        """Apply professional styling to chart"""
        
        layout_updates = {
            'template': 'plotly_dark',
            'height': 800 if include_volume else 800,
            'showlegend': True,
            'legend': dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1
            ),
            'hovermode': 'x unified',
            'dragmode': 'zoom',
            'paper_bgcolor': self.color_scheme['background'],
            'plot_bgcolor': self.color_scheme['background']
        }
        
        # X-axis styling
        xaxis_style = {
            'title': 'Date',
            'gridcolor': 'rgba(128, 128, 128, 0.2)',
            'showgrid': True,
            'zeroline': False,
            'rangeslider': {'visible': False}
        }
        
        # Y-axis styling  
        yaxis_style = {
            'title': 'Price (₹)',
            'gridcolor': 'rgba(128, 128, 128, 0.2)',
            'showgrid': True,
            'zeroline': False,
            'side': 'right'
        }
        
        if include_volume:
            layout_updates.update({
                'xaxis': xaxis_style,
                'yaxis': yaxis_style,
                'xaxis2': xaxis_style,
                'yaxis2': {
                    'title': 'Volume',
                    'gridcolor': 'rgba(128, 128, 128, 0.2)',
                    'showgrid': True,
                    'side': 'right'
                }
            })
        else:
            layout_updates.update({
                'xaxis': xaxis_style,
                'yaxis': yaxis_style
            })
        
        fig.update_layout(**layout_updates)
    
    def _create_empty_chart(self, message="No data"):
        """Create empty chart with message"""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5, y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=20, color='gray'),
            xref="paper", yref="paper"
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            paper_bgcolor='#0d1117',
            plot_bgcolor='#0d1117'
        )
        
        return fig
    
    def create_main_dashboard_chart(self, data, predictions=None, title="Stock Analysis"):
        """Create main dashboard chart (backward compatibility)"""
        return self.create_ultra_advanced_chart(
            data, predictions, 
            include_volume=True, 
            include_indicators=True,
            title=title
        )
    
    def create_prediction_chart(self, historical_data, predictions):
        """Create prediction-focused chart"""
        try:
            fig = go.Figure()
            
            # Historical prices (last 30 days)
            recent_data = historical_data.tail(30)
            
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['Close'],
                name='Historical Prices',
                line=dict(color='blue', width=2)
            ))
            
            # Predictions
            if predictions and len(predictions) > 0:
                last_date = historical_data.index[-1]
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=len(predictions),
                    freq='D'
                )
                
                if isinstance(predictions[0], dict):
                    prices = [p.get('price', 0) for p in predictions]
                else:
                    prices = predictions
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=prices,
                    name='AI Predictions',
                    line=dict(color='red', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title='Price Prediction Chart',
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                template='plotly_dark',
                height=400
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_chart(f"Prediction chart error: {str(e)}")
    
    def create_volatility_surface(self, data):
        """Create volatility surface visualization"""
        try:
            fig = go.Figure()
            
            # Calculate rolling volatility
            returns = data['Close'].pct_change().dropna()
            volatilities = []
            windows = [10, 20, 30, 60]
            
            for window in windows:
                vol = returns.rolling(window).std() * np.sqrt(252) * 100
                volatilities.append(vol)
            
            # Plot volatility curves
            colors = ['red', 'orange', 'yellow', 'green']
            for i, (window, vol) in enumerate(zip(windows, volatilities)):
                fig.add_trace(go.Scatter(
                    x=vol.index[-100:],  # Last 100 days
                    y=vol.iloc[-100:],
                    name=f'{window}D Vol',
                    line=dict(color=colors[i], width=2)
                ))
            
            fig.update_layout(
                title='Volatility Surface',
                xaxis_title='Date',
                yaxis_title='Volatility (%)',
                template='plotly_dark',
                height=400
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_chart(f"Volatility chart error: {str(e)}")
    
    def create_correlation_heatmap(self, correlation_matrix):
        """Create correlation heatmap"""
        try:
            if correlation_matrix is None or correlation_matrix.empty:
                return self._create_empty_chart("No correlation data")
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdYlBu',
                zmid=0,
                text=correlation_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={'size': 12},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Asset Correlation Matrix',
                template='plotly_dark',
                height=400,
                width=400
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_chart(f"Correlation chart error: {str(e)}")

# Create global instance
visualization_engine = AdvancedVisualizationEngine()
