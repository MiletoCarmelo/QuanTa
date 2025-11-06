import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List
from quanta.utils.ta import TAClient, RSI, MACD, SMA, EMA, BollingerBands, Indicator, Stochastic, ATR
from quanta.utils.trace import Trace, Candlesticks, Volume, Line


class ChartClient:
    """Client pour créer des graphiques financiers avec indicateurs techniques."""
    
    def __init__(self):
        self.ta_client = TAClient()
    
    def plot(self, df: pl.DataFrame, symbol: str = "Stock", 
             traces: Optional[List] = None, 
             indicators: Optional[List] = None,
             trades_df: Optional[pl.DataFrame] = None,
             max_bars: Optional[int] = None, 
             theme: str = 'professional',
             x_axis_type: str = 'row_nb',
             ):

        """
        Affiche un graphique avec chandeliers et indicateurs techniques.
        
        Args:
            df: DataFrame Polars avec colonnes 'datetime', 'open', 'high', 'low', 'close', 'volume'
            symbol: Nom du symbole pour le titre
            traces: Liste de traces [Candlesticks(), Volume(), Line('my_col')]
            indicators: Liste d'indicateurs techniques [SMA(50), RSI(), MACD()]
            trades_df: DataFrame avec les trades (timestamp, action, price)
            max_bars: Nombre maximum de barres à afficher (optionnel)
        """
        if df is None or len(df) == 0:
            print("Aucune donnée disponible")
            return
        
        # Vérifier les colonnes requises
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"Le DataFrame doit contenir les colonnes: {required_cols}")
            return
                
        # Limiter le nombre de barres si demandé
        if max_bars and len(df) > max_bars:
            df = df.tail(max_bars)
            
        # Configuration spécifique selon le type d'axe X
        if x_axis_type == 'row_nb':
            df = df.with_columns(pl.int_range(0, pl.len()).alias('x_index'))
            x_column = 'x_index'
        else:  # datetime (défaut)
            x_column = 'datetime'
            
        # configurer trades_df si fourni et x_axis_type est 'row_nb'
        if x_axis_type == 'row_nb' and trades_df is not None:
            trades_df = trades_df.join(
                df.select(['datetime', 'x_index']),
                left_on='timestamp',
                right_on='datetime',
                how='left'
            )
            
        print(f"Plotting {len(df)} bars for {symbol} with x_axis_type='{x_axis_type}'")
        # print(df)
        if trades_df is not None:
            print(f"With {len(trades_df)} trades")
        #     print(trades_df)
        
        # Par défaut
        if traces is None and indicators is None:
            traces = [Candlesticks(), Volume()]
            indicators = [SMA(50), SMA(200), RSI()]
        elif traces is None:
            traces = [Candlesticks()]
        elif indicators is None:
            indicators = []
        
        # Calculer tous les indicateurs
        df = self.ta_client.calculate_indicators(df, indicators)
        
        # Analyser les traces et indicateurs
        has_candlesticks = any(isinstance(p, Candlesticks) for p in traces)
        volume_trace = next((p for p in traces if isinstance(p, Volume)), None)
        overlay_lines = [p for p in traces if isinstance(p, Line)]
        
        # Indicateurs overlay (SMA, EMA, BB)
        overlay_indicators = [ind for ind in indicators if isinstance(ind, (SMA, EMA, BollingerBands))]
        
        # Indicateurs subplot (RSI, MACD, etc.)
        subplot_indicators = [ind for ind in indicators if isinstance(ind, (RSI, MACD, Stochastic, ATR))]
        
        # Créer les subplots
        rows = 1
        row_heights = [0.6]
        
        if volume_trace:
            rows += 1
            row_heights.append(0.15)
        
        for _ in subplot_indicators:
            rows += 1
            row_heights.append(0.125)
        
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights
        )
        
        # === ROW 1: Prix principal ===
        
        # Chandeliers
        if has_candlesticks:
            
            fig.add_trace(go.Candlestick(
                x=df[x_column],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC',
            ), row=1, col=1)
                  
        
        # Lignes custom (traces)
        for line in overlay_lines:
            if line.column in df.columns:                    
                fig.add_trace(go.Scatter(
                    x=df[x_column], 
                    y=df[line.column],
                    name=line.name,
                    line=dict(color=line.color, width=line.width) if line.color else dict(width=line.width),
                ), row=1, col=1)
        
        # Indicateurs overlay (SMA, EMA, BB)
        for ind in overlay_indicators:
            if isinstance(ind, SMA):
                if ind.name in df.columns:
                    color = 'blue' if ind.period == 50 else 'orange' if ind.period == 200 else None
                    fig.add_trace(go.Scatter(
                        x=df[x_column], y=df[ind.name],
                        name=ind.name, 
                        line=dict(color=color, width=1.5) if color else dict(width=1.5),
                    ), row=1, col=1)
            
            elif isinstance(ind, EMA):
                if ind.name in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df[x_column], y=df[ind.name],
                        name=ind.name, line=dict(width=1.5),
                    ), row=1, col=1)
            
            elif isinstance(ind, BollingerBands):
                if 'BB_upper' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df[x_column], y=df['BB_upper'],
                        name='BB Upper', line=dict(color='gray', width=1, dash='dash')
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=df[x_column], y=df['BB_middle'],
                        name='BB Middle', line=dict(color='gray', width=1),
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=df[x_column], y=df['BB_lower'],
                        name='BB Lower', line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
                    ), row=1, col=1)
        
        current_row = 2
        
        # === Trades ===
        if trades_df is not None and len(trades_df) > 0:
            # Préparer les données pour l'axe X
            x_col = 'x_value'
            if x_axis_type == 'row_nb' and 'x_index' in trades_df.columns:
                # Renommer x_index en x_value
                trades_df = trades_df.rename({'x_index': 'x_value'})
                x_col = 'x_value'
            else:
                trades_df = trades_df.with_columns([
                    pl.col('timestamp').alias('x_value')
                ])
                
            print(f"Debug: {len(trades_df)} trades après traitement")
            if len(trades_df) > 0:
                print(trades_df.select(['timestamp', 'x_value', 'action', 'price']).head())
        
            # ACHATS (triangle up vert)
            buy_trades = trades_df.filter(pl.col('action') == 'BUY')
            if len(buy_trades) > 0:
                fig.add_trace(go.Scatter(
                    x=buy_trades[x_col],
                    y=buy_trades['price'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=14,
                        color='green',
                        line=dict(color='darkgreen', width=2),
                        opacity=0.8
                    ),
                    name='Achat',
                    hovertemplate='<b>Achat</b><br>Prix: %{y:.2f}<br><extra></extra>'
                ), row=1, col=1)
            
            # VENTES (triangle down rouge)
            sell_trades = trades_df.filter(pl.col('action') == 'SELL')
            if len(sell_trades) > 0:
                fig.add_trace(go.Scatter(
                    x=sell_trades[x_col],
                    y=sell_trades['price'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=14,
                        color='red',
                        line=dict(color='darkred', width=2),
                        opacity=0.8
                    ),
                    name='Vente',
                    hovertemplate='<b>Vente</b><br>Prix: %{y:.2f}<br><extra></extra>'
                ), row=1, col=1)
        
        # === Volume ===
        if volume_trace:
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(df['close'], df['open'])]
            fig.add_trace(go.Bar(
                x=df[x_column], y=df['volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False,
            ), row=current_row, col=1)
            fig.update_yaxes(title_text="Volume", row=current_row, col=1)
            current_row += 1
        
        # === Indicateurs subplots ===
        for ind in subplot_indicators:
            if isinstance(ind, RSI):
                if ind.name in df.columns:  # ✅ BON (utilise le nom dynamique RSI14, RSI21, etc.)
                    fig.add_trace(go.Scatter(
                        x=df[x_column], y=df[ind.name],
                        name=ind.name, line=dict(color='purple', width=1.5),  # ← Utiliser ind.name
                    ), row=current_row, col=1)
                    
                    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                                opacity=0.5, row=current_row, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", 
                                opacity=0.5, row=current_row, col=1)
                    fig.update_yaxes(title_text=ind.name, row=current_row, col=1)  # ← Utiliser ind.name
                    current_row += 1
            
            elif isinstance(ind, MACD):
                if 'MACD' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df[x_column], y=df['MACD'],
                        name='MACD', line=dict(color='blue', width=1.5)
                    ), row=current_row, col=1)
                    fig.add_trace(go.Scatter(
                        x=df[x_column], y=df['MACD_signal'],
                        name='Signal', line=dict(color='orange', width=1.5)
                    ), row=current_row, col=1)
                    
                    colors = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
                    fig.add_trace(go.Bar(
                        x=df[x_column], y=df['MACD_hist'],
                        name='Histogram',
                        marker_color=colors,
                        marker_line_color=colors,
                        marker_line_width=0,
                        opacity=0.7,
                        showlegend=False
                    ), row=current_row, col=1)
                    fig.update_yaxes(title_text="MACD", row=current_row, col=1)
                    current_row += 1
            
            elif isinstance(ind, Stochastic):
                if 'STOCH_K' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df[x_column], y=df['STOCH_K'],
                        name='%K', line=dict(color='blue', width=1.5)
                    ), row=current_row, col=1)
                    fig.add_trace(go.Scatter(
                        x=df[x_column], y=df['STOCH_D'],
                        name='%D', line=dict(color='orange', width=1.5)
                    ), row=current_row, col=1)
                    
                    fig.add_hline(y=80, line_dash="dash", line_color="red", 
                                 opacity=0.5, row=current_row, col=1)
                    fig.add_hline(y=20, line_dash="dash", line_color="green", 
                                 opacity=0.5, row=current_row, col=1)
                    fig.update_yaxes(title_text="Stochastic", row=current_row, col=1)
                    current_row += 1
            
            elif isinstance(ind, ATR):
                if ind.name in df.columns:  # ✅ BON
                    fig.add_trace(go.Scatter(
                        x=df[x_column], y=df[ind.name],
                        name=ind.name, line=dict(color='orange', width=1.5),  # ← Utiliser ind.name
                    ), row=current_row, col=1)
                    fig.update_yaxes(title_text=ind.name, row=current_row, col=1)  # ← Utiliser ind.name
                    current_row += 1
        
        # Mise en forme professionnelle
        if theme == 'professional':
            # Style scientifique inspiré de ROOT/matplotlib
            template = 'plotly_white'
            grid_color = 'rgba(200, 200, 200, 0.3)'
            bg_color = 'white'
            paper_bg = '#f8f9fa'
            font_family = 'Computer Modern, serif'
            title_font_size = 18
            axis_font_size = 12
        elif theme == 'dark':
            template = 'plotly_dark'
            grid_color = 'rgba(100, 100, 100, 0.3)'
            bg_color = '#111111'
            paper_bg = '#0a0a0a'
            font_family = 'Computer Modern, monospace'
            title_font_size = 18
            axis_font_size = 12
        else:
            template = 'plotly'
            grid_color = 'rgba(200, 200, 200, 0.3)'
            bg_color = 'white'
            paper_bg = 'white'
            font_family = 'Arial, sans-serif'
            title_font_size = 16
            axis_font_size = 11
        
        fig.update_layout(
            title={
                'text': f'<b>{symbol}</b>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': title_font_size, 'family': font_family}
            },
            template=template,
            xaxis_rangeslider_visible=False,
            height=700,
            showlegend=True,
            hovermode='x unified',
            plot_bgcolor=bg_color,
            paper_bgcolor=paper_bg,
            font={'family': font_family, 'size': axis_font_size},
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'right',
                'x': 1,
                'bgcolor': 'rgba(255, 255, 255, 0.8)',
                'bordercolor': grid_color,
                'borderwidth': 1
            },
            margin=dict(l=60, r=40, t=80, b=50)
        )
        
        # Configuration de l'axe X selon le type
        if x_axis_type == 'row_nb':
            # Échantillonner les dates pour les labels
            step = max(1, len(df) // 15)  # ~15 labels maximum
            tick_indices = list(range(0, len(df), step))
            if tick_indices[-1] != len(df) - 1:
                tick_indices.append(len(df) - 1)
            
            # Convertir en listes Python pour l'accès par index
            x_index_list = df['x_index'].to_list()
            datetime_list = df['datetime'].to_list()
            
            # IMPORTANT : tick_indices contient les INDICES, pas les valeurs !
            tickvals = [x_index_list[i] for i in tick_indices]
            ticktext = [datetime_list[i].strftime('%Y-%m-%d') for i in tick_indices]
            
            # Appliquer à tous les subplots
            for i in range(1, rows + 1):
                fig.update_xaxes(
                    tickmode='array',
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickangle=-45,
                    row=i, col=1
                )

        # S'assurer que le rangeslider est bien désactivé
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        
        # Grille professionnelle pour tous les axes
        for i in range(1, rows + 1):
            xaxis_update = {
                'showgrid': True,
                'gridwidth': 1,
                'gridcolor': grid_color,
                'showline': True,
                'linewidth': 1,
                'linecolor': grid_color,
                'mirror': True,
                'ticks': 'outside',
                'tickwidth': 1,
                'tickcolor': grid_color,
                'ticklen': 5,
            }
            
            fig.update_xaxes(**xaxis_update, row=i, col=1)
            
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=grid_color,
                showline=True,
                linewidth=1,
                linecolor=grid_color,
                mirror=True,
                ticks='outside',
                tickwidth=1,
                tickcolor=grid_color,
                ticklen=5,
                row=i, col=1
            )
        
        fig.show()


# Exemple d'utilisation
if __name__ == "__main__":
    from yahoo_finance_client import YahooFinanceClient
    from ta_client import SMA, RSI, MACD, BollingerBands
    from traces import Candlesticks, Volume, Line
    
    # Récupérer les données
    yahoo_client = YahooFinanceClient()
    df = yahoo_client.get_price("AAPL", from_date="2023-01-01", to_date="2024-12-31")
    
    # Créer le graphique
    chart_client = ChartClient()
    
    # Style clean avec séparation traces/indicators
    traces = [
        Candlesticks(),
        Volume()
    ]
    
    indicators = [
        SMA(50),
        SMA(200),
        RSI()
    ]
    
    chart_client.plot(df, symbol="AAPL", 
                     traces=traces, 
                     indicators=indicators, 
                     max_bars=250)
    
    # Avec indicateur custom
    from ta_client import TAClient
    ta = TAClient()
    df = df.with_columns((df['high'] + df['low']).alias('my_custom'))
    
    traces = [
        Candlesticks(),
        Line('my_custom', name='Mon Indicateur', color='purple'),
        Volume()
    ]
    
    indicators = [SMA(50), MACD()]
    
    chart_client.plot(df, symbol="AAPL", traces=traces, indicators=indicators)
