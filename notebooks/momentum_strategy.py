# %%
# set working directory :
import os
pwd = os.getcwd() + "/../"
os.chdir(pwd)


from quanta.clients.yfinance import YahooFinanceClient
from datetime import datetime, timedelta
import polars as pl
import numpy as np


# %%
tickers_map = {
    # Energy
    'CL': 'CL=F',   # WTI Crude Oil
    'NG': 'NG=F',   # Natural Gas
    'RB': 'RB=F',   # Gasoline
    'HO': 'HO=F',   # Heating Oil
    
    # Metals
    'GC': 'GC=F',   # Gold
    'SI': 'SI=F',   # Silver
    'HG': 'HG=F',   # Copper
    'PL': 'PL=F',   # Platinum
    
    # Agriculture
    'ZC': 'ZC=F',   # Corn
    'ZW': 'ZW=F',   # Wheat
    'ZS': 'ZS=F',   # Soybeans
    'KC': 'KC=F',   # Coffee
    'SB': 'SB=F',   # Sugar
    'CT': 'CT=F',   # Cotton
    
    # Livestock
    'LE': 'LE=F',   # Live Cattle
    'HE': 'HE=F',   # Lean Hogs
}

initial = 100_000_000
window_days = 30*120  # 12 months
from_date = datetime.now() - timedelta(days=window_days)
to_date = datetime.now() - timedelta(days=1)

timeframe = "1d"

portfolio_target_sigma = 0.43

# %%
def clean_price_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Nettoie automatiquement les données de prix invalides
    
    Enlève:
    - Prix négatifs ou nuls
    - Prix avec variation >50% en 1 période (erreurs de données)
    - Lignes avec valeurs manquantes
    """
    # 1. Enlève prix négatifs/nuls
    df = df.filter(
        (pl.col('open') > 0) & 
        (pl.col('high') > 0) & 
        (pl.col('low') > 0) & 
        (pl.col('close') > 0)
    )
    
    # 2. Enlève lignes avec NaN
    df = df.drop_nulls(subset=['open', 'high', 'low', 'close'])
    
    # 3. Enlève variations extrêmes (>50% en 1 période = probablement erreur)
    df = df.with_columns(
        (pl.col('close') / pl.col('close').shift(1))
        .over('symbol')
        .alias('price_ratio')
    )
    
    df = df.filter(
        (pl.col('price_ratio').is_null()) |  # Garde première ligne
        ((pl.col('price_ratio') > 0.5) & (pl.col('price_ratio') < 2.0))  # ±50%
    )
    
    df = df.drop('price_ratio')
    
    # 4. Vérifie cohérence OHLC (high >= low, close entre low et high)
    df = df.filter(
        (pl.col('high') >= pl.col('low')) &
        (pl.col('close') <= pl.col('high')) &
        (pl.col('close') >= pl.col('low'))
    )
    
    return df

# %%
def add_daily_returns(history: pl.DataFrame, method: str = 'log') -> pl.DataFrame:
    """
    Ajoute une colonne 'daily_return' au DataFrame
    
    Parameters:
    -----------
    method : str
        'delta' : prix[t] - prix[t-1] (pour futures avec prix négatifs possibles)
        'pct' : (prix[t] - prix[t-1]) / |prix[t-1]| (pourcentage)
        'log' : ln(prix[t] / prix[t-1]) (classique mais problème avec négatifs)
    """
    dtcol = "datetime" if "datetime" in history.columns else "timestamp"
    
    # Ajoute la colonne date au df principal
    history = history.with_columns(
        pl.col(dtcol).dt.date().alias("date")
    )
    
    # Aggregate à daily
    daily = (
        history
        .group_by(["symbol", "date"], maintain_order=True)
        .agg(pl.col("close").last())
        .sort(["symbol", "date"])
    )
    
    # Calcule returns selon méthode
    if method == 'delta':
        daily = daily.with_columns(
            (pl.col("close") - pl.col("close").shift(1))
            .over("symbol")
            .alias("daily_return")
        )
    elif method == 'pct':
        daily = daily.with_columns(
            ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(1).abs())
            .over("symbol")
            .alias("daily_return")
        )
    elif method == 'log':
        daily = daily.with_columns(
            (pl.col("close") / pl.col("close").shift(1)).log()
            .over("symbol")
            .alias("daily_return")
        )
    
    # Join back to original (maintenant "date" existe dans les deux)
    return history.join(
        daily.select(["symbol", "date", "daily_return"]),
        on=["symbol", "date"],
        how="left"
    )

# %%
# declare dataframe 
df_list = []  # Liste pour accumuler les DataFrames

yh = YahooFinanceClient()

for sym_key, sym_value in tickers_map.items():  # Itère sur (clé, valeur)
    df_cl = yh.get_price(
        sym_value,  # Utilise la valeur du dictionnaire ('CL=F', 'NG=F', etc.)
        from_date=from_date.strftime("%Y-%m-%d"),
        to_date=to_date.strftime("%Y-%m-%d"),
        interval=timeframe, 
        postclean=True
    )
    # Ajoute la colonne symbol
    df_cl = df_cl.with_columns(pl.lit(sym_value).alias("symbol"))
    # Ajoute à la liste
    df_list.append(df_cl)

# Concatène tous les DataFrames
df = pl.concat(df_list) if df_list else pl.DataFrame()
df = clean_price_data(df)
df = add_daily_returns(df)
df

# %%
def get_trading_signal(history: pl.DataFrame, threshold: float = 1.00) -> dict:
    """
    TREND signal avec double step function (Baltas & Kosowski)
    threshold: seuil de significativité (1.96 = 95% confidence)
    
    Retourne:
    - +1 si t-stat > threshold (momentum haussier significatif)
    -  0 si |t-stat| <= threshold (pas de trend clair)
    - -1 si t-stat < -threshold (momentum baissier significatif)
    """
    symbols = sorted(history['symbol'].unique().to_list())
    dict_results = {}
    
    for symbol in symbols:
        returns = (
            history
            .filter(pl.col('symbol') == symbol)
            .select('daily_return')
            .drop_nulls()
            .to_series()
            .to_numpy()
        )
        
        if len(returns) < 2:
            dict_results[symbol] = 0
            continue
        
        # t-statistic
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        n = len(returns)
        
        if std == 0:
            t_stat = 0
        else:
            t_stat = mean / (std / np.sqrt(n))
        
        # ✅ Double step function (DISCRETE)
        if t_stat > threshold:
            signal = 1
        elif t_stat < -threshold:
            signal = -1
        else:
            signal = 0
        
        dict_results[symbol] = signal
    
    return dict_results

# %%
def get_y_z_volatility(
        history: pl.DataFrame, 
        available_symbols: list[str], 
        one_month: int = 30,
        data_frequency: str = "1h"
    ):
        """
        Yang & Zhang Drift-Independent Volatility Estimation
        VERSION AVEC FILTRES POUR PRIX NÉGATIFS
        """
        results = []
        available_symbols = sorted(available_symbols)
        
        annualization_factors = {
            "1h": np.sqrt(252 * 24),
            "4h": np.sqrt(252 * 6),
            "1d": np.sqrt(252),
            "1D": np.sqrt(252),
            "daily": np.sqrt(252),
        }
        
        annualization = annualization_factors.get(data_frequency, np.sqrt(252))
        
        if "datetime" not in history.columns:
            raise ValueError("history must have a 'datetime' column")
        
        history = history.with_columns(
            pl.col("datetime").cast(pl.Datetime("ns"))
        )
        
        latest_date = history.filter(pl.col("symbol") == available_symbols[0])["datetime"].max()
        cutoff_date = latest_date - timedelta(days=one_month)
        
        for ticker in available_symbols:
            past_month = (
                history.filter(
                    (pl.col("symbol") == ticker) & 
                    (pl.col("datetime") >= cutoff_date)
                )
                .sort("datetime")
            )
            
            estimation_period = past_month.shape[0]
            
            if estimation_period <= 1:
                results.append(np.nan)
                continue
            
            # Convert to NumPy
            o = past_month["open"].to_numpy()
            h = past_month["high"].to_numpy()
            l = past_month["low"].to_numpy()
            c = past_month["close"].to_numpy()
            
            # ✅ FILTRE CRITIQUE: enlève les prix négatifs/nuls
            mask = (o > 0) & (h > 0) & (l > 0) & (c > 0)
            o, h, l, c = o[mask], h[mask], l[mask], c[mask]
            
            if len(c) < 2:
                results.append(np.nan)
                continue
            
            # Calculate k
            k = 0.34 / (1.34 + (len(c) + 1) / max(len(c) - 1, 1))
            
            # sigma_o_j : overnight jump vol
            oc_log_returns = np.log(o[1:] / c[:-1])
            oc_log_returns = oc_log_returns[np.isfinite(oc_log_returns)]
            
            if len(oc_log_returns) < 2:
                results.append(np.nan)
                continue
            
            sigma_oj = np.std(oc_log_returns, ddof=1)
            
            # sigma_s_d : standard vol
            cc_log_returns = np.log(c[1:] / c[:-1])
            cc_log_returns = cc_log_returns[np.isfinite(cc_log_returns)]
            
            if len(cc_log_returns) < 2:
                results.append(np.nan)
                continue
            
            sigma_sd = np.std(cc_log_returns, ddof=1)
            
            # sigma_r_s : Rogers & Satchell
            H = np.log(h / o)
            L = np.log(l / o)
            C = np.log(c / o)
            
            # Filtre aussi les valeurs infinies ici
            rs_values = H * (H - C) + L * (L - C)
            rs_values = rs_values[rs_values >= 0]  # Uniquement valeurs positives
            
            if len(rs_values) == 0:
                results.append(np.nan)
                continue
            
            sigma_rs_daily = np.sqrt(rs_values)
            sigma_rs_daily = sigma_rs_daily[np.isfinite(sigma_rs_daily)]
            
            if len(sigma_rs_daily) == 0:
                results.append(np.nan)
                continue
            
            sigma_rs = np.mean(sigma_rs_daily)
            
            # Yang & Zhang volatility
            sigma_yz = np.sqrt(sigma_oj**2 + k * sigma_sd**2 + (1 - k) * sigma_rs**2)
            
            # Check final value
            if np.isnan(sigma_yz) or sigma_yz == 0:
                results.append(np.nan)
                continue
            
            results.append(sigma_yz * annualization)
        
        return results

# %%
def get_correlation_factor(
        history: pl.DataFrame, 
        trade_signals: dict, 
        available_symbols: list,
        window: int = 90
    ):
        """
        Calculate Correlation Factor
        Utilise la colonne 'daily_return' pré-calculée (doit etre en log)
        """
        dtcol = "datetime" if "datetime" in history.columns else "timestamp"
        lookback_date = history[dtcol].max() - timedelta(days=window)
        
        available_symbols = sorted(available_symbols)
        all_returns = []
        
        for symbol in available_symbols:
            # Utilise directement daily_return
            daily = (
                history
                .filter(
                    (pl.col("symbol") == symbol) & 
                    (pl.col(dtcol) >= lookback_date) 
                )
                .select(['date', 'daily_return'])
                .drop_nulls()
                .unique(subset=['date'])
                .sort('date')
                .rename({'daily_return': symbol})
            )
            
            all_returns.append(daily)

        # Merge sur les dates communes
        returns_df = all_returns[0]
        for df in all_returns[1:]:
            returns_df = returns_df.join(df, on='date', how='inner')
        

        import pandas as pd
        returns_pd = returns_df.drop('date').to_pandas()
        
        if len(returns_pd) < 10:
            return np.sqrt(len(available_symbols))
        
        corr_matrix = returns_pd.corr()
        n_assets = len(available_symbols)
        
        # Calculate rho_bar
        summation = 0
        count = 0
        for i in range(n_assets - 1):
            for j in range(i + 1, n_assets):
                symbol_i = available_symbols[i]
                symbol_j = available_symbols[j]
                
                x_i = trade_signals[symbol_i]
                x_j = trade_signals[symbol_j]
                rho_ij = corr_matrix.loc[symbol_i, symbol_j]
                
                if not np.isnan(rho_ij):
                    summation += x_i * x_j * rho_ij
                    count += 1
        
        if count == 0:
            return np.sqrt(n_assets)
        
        rho_bar = (2 * summation) / (n_assets * (n_assets - 1))
        cf = np.sqrt(n_assets / (1 + (n_assets - 1) * rho_bar))
        
        return cf

# %%
def rebalance_portfolio_correct(
        history: pl.DataFrame,
        current_positions: dict,
        portfolio_target_sigma: float = 0.12,
        capital: float = 100000,
        window: int = 90,
        data_frequency: str = None
    ):
        """
        Rebalance portfolio - Baltas & Kosowski method
        
        Returns positions in NUMBER OF CONTRACTS
        
        Parameters:
        -----------
        history : pl.DataFrame
            DataFrame avec colonnes: datetime, symbol, open, high, low, close, daily_return
        portfolio_target_sigma : float
            Target annualized volatility (ex: 0.12 = 12%)
        capital : float
            Total capital disponible
        window : int
            Lookback window en jours pour correlation factor
        data_frequency : str
            Fréquence des données ('1h', '1d', '4h', etc.)
        """
        if data_frequency is None:
            # Auto-detect frequency
            dtcol = "datetime" if "datetime" in history.columns else "timestamp"
            sample = history.sort(dtcol).head(100)
            time_diffs = sample[dtcol].diff().drop_nulls()
            avg_diff_seconds = time_diffs.mean().total_seconds()
            
            if avg_diff_seconds < 3600 * 2:
                data_frequency = "1h"
            elif avg_diff_seconds < 3600 * 12:
                data_frequency = "4h"
            else:
                data_frequency = "1d"
        
        available_symbols = sorted(history['symbol'].unique().to_list())
        
        if len(available_symbols) == 0:
            return current_positions, {}
        
        # 1. Calculate components
        trade_signals = get_trading_signal(history, 0.5)
        volatility = get_y_z_volatility(history, available_symbols, data_frequency=data_frequency)
        c_f_rho_bar = get_correlation_factor(history, trade_signals, available_symbols, window=window)
        
        # 2. Calculate weights and positions
        n_assets = len(available_symbols)
        new_positions = {}
        
        # Contract multipliers
        contract_multipliers = {
            'CL=F': 1000,   # 1000 barrels
            'RB=F': 42000,  # 42000 gallons
            'NG=F': 10000,  # 10000 MMBtu
            'HG=F': 25000,  # 25000 pounds
            'GC=F': 100,    # 100 troy ounces
            'SI=F': 5000,   # 5000 troy ounces
            'PL=F': 50,     # 50 troy ounces
        }
        
        for i, symbol in enumerate(available_symbols):
            signal = trade_signals[symbol]
            vol = volatility[i]
            
            if np.isnan(vol) or vol == 0:
                new_positions[symbol] = 0
                continue
            
            # Baltas & Kosowski weight formula
            weight = (signal * portfolio_target_sigma * c_f_rho_bar) / (n_assets * vol)
            # Mais aussi dans le calcul des weights pour le return dict:
            weights = {
                symbol: (
                    (trade_signals[symbol] * portfolio_target_sigma * c_f_rho_bar) / (n_assets * volatility[i])
                    if not np.isnan(volatility[i]) and volatility[i] != 0
                    else 0  # ✅ Fallback
                )
                for i, symbol in enumerate(available_symbols)
            }
            
            # Get last price
            last_price = history.filter(pl.col('symbol') == symbol)['close'].tail(1)[0]
            
            # Get contract multiplier
            multiplier = contract_multipliers.get(symbol, 1)
            
            # Calculate dollar allocation
            dollar_allocation = weight * capital

            # Calculate number of contracts
            contract_value = last_price * multiplier

            # Vérifier que contract_value est valide
            if np.isnan(contract_value) or contract_value == 0:
                new_positions[symbol] = 0
                continue

            num_contracts = dollar_allocation / contract_value

            # Vérifier que num_contracts est valide avant de l'arrondir
            if np.isnan(num_contracts) or not np.isfinite(num_contracts):
                new_positions[symbol] = 0
            else:
                new_positions[symbol] = round(num_contracts)
        
        return new_positions, {
            'weights': {symbol: (trade_signals[symbol] * portfolio_target_sigma * c_f_rho_bar) / (n_assets * volatility[i]) 
                    for i, symbol in enumerate(available_symbols)},
            'signals': trade_signals,
            'volatilities': dict(zip(available_symbols, volatility)),
            'correlation_factor': c_f_rho_bar
        }

# %%

# Usage
# Note: passez data_frequency="1h" si vos données sont horaires, "1d" si quotidiennes, etc.
new_positions, details = rebalance_portfolio_correct(
    history=df,
    current_positions={},
    portfolio_target_sigma=portfolio_target_sigma,
    capital=initial,
    window=window_days,
)


print("\n=== PORTFOLIO REBALANCE ===")
print(f"Capital: ${initial:,.0f}")
print(f"\nNew Positions (number of contracts):")
for symbol, contracts in new_positions.items():
    direction = "LONG" if contracts > 0 else "SHORT"
    print(f"  {symbol}: {abs(contracts)} contracts {direction}")

print(f"\nDetails:")
print(f"  Signals: {details['signals']}")
print(f"  Volatilities: {details['volatilities']}")
print(f"  Correlation Factor: {details['correlation_factor']:.4f}")
print(f"  Weights: {details['weights']}")


# %%
def backtest_baltas_kosowski_fractional(
        history: pl.DataFrame, 
        initial_capital=10000,
        portfolio_target_sigma=0.12,
        window: int = 90
    ):
        """
        Backtest avec positions fractionnaires
        Assume que 'daily_return' column existe déjà dans history
        """
        results = []
        capital = initial_capital
        
        history_with_month = history.with_columns(
            pl.col('datetime').dt.truncate('1mo').alias('month')
        )
        
        months = history_with_month['month'].unique().sort()

        for i, month in enumerate(months[1:]):
            hist_until_month = history.filter(pl.col('datetime') < month)
            
            # Calcule les weights
            _, details = rebalance_portfolio_correct(
                hist_until_month, 
                {}, 
                capital=capital,
                portfolio_target_sigma=portfolio_target_sigma,
                window=window
            )

            # ✅ Debug: toujours calculer
            active_signals = sum(1 for s in details['signals'].values() if s != 0)
            active_weights = sum(1 for w in details['weights'].values() if abs(w) > 0.0001)
            
            # ✅ Print seulement premiers/derniers
            if i < 5 or i > len(months) - 10:
                print(f"{month}: {active_signals}/16 signaux, {active_weights}/16 poids > 0.0001")
            
            weights = details['weights']
            
            # ✅ Si NaN volatility → SKIP au lieu de BREAK
            if any(np.isnan(v) for v in details['volatilities'].values()):
                print(f"⚠️ Skipping {month} due to NaN volatility")
                continue  # ← CONTINUE au lieu de BREAK
            
            if np.isnan(details['correlation_factor']):
                print(f"⚠️ Skipping {month} due to NaN correlation")
                continue  # ← CONTINUE au lieu de BREAK
            
            # Determine next month boundary
            if i+1 < len(months) - 1:
                next_month = months[i+2]
            else:
                next_month = month + timedelta(days=60)
            
            # Get month data
            month_data = history.filter(
                (pl.col('datetime') >= month) & 
                (pl.col('datetime') < next_month)
            )
            
            # Calculate returns
            month_returns = {}
            for symbol in weights.keys():
                symbol_data = month_data.filter(pl.col('symbol') == symbol)
                
                if len(symbol_data) > 1:
                    start_price = symbol_data['close'].head(1)[0]
                    end_price = symbol_data['close'].tail(1)[0]
                    
                    if start_price > 0:
                        month_returns[symbol] = (end_price - start_price) / abs(start_price)
                    else:
                        month_returns[symbol] = 0
                else:
                    month_returns[symbol] = 0

            # Portfolio return
            portfolio_return = sum(
                weights[symbol] * month_returns[symbol] 
                for symbol in weights.keys()
            )
            
            capital *= (1 + portfolio_return)
  
            results.append({
                'month': month,
                'weights': weights,
                'returns': month_returns,
                'portfolio_return': portfolio_return,
                'capital': capital
            })
        
        return pl.DataFrame(results)

# %%
# TESTE CETTE VERSION
backtest_frac = backtest_baltas_kosowski_fractional(
    df, 
    initial_capital=initial,
    portfolio_target_sigma=portfolio_target_sigma
)

# Performance finale (CORRIGÉE)
final = backtest_frac['capital'].tail(1)[0]
total_return = (final - initial) / initial

print(f"\nInitial Capital: ${initial:,.0f}")
print(f"Final Capital: ${final:,.2f}")
print(f"Total Return: {total_return:.2%}")

# Calcule aussi le Sharpe
monthly_returns = backtest_frac['portfolio_return']
sharpe = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)
print(f"Sharpe Ratio: {sharpe:.2f}")


# %%
def create_trades_from_backtest(backtest_results: pl.DataFrame, tickers: list = ['CL=F'], initial_capital: float = 100000):
    """
    Version avec debug
    """
    trades_list = []
    
    print(f"Creating trades for {len(tickers)} tickers")
    print(f"Backtest has {len(backtest_results)} months")
    
    for t in tickers:
        symbol_to_plot = t
        trades_for_ticker = 0
        
        for i, row in enumerate(backtest_results.iter_rows(named=True)):
            month = row['month']
            weights = row['weights']
            
            if symbol_to_plot not in weights:
                continue
            
            weight = weights[symbol_to_plot]
            
            # ✅ DEBUG: Log tous les poids
            if i < 5 or i > len(backtest_results) - 5:  # Premiers et derniers
                print(f"{symbol_to_plot} @ {month}: weight={weight:.6f}")
            
            if abs(weight) < 0.0001:
                continue
            
            action = "BUY" if weight > 0 else "SELL"
            
            # IMPORTANT: Trouve le prix
            month_data = df.filter(
                (pl.col('symbol') == symbol_to_plot) &
                (pl.col('datetime') >= month)
            )
            
            if len(month_data) == 0:
                print(f"⚠️ No data for {symbol_to_plot} at {month}")  # ← BUG ICI?
                continue
            
            price = month_data['close'].head(1)[0]
            
            trades_list.append({
                'ticker': symbol_to_plot,
                'datetime': month,
                'position_number': i,
                'action': action,
                'price': price,
                'quantity_usd': abs(weight) * row['capital'],
                'position_size': abs(weight),
                'pnl': row['portfolio_return'] * row['capital'] if i > 0 else 0.0,
                'cumulative_capital': row['capital'] / initial_capital
            })
            
            trades_for_ticker += 1
        
        print(f"{symbol_to_plot}: {trades_for_ticker} trades created")
    
    return pl.DataFrame(trades_list)

# Test
trades_bk = create_trades_from_backtest(backtest_frac, tickers=['CL=F', 'RB=F'])

# %%
tickers = [sym_value for _, sym_value in tickers_map.items()]
tickers

# %%
# UTILISE CETTE VERSION
trades_bk = create_trades_from_backtest(
    backtest_frac, 
    tickers=tickers,  # ← Spécifie CL uniquement
    initial_capital=initial
)

# %%
# check if trades_bk is empty
df.group_by('symbol').agg(
    pl.col('datetime').max().alias('max_date'),
    pl.col('datetime').min().alias('min_date')
).to_pandas()


# %%
# check if trades_bk is empty
trades_bk.group_by('ticker').agg(
    pl.col('datetime').max().alias('max_date'),
    pl.col('datetime').min().alias('min_date')
).to_pandas()

# %%
trades_bk.group_by('ticker').agg(
    pl.col('datetime').max().alias('max_date'),
    pl.col('datetime').min().alias('min_date')
)

# %%
from quanta.clients.chart import ChartClient
chart_client = ChartClient()
chart_client.plot(
    df, 
    "GC=F",  
    trades_df=trades_bk, 
    theme='professional',
    x_axis_type='datetime'
)

# %%
def plot_performance_overview(backtest_results: pl.DataFrame, initial_capital: float):
    """
    Page 1: Vue d'ensemble de la performance (style institutionnel)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    months = backtest_results['month'].to_list()
    capital = backtest_results['capital'].to_list()
    monthly_returns = backtest_results['portfolio_return'].to_numpy()
    
    # Drawdowns
    running_max = [capital[0]]
    for c in capital[1:]:
        running_max.append(max(running_max[-1], c))
    drawdowns = [(c - mx) / mx * 100 for c, mx in zip(capital, running_max)]
    
    # 3 subplots verticaux
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.08,
        subplot_titles=('Portfolio Equity Curve', 'Underwater Plot (Drawdown)', 'Monthly Returns')
    )
    
    # === EQUITY CURVE ===
    fig.add_trace(go.Scatter(
        x=months, y=capital,
        line=dict(color='#2E86AB', width=3),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.15)',
        name='Portfolio Value',
        hovertemplate='<b>Portfolio Value</b><br>Date: %{x|%Y-%m}<br>Value: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_hline(
        y=initial_capital,
        line=dict(dash="dash", color="gray", width=2),
        annotation=dict(text="Initial Capital", font=dict(size=11)),
        row=1, col=1
    )
    
    # === DRAWDOWN ===
    fig.add_trace(go.Scatter(
        x=months, y=drawdowns,
        line=dict(color='#A23B72', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(162, 59, 114, 0.2)',
        name='Drawdown',
        hovertemplate='<b>Drawdown</b><br>Date: %{x|%Y-%m}<br>DD: %{y:.2f}%<extra></extra>'
    ), row=2, col=1)
    
    fig.add_hline(y=0, line=dict(dash="dot", color="gray", width=1), row=2, col=1)
    
    # === MONTHLY RETURNS BAR ===
    colors = ['#2E7D32' if r > 0 else '#C62828' for r in monthly_returns]
    
    fig.add_trace(go.Bar(
        x=months,
        y=monthly_returns * 100,
        marker_color=colors,
        marker_line_width=0,
        name='Monthly Returns',
        hovertemplate='<b>Monthly Return</b><br>Date: %{x|%Y-%m}<br>Return: %{y:.2f}%<extra></extra>'
    ), row=3, col=1)
    
    fig.add_hline(y=0, line=dict(dash="dot", color="gray", width=1), row=3, col=1)
    
    # === METRICS BOX ===
    mean_ret = np.mean(monthly_returns) * 100
    std_ret = np.std(monthly_returns) * 100
    sharpe = (np.mean(monthly_returns) / np.std(monthly_returns)) * np.sqrt(12)
    final_return = (capital[-1] - initial_capital) / initial_capital * 100
    max_dd = min(drawdowns)
    
    metrics_text = (
        f"<b>PERFORMANCE METRICS</b><br><br>"
        f"<b>Returns</b><br>"
        f"Total Return: <b>{final_return:+.2f}%</b><br>"
        f"Annualized: <b>{(((capital[-1]/initial_capital)**(12/len(months)))-1)*100:+.2f}%</b><br>"
        f"Monthly Avg: <b>{mean_ret:+.2f}%</b><br><br>"
        f"<b>Risk</b><br>"
        f"Volatility: <b>{std_ret:.2f}%</b><br>"
        f"Max Drawdown: <b>{max_dd:.2f}%</b><br>"
        f"Sharpe Ratio: <b>{sharpe:.2f}</b><br><br>"
        f"<b>Trading</b><br>"
        f"Months: <b>{len(months)}</b><br>"
        f"Best Month: <b>{monthly_returns.max()*100:+.2f}%</b><br>"
        f"Worst Month: <b>{monthly_returns.min()*100:+.2f}%</b>"
    )
    
    fig.add_annotation(
        text=metrics_text,
        xref="paper", yref="paper",
        x=1.15, y=0.95,
        xanchor='left', yanchor='top',
        showarrow=False,
        bgcolor="rgba(250, 250, 250, 0.98)",
        bordercolor="#2E86AB",
        borderwidth=2,
        font=dict(size=11, family="Arial"),
        align='left'
    )
    
    # === LAYOUT ===
    fig.update_layout(
        title={
            'text': '<b>BALTAS & KOSOWSKI MOMENTUM STRATEGY</b><br><sup>Performance Analysis Report</sup>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'family': 'Arial', 'color': '#1a1a1a'}
        },
        height=1000,
        template='plotly_white',
        showlegend=False,
        font=dict(family="Arial", size=11),
        margin=dict(l=80, r=250, t=100, b=60)
    )
    
    # Axes styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
    
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1, title_font=dict(size=12))
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1, title_font=dict(size=12))
    fig.update_yaxes(title_text="Return (%)", row=3, col=1, title_font=dict(size=12))
    fig.update_xaxes(title_text="Date", row=3, col=1, title_font=dict(size=12))
    
    fig.show()


plot_performance_overview(backtest_frac, initial)

# %%
def plot_risk_analysis(backtest_results: pl.DataFrame):
    """
    Page 2: Analyse de risque détaillée
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    monthly_returns = backtest_results['portfolio_return'].to_numpy()
    months = backtest_results['month'].to_list()
    
    # Rolling metrics
    window = 12
    rolling_sharpe = []
    rolling_vol = []
    
    for i in range(len(monthly_returns)):
        if i < window:
            rolling_sharpe.append(None)
            rolling_vol.append(None)
        else:
            window_rets = monthly_returns[i-window:i]
            mean_r = np.mean(window_rets)
            std_r = np.std(window_rets)
            rolling_sharpe.append((mean_r / std_r * np.sqrt(12)) if std_r > 0 else 0)
            rolling_vol.append(std_r * np.sqrt(12) * 100)
    
    # 2x2 layout
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.5, 0.5],
        column_widths=[0.6, 0.4],
        subplot_titles=(
            'Returns Distribution', 
            'Risk Metrics',
            '12-Month Rolling Sharpe Ratio',
            '12-Month Rolling Volatility'
        ),
        specs=[
            [{"type": "histogram"}, {"type": "box"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    # === HISTOGRAM ===
    fig.add_trace(go.Histogram(
        x=monthly_returns * 100,
        nbinsx=25,
        marker_color='#2E86AB',
        marker_line_color='white',
        marker_line_width=1,
        opacity=0.8,
        name='Returns',
        hovertemplate='<b>Return Range</b><br>%{x:.1f}%<br>Count: %{y}<extra></extra>'
    ), row=1, col=1)
    
    # Normal distribution overlay
    mean = np.mean(monthly_returns * 100)
    std = np.std(monthly_returns * 100)
    x_norm = np.linspace(mean - 3*std, mean + 3*std, 100)
    y_norm = len(monthly_returns) * (monthly_returns.max() - monthly_returns.min()) * 100 / 25 * \
              np.exp(-0.5 * ((x_norm - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    
    fig.add_trace(go.Scatter(
        x=x_norm, y=y_norm,
        line=dict(color='#A23B72', width=3, dash='dash'),
        name='Normal Dist.',
        hovertemplate='Normal Distribution<extra></extra>'
    ), row=1, col=1)
    
    # === BOX PLOT ===
    fig.add_trace(go.Box(
        y=monthly_returns * 100,
        marker_color='#2E86AB',
        boxmean='sd',
        name='Returns',
        hovertemplate='<b>Statistics</b><br>Value: %{y:.2f}%<extra></extra>'
    ), row=1, col=2)
    
    # === ROLLING SHARPE ===
    fig.add_trace(go.Scatter(
        x=months, y=rolling_sharpe,
        line=dict(color='#6A4C93', width=3),
        fill='tonexty',
        name='Rolling Sharpe',
        hovertemplate='<b>12M Sharpe</b><br>%{x|%Y-%m}<br>Sharpe: %{y:.2f}<extra></extra>'
    ), row=2, col=1)
    
    fig.add_hline(y=0, line=dict(dash="dash", color="gray", width=2), row=2, col=1)
    fig.add_hline(y=1, line=dict(dash="dot", color="green", width=1), 
                 annotation=dict(text="Target", font=dict(size=10)), row=2, col=1)
    
    # === ROLLING VOL ===
    fig.add_trace(go.Scatter(
        x=months, y=rolling_vol,
        line=dict(color='#F18F01', width=3),
        fill='tozeroy',
        fillcolor='rgba(241, 143, 1, 0.2)',
        name='Rolling Vol',
        hovertemplate='<b>12M Volatility</b><br>%{x|%Y-%m}<br>Vol: %{y:.2f}%<extra></extra>'
    ), row=2, col=2)
    
    # === LAYOUT ===
    fig.update_layout(
        title={
            'text': '<b>RISK ANALYSIS</b><br><sup>Statistical Distribution & Rolling Metrics</sup>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'family': 'Arial'}
        },
        height=900,
        template='plotly_white',
        showlegend=False,
        font=dict(family="Arial", size=11)
    )
    
    fig.update_xaxes(title_text="Monthly Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=2)
    
    fig.show()

plot_risk_analysis(backtest_frac)

# %%
def plot_allocation_analysis(backtest_results: pl.DataFrame):
    """
    Page 3: Analyse de l'allocation du portfolio
    """
    import plotly.graph_objects as go
    
    months = backtest_results['month'].to_list()
    
    # Prepare heatmap data
    tickers_data = {}
    for row in backtest_results.iter_rows(named=True):
        for ticker, weight in row['weights'].items():
            if ticker not in tickers_data:
                tickers_data[ticker] = []
            tickers_data[ticker].append(weight)
    
    tickers = sorted(tickers_data.keys())
    z_data = [tickers_data[t] for t in tickers]
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[m.strftime('%Y-%m') for m in months],
        y=tickers,
        colorscale=[
            [0.0, '#8B0000'],   # Dark red (short)
            [0.25, '#FF6B6B'],  # Light red
            [0.5, '#FFFFFF'],   # White (neutral)
            [0.75, '#4ECDC4'],  # Light green
            [1.0, '#006400']    # Dark green (long)
        ],
        zmid=0,
        colorbar=dict(
            title=dict(
                text="<b>Position<br>Weight</b>",
                side="right"  # ✅ Fix: side au lieu de titleside
            ),
            tickmode="linear",
            tick0=-0.5,
            dtick=0.25,
            thickness=20,
            len=0.7
        ),
        hovertemplate='<b>%{y}</b><br>Date: %{x}<br>Weight: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '<b>PORTFOLIO ALLOCATION OVER TIME</b><br><sup>Monthly Rebalancing Heatmap</sup>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'family': 'Arial'}
        },
        height=max(600, len(tickers) * 40),
        template='plotly_white',
        xaxis_title="<b>Date</b>",
        yaxis_title="<b>Commodity</b>",
        font=dict(family="Arial", size=12),
        xaxis=dict(
            tickangle=-45,
            showgrid=False
        ),
        yaxis=dict(
            showgrid=False,
            categoryorder='category ascending'
        )
    )
    
    fig.show()

plot_allocation_analysis(backtest_frac)

# %%
df_cl = df_cl.with_columns([
    pl.lit(details['signals']['CL=F']).alias('momentum_signal')
])

signals_over_time = []
for row in backtest_frac.iter_rows(named=True):
    signals_dict = row['weights']  # C'est un dict {'CL=F': 0.16, 'RB=F': -0.21}
    signals_over_time.append({
        'datetime': row['month'] + timedelta(hours=6),
        'signal': signals_dict.get('CL=F', 0)  # ← Extract CL=F weight
    })

signals_df = pl.DataFrame(signals_over_time)

# full join on datetime
df_cl = df_cl.join(signals_df, on='datetime', how='left').fill_null(strategy='forward').sort('datetime')


weights_over_time = []
for row in backtest_frac.iter_rows(named=True):
    weights_over_time.append({
        'datetime': row['month'] + timedelta(hours=6),
        'strategy_weight': row['weights']['CL=F']
    })

weights_df = pl.DataFrame(weights_over_time)

# 2. Join avec df_cl
df_cl = df_cl.join(weights_df, on='datetime', how='left').fill_null(strategy='forward')



# %%
traces = [
    Candlesticks(),
    Line('strategy_weight', name='Strategy Weight', color='purple'),
    Volume()
]
traces

# %%
from quanta.clients.chart import ChartClient
chart_client = ChartClient()
chart_client.plot(
    df_cl, 
    "cl=F",  
    trades_df=trades_bk, 
    traces=traces,
    theme='professional',
    x_axis_type='datetime'
)

# %%


# %%



