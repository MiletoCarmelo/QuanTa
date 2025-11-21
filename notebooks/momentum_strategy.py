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


# %%
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
df = add_daily_returns(df, method='log')
df


# %%
def get_trading_signal(history: pl.DataFrame, threshold: float = 1.0) -> dict:
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
        trade_signals = get_trading_signal(history)
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
            weight = np.clip(weight, -1, 1)
            
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
            
            weights = details['weights']
            
            # Si un des composants est NaN, ça explique tout
            if any(np.isnan(v) for v in details['volatilities'].values()):
                print("⚠️ VOLATILITY NaN detected!")
                print(f"\n=== Month {i}: {month} ===")
                print(f"Data points: {len(hist_until_month)}")
                print(f"Signals: {details['signals']}")
                print(f"Volatilities: {details['volatilities']}")
                print(f"Weights: {details['weights']}")
                break
            
            if np.isnan(details['correlation_factor']):
                print("⚠️ CORRELATION FACTOR NaN detected!")
                print(f"\n=== Month {i}: {month} ===")
                print(f"Data points: {len(hist_until_month)}")
                print(f"Signals: {details['signals']}")
                print(f"Correlation Factor: {details['correlation_factor']}")
                print(f"Weights: {details['weights']}")
                break
            
            
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

            
            # Calculate returns using pre-computed daily_return
            month_returns = {}
            for symbol in weights.keys():
                symbol_data = month_data.filter(pl.col('symbol') == symbol)
                
                if len(symbol_data) > 1:
                    # ✅ Simple: prix final / prix initial - 1
                    start_price = symbol_data['close'].head(1)[0]
                    end_price = symbol_data['close'].tail(1)[0]
                    
                    if start_price > 0:
                        month_returns[symbol] = (end_price - start_price) / abs(start_price)
                    else:
                        month_returns[symbol] = 0
                else:
                    month_returns[symbol] = 0

            # Portfolio return = Σ(weight_i × return_i)
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
    Crée des trades pour UN SEUL symbol (pour plotting)
    VERSION CORRIGÉE
    """
    trades_list = []
    
    for t in tickers:
        symbol_to_plot = t
        for i, row in enumerate(backtest_results.iter_rows(named=True)):
            month = row['month']
            weights = row['weights']
            portfolio_return = row['portfolio_return']
            capital = row['capital']
            
            # Weight pour le symbol spécifique
            if symbol_to_plot not in weights:
                continue
                
            weight = weights[symbol_to_plot]
            
            # Skip si weight trop petit
            if abs(weight) < 0.0001:
                continue
            
            # Action
            action = "BUY" if weight > 0 else "SELL"
            
            # IMPORTANT: Trouve le VRAI prix du symbol à cette date
            month_data = df.filter(
                (pl.col('symbol') == symbol_to_plot) &  # ← FILTRE LE BON SYMBOL
                (pl.col('datetime') >= month)
            )
            
            if len(month_data) == 0:
                continue
                
            price = month_data['close'].head(1)[0]  # ← PRIX RÉEL, pas weight !
            
            trades_list.append({
                'ticker': symbol_to_plot,
                'datetime': month,
                'position_number': i,
                'action': action,
                'price': price,  # ← Vrai prix CL (55-75)
                'quantity_usd': abs(weight) * capital,
                'position_size': abs(weight),
                'pnl': portfolio_return * capital if i > 0 else 0.0,
                'cumulative_capital': capital / initial_capital
            })
    
    return pl.DataFrame(trades_list)


# %%
tickers = [sym_value for _, sym_value in tickers_map.items()]

# UTILISE CETTE VERSION
trades_bk = create_trades_from_backtest(
    backtest_frac, 
    tickers=tickers,  # ← Spécifie CL uniquement
    initial_capital=initial
)
trades_bk


# %%
from quanta.clients.chart import ChartClient
chart_client = ChartClient()
chart_client.plot(
    df_cl, 
    "cl=F",  
    trades_df=trades_bk, 
    theme='professional',
    x_axis_type='datetime'
)
