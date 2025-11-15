import polars as pl
import numpy as np
from abc import ABC, abstractmethod
from quanta.clients.yfinance import YahooFinanceClient
import talib
# ========== Base Class ==========

class Indicator(ABC):
    """Base class for all technical indicators."""
    
    @abstractmethod
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculates the indicator and adds it to the DataFrame."""
        pass
    
    @abstractmethod
    def get_column_names(self):
        """Returns the names of columns created by the indicator."""
        pass


# ========== Moving Averages ==========

class SMA(Indicator):
    """Simple Moving Average."""
    
    def __init__(self, period: int = 50, column: str = 'close'):
        self.period = period
        self.column = column
        self.name = f'SMA{period}'
    
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(self.column).rolling_mean(window_size=self.period).alias(self.name)
        )
    
    def get_column_names(self):
        return [self.name]


class EMA(Indicator):
    """Exponential Moving Average."""
    
    def __init__(self, period: int = 12, column: str = 'close'):
        self.period = period
        self.column = column
        self.name = f'EMA{period}'
    
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(self.column).ewm_mean(span=self.period).alias(self.name)
        )
    
    def get_column_names(self):
        return [self.name]


class WMA(Indicator):
    """Weighted Moving Average."""
    
    def __init__(self, period: int = 20, column: str = 'close'):
        self.period = period
        self.column = column
        self.name = f'WMA{period}'
    
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        # Simplified WMA calculation
        weights = list(range(1, self.period + 1))
        weight_sum = sum(weights)
        
        def weighted_avg(values):
            if len(values) < self.period:
                return None
            return sum(v * w for v, w in zip(values[-self.period:], weights)) / weight_sum
        
        wma = df[self.column].rolling_map(weighted_avg, window_size=self.period)
        return df.with_columns(wma.alias(self.name))
    
    def get_column_names(self):
        return [self.name]


# ========== Bollinger Bands ==========

class BollingerBands(Indicator):
    """Bollinger Bands."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, column: str = 'close'):
        self.period = period
        self.std_dev = std_dev
        self.column = column
    
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        middle = pl.col(self.column).rolling_mean(window_size=self.period)
        std = pl.col(self.column).rolling_std(window_size=self.period)
        
        return df.with_columns([
            middle.alias('BB_middle'),
            (middle + self.std_dev * std).alias('BB_upper'),
            (middle - self.std_dev * std).alias('BB_lower')
        ])
    
    def get_column_names(self):
        return ['BB_middle', 'BB_upper', 'BB_lower']


# ========== Momentum Indicators ==========

class RSI(Indicator):
    """Relative Strength Index."""
   
    def __init__(self, period: int = 14, column: str = 'close'):
        self.period = period
        self.column = column
        self.name = f'RSI{period}'  # ✅ RSI, not WMA
   
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        delta = df.select(pl.col(self.column).diff().alias('delta'))['delta']
       
        gain = delta.map_elements(lambda x: x if x > 0 else 0, return_dtype=pl.Float64)
        loss = delta.map_elements(lambda x: -x if x < 0 else 0, return_dtype=pl.Float64)
       
        avg_gain = gain.rolling_mean(window_size=self.period)
        avg_loss = loss.rolling_mean(window_size=self.period)
       
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
       
        return df.with_columns(rsi.alias(self.name))
   
    def get_column_names(self):
        return [self.name]


class MACD(Indicator):
    """Moving Average Convergence Divergence."""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, column: str = 'close'):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.column = column
    
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        ema_fast = df.select(pl.col(self.column).ewm_mean(span=self.fast).alias('ema_fast'))['ema_fast']
        ema_slow = df.select(pl.col(self.column).ewm_mean(span=self.slow).alias('ema_slow'))['ema_slow']
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm_mean(span=self.signal)
        histogram = macd_line - signal_line
        
        return df.with_columns([
            macd_line.alias('MACD'),
            signal_line.alias('MACD_signal'),
            histogram.alias('MACD_hist')
        ])
    
    def get_column_names(self):
        return ['MACD', 'MACD_signal', 'MACD_hist']


class Stochastic(Indicator):
    """Stochastic Oscillator - momentum indicator comparing closing price to price range.
    
    Returns %K and %D (smoothed %K).
    Values: 0-100
    - < 20: Oversold (potential buy signal)
    - > 80: Overbought (potential sell signal)
    """
    
    def __init__(self, k_period: int = 14, k_slow: int = 3, d_period: int = 3):
        self.k_period = k_period
        self.k_slow = k_slow
        self.d_period = d_period
        self.name = f'Stoch{k_period}_{k_slow}_{d_period}'
    
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        # Convert Polars columns to numpy arrays for TA-Lib
        high_arr = df['high'].to_numpy()
        low_arr = df['low'].to_numpy()
        close_arr = df['close'].to_numpy()
        
        # Calculate Stochastic using TA-Lib
        slowk, slowd = talib.STOCH(high_arr, low_arr, close_arr, 
                                    fastk_period=self.k_period,
                                    slowk_period=self.k_slow,
                                    slowk_matype=0,
                                    slowd_period=self.d_period,
                                    slowd_matype=0)
        
        return df.with_columns([
            pl.Series(f'{self.name}_K', slowk),
            pl.Series(f'{self.name}_D', slowd)
        ])
    
    def get_column_names(self):
        return [f'{self.name}_K', f'{self.name}_D']


class CCI(Indicator):
    """Commodity Channel Index - identifies cyclical trends and overbought/oversold conditions.
    
    Values: typically -100 to +100, but can exceed
    - < -100: Oversold (potential buy signal)
    - > +100: Overbought (potential sell signal)
    - Between -100 and +100: Normal range
    """
    
    def __init__(self, period: int = 14):
        self.period = period
        self.name = f'CCI{period}'
    
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        # Convert Polars columns to numpy arrays for TA-Lib
        high_arr = df['high'].to_numpy()
        low_arr = df['low'].to_numpy()
        close_arr = df['close'].to_numpy()
        
        # Calculate CCI using TA-Lib
        cci = talib.CCI(high_arr, low_arr, close_arr, timeperiod=self.period)
        
        return df.with_columns(pl.Series(self.name, cci))
    
    def get_column_names(self):
        return [self.name]


# ========== Volatility Indicators ==========

class ATR(Indicator):
    """Average True Range."""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.name = f'ATR{period}'
    
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        high_low = pl.col('high') - pl.col('low')
        high_close = (pl.col('high') - pl.col('close').shift(1)).abs()
        low_close = (pl.col('low') - pl.col('close').shift(1)).abs()
        
        true_range = pl.max_horizontal([high_low, high_close, low_close])
        atr = true_range.rolling_mean(window_size=self.period)

        return df.with_columns(atr.alias(self.name))

    def get_column_names(self):
        return [self.name]


class Volatility(Indicator):
    """Historical Volatility (standard deviation of returns)."""
    
    def __init__(self, period: int = 20, annualize: bool = True, column: str = 'close'):
        self.period = period
        self.annualize = annualize
        self.column = column
        self.name = f'Volatility{period}'
    
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        # Calculate returns (percentage change)
        returns = pl.col(self.column).pct_change()
        
        # Calculate rolling standard deviation of returns
        volatility = returns.rolling_std(window_size=self.period)
        
        # Annualize if requested (multiply by sqrt of periods per year)
        # Assuming daily data (252 trading days per year)
        # For other timeframes, users can set annualize=False
        if self.annualize:
            volatility = volatility * (252 ** 0.5)
        
        return df.with_columns(volatility.alias(self.name))

    def get_column_names(self):
        return [self.name]


# ========== Candlestick Patterns ==========

class EngulfingPattern(Indicator):
    """Bullish and Bearish Engulfing Pattern using TA-Lib CDLENGULFING."""
    
    def __init__(self, separate_columns: bool = False):
        """
        Initialize Engulfing Pattern indicator.
        
        Args:
            separate_columns: If True, creates separate columns for bullish and bearish.
                            If False, creates a single column with values: 1 (bullish), -1 (bearish), 0 (none)
        """
        self.separate_columns = separate_columns
    
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Engulfing Pattern using TA-Lib CDLENGULFING.
        
        CDLENGULFING returns:
        - 100 for Bullish Engulfing
        - -100 for Bearish Engulfing  
        - 0 otherwise
        """
        # Convert Polars columns to numpy arrays for TA-Lib
        open_arr = df['open'].to_numpy()
        high_arr = df['high'].to_numpy()
        low_arr = df['low'].to_numpy()
        close_arr = df['close'].to_numpy()
        
        # Calculate engulfing pattern using TA-Lib
        engulfing = talib.CDLENGULFING(open_arr, high_arr, low_arr, close_arr)
        
        if self.separate_columns:
            # Create separate columns for bullish and bearish
            bullish = (engulfing == 100).astype(np.int32)
            bearish = (engulfing == -100).astype(np.int32)
            
            return df.with_columns([
                pl.Series('Engulfing_Bullish', bullish),
                pl.Series('Engulfing_Bearish', bearish)
            ])
        else:
            # Create single column: 1 for bullish, -1 for bearish, 0 for none
            pattern = np.where(engulfing == 100, 1, np.where(engulfing == -100, -1, 0))
            return df.with_columns(pl.Series('Engulfing', pattern))
    
    def get_column_names(self):
        if self.separate_columns:
            return ['Engulfing_Bullish', 'Engulfing_Bearish']
        else:
            return ['Engulfing']


# ========== Volume Indicators ==========

class OBV(Indicator):
    """On-Balance Volume."""
    
    def __init__(self):
        pass
    
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        direction = (pl.col('close') > pl.col('close').shift(1)).cast(pl.Int32) - \
                   (pl.col('close') < pl.col('close').shift(1)).cast(pl.Int32)
        
        volume_direction = direction * pl.col('volume')
        obv = volume_direction.cum_sum()
        
        return df.with_columns(obv.alias('OBV'))
    
    def get_column_names(self):
        return ['OBV']


class VWAP(Indicator):
    """Volume Weighted Average Price."""
    
    def __init__(self):
        pass
    
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        typical_price = (pl.col('high') + pl.col('low') + pl.col('close')) / 3
        vwap = (typical_price * pl.col('volume')).cum_sum() / pl.col('volume').cum_sum()
        
        return df.with_columns(vwap.alias('VWAP'))
    
    def get_column_names(self):
        return ['VWAP']


class ADX(Indicator):
    """Average Directional Index - measures trend strength (not direction).
    
    ADX values:
    - 0-20: Weak/no trend (good for mean reversion strategies)
    - 20-40: Moderate trend
    - 40+: Strong trend (avoid mean reversion)
    """
    
    def __init__(self, period: int = 14):
        self.period = period
        self.name = f'ADX{period}'
    
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        # Convert Polars columns to numpy arrays for TA-Lib
        high_arr = df['high'].to_numpy()
        low_arr = df['low'].to_numpy()
        close_arr = df['close'].to_numpy()
        
        # Calculate ADX using TA-Lib
        adx = talib.ADX(high_arr, low_arr, close_arr, timeperiod=self.period)
        
        return df.with_columns(pl.Series(self.name, adx))
    
    def get_column_names(self):
        return [self.name]


class WilliamsR(Indicator):
    """Williams %R - momentum indicator similar to Stochastic but inverted.
    
    Values: -100 to 0
    - < -80: Oversold (potential buy signal)
    - > -20: Overbought (potential sell signal)
    """
    
    def __init__(self, period: int = 14):
        self.period = period
        self.name = f'WilliamsR{period}'
    
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        # Convert Polars columns to numpy arrays for TA-Lib
        high_arr = df['high'].to_numpy()
        low_arr = df['low'].to_numpy()
        close_arr = df['close'].to_numpy()
        
        # Calculate Williams %R using TA-Lib
        willr = talib.WILLR(high_arr, low_arr, close_arr, timeperiod=self.period)
        
        return df.with_columns(pl.Series(self.name, willr))
    
    def get_column_names(self):
        return [self.name]


# ========== TAClient ==========

class TAClient:
    """Client to manage and calculate technical indicators."""
    
    def __init__(self):
        pass
    
    def calculate_indicators(self, df: pl.DataFrame, indicators: list) -> pl.DataFrame:
        """
        Calculate all indicators on the DataFrame.
        
        Args:
            df: Polars DataFrame
            indicators: List of Indicator objects
            
        Returns:
            DataFrame with all calculated indicators
        """
        for indicator in indicators:
            if isinstance(indicator, Indicator):
                df = indicator.calculate(df)
        return df
    
    
INDICATOR_CLASSES = {
    'SMA': SMA,
    'RSI': RSI,
    'MACD': MACD,
    'BollingerBands': BollingerBands,
    'ATR': ATR,
    'Volatility': Volatility,
    'EngulfingPattern': EngulfingPattern,
    'OBV': OBV,
    'VWAP': VWAP,
    'ADX': ADX,
    'Stochastic': Stochastic,
    'WilliamsR': WilliamsR,
    'CCI': CCI,
}


# Usage example
if __name__ == "__main__":
    
    # Get data
    yahoo_client = YahooFinanceClient()
    df = yahoo_client.get_price("AAPL", from_date="2023-01-01", to_date="2024-12-31")
    
    # Define indicators (technical calculations only)
    indicators = [
        SMA(50),
        SMA(200),
        EMA(12),
        RSI(14),
        MACD(),
        BollingerBands(),
        ATR(),
        OBV(),
        VWAP()
    ]
    
    # Calculate indicators
    ta_client = TAClient()
    df = ta_client.calculate_indicators(df, indicators)
    
    print(df.select(['datetime', 'close', 'SMA50', 'SMA200', 'RSI']).tail(10))