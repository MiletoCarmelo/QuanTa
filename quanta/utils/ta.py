import polars as pl
from abc import ABC, abstractmethod
from quanta.clients.yfinance import YahooFinanceClient

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
    """Stochastic Oscillator."""
    
    def __init__(self, period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
        self.period = period
        self.smooth_k = smooth_k
        self.smooth_d = smooth_d
    
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        lowest_low = pl.col('low').rolling_min(window_size=self.period)
        highest_high = pl.col('high').rolling_max(window_size=self.period)
        
        k = ((pl.col('close') - lowest_low) / (highest_high - lowest_low)) * 100
        k_smooth = k.rolling_mean(window_size=self.smooth_k)
        d = k_smooth.rolling_mean(window_size=self.smooth_d)
        
        return df.with_columns([
            k_smooth.alias('STOCH_K'),
            d.alias('STOCH_D')
        ])
    
    def get_column_names(self):
        return ['STOCH_K', 'STOCH_D']


class CCI(Indicator):
    """Commodity Channel Index."""
    
    def __init__(self, period: int = 20):
        self.period = period
    
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        tp = (pl.col('high') + pl.col('low') + pl.col('close')) / 3
        sma = tp.rolling_mean(window_size=self.period)
        mad = (tp - sma).abs().rolling_mean(window_size=self.period)
        cci = (tp - sma) / (0.015 * mad)
        
        return df.with_columns(cci.alias('CCI'))
    
    def get_column_names(self):
        return ['CCI']


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
        OBV()
    ]
    
    # Calculate indicators
    ta_client = TAClient()
    df = ta_client.calculate_indicators(df, indicators)
    
    print(df.select(['datetime', 'close', 'SMA50', 'SMA200', 'RSI']).tail(10))