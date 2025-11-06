import requests
import polars as pl
from typing import Optional, Dict, Any, Union
from datetime import datetime


class YahooFinanceClient:
    """Basic client to interact with Yahoo Finance."""
    
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_quote(self, symbol: str, from_date: Optional[str] = None, 
                  to_date: Optional[str] = None, interval: str = '1d') -> Optional[Dict[str, Any]]:
        """
        Retrieves quote information for a given symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            from_date: Start date (Unix timestamp or 'YYYY-MM-DD' format). If None, returns latest quote
            to_date: End date (Unix timestamp or 'YYYY-MM-DD' format). If None, uses today
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')
            
        Returns:
            Dictionary containing quote data or None on error
        """
        url = f"{self.BASE_URL}/chart/{symbol}"
        params = {'interval': interval}
        
        # If no date is provided, use range for latest quote
        if from_date is None and to_date is None:
            params['range'] = '1d'
        else:
            # Handle dates
            if from_date:
                # Convert date if necessary
                if isinstance(from_date, str) and '-' in from_date:
                    from datetime import datetime
                    from_timestamp = int(datetime.strptime(from_date, '%Y-%m-%d').timestamp())
                    params['period1'] = from_timestamp
                else:
                    params['period1'] = from_date
            
            if to_date:
                if isinstance(to_date, str) and '-' in to_date:
                    from datetime import datetime
                    to_timestamp = int(datetime.strptime(to_date, '%Y-%m-%d').timestamp())
                    params['period2'] = to_timestamp
                else:
                    params['period2'] = to_date
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart']:
                return data['chart']['result'][0]
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
    
    def get_market_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves general market information (timezone, official opening hours).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing market information
        """
        url = f"{self.BASE_URL}/chart/{symbol}"
        params = {'range': '1d', 'interval': '1d'}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart']:
                result = data['chart']['result'][0]
                meta = result.get('meta', {})
                current_period = meta.get('currentTradingPeriod', {})
                
                return {
                    'symbol': meta.get('symbol'),
                    'timezone': meta.get('timezone'),
                    'exchange': meta.get('exchangeName'),
                    'exchangeTimezoneName': meta.get('exchangeTimezoneName'),
                    'currency': meta.get('currency'),
                    'regularMarketTime': meta.get('regularMarketTime'),
                    'regularMarketPrice': meta.get('regularMarketPrice'),
                    # Official opening/closing hours
                    'regular': current_period.get('regular', {}),
                    'pre': current_period.get('pre', {}),
                    'post': current_period.get('post', {})
                }
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
    
    def get_trading_hours(self, symbol: str, from_date: str, to_date: Optional[str] = None, 
                         interval: str = '1h') -> Optional[pl.DataFrame]:
        """
        Retrieves trading hours and identifies night periods (market closed).
        
        Args:
            symbol: Stock symbol
            from_date: Start date ('YYYY-MM-DD' format)
            to_date: End date ('YYYY-MM-DD' format), if None uses today
            interval: Interval ('1h', '30m', '15m', '5m', '1m')
            
        Returns:
            DataFrame with datetime, is_market_hours, is_night
        """
        # Get intraday data
        df = self.get_price(symbol, from_date=from_date, to_date=to_date, 
                           interval=interval, postclean=False)
        
        if df is None:
            return None
        
        # Extract hour from each timestamp
        df = df.with_columns([
            pl.col('datetime').dt.hour().alias('hour'),
            pl.col('datetime').dt.weekday().alias('weekday'),  # 0=Monday, 6=Sunday
        ])
        
        # Identify market hours (approximation for US markets: 9:30am-4pm EST)
        # Note: For more precision, timezone should be taken into account
        df = df.with_columns([
            # Market open if: weekday (0-4) AND during trading hours
            ((pl.col('weekday') < 5) & 
             (pl.col('volume') > 0) &
             (pl.col('open').is_not_null())
            ).alias('is_market_hours'),
            
            # Night = not during market hours
            ~((pl.col('weekday') < 5) & 
              (pl.col('volume') > 0) &
              (pl.col('open').is_not_null())
            ).alias('is_night')
        ])
        
        return df.select(['datetime', 'hour', 'weekday', 'is_market_hours', 'is_night', 'volume'])
    
    def get_trading_days(self, symbol: str, from_date: str, to_date: Optional[str] = None) -> Optional[pl.DataFrame]:
        """
        Retrieves actual trading days over a period.
        
        Args:
            symbol: Stock symbol
            from_date: Start date ('YYYY-MM-DD' format)
            to_date: End date ('YYYY-MM-DD' format), if None uses today
            
        Returns:
            DataFrame with trading dates (datetime) and whether market was open
        """
        # Get data for the period with daily interval
        df = self.get_price(symbol, from_date=from_date, to_date=to_date, interval='1d', postclean=False)
        
        if df is None:
            return None
        
        # Identify trading days (where there was activity)
        df = df.with_columns([
            # A trading day has non-null values and volume > 0
            ((pl.col('open').is_not_null()) & 
             (pl.col('volume') > 0) &
             ~((pl.col('high') == pl.col('low')) & 
               (pl.col('high') == pl.col('open')) & 
               (pl.col('high') == pl.col('close')) &
               (pl.col('volume') == 0))
            ).alias('is_trading_day')
        ])
        
        return df.select(['datetime', 'is_trading_day'])
    
    def _remove_closed_periods(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """
        Removes periods when market is closed (weekends, holidays).
        
        This method automatically filters rows with no market activity.
        
        Args:
            df: DataFrame with price data
            symbol: Stock symbol (not currently used, for future extension)
            
        Returns:
            Cleaned DataFrame without closed periods
        """
        # Simple method: remove rows where all OHLC values are null or identical
        # This usually indicates a closed market period
        
        # Filter rows where at least one OHLC value is non-null
        df_clean = df.filter(
            (pl.col('open').is_not_null()) &
            (pl.col('high').is_not_null()) &
            (pl.col('low').is_not_null()) &
            (pl.col('close').is_not_null())
        )
        
        # Filter rows where there was price variation (not closed market)
        # If high == low == open == close and volume == 0, it's probably closed
        df_clean = df_clean.filter(
            ~((pl.col('high') == pl.col('low')) & 
              (pl.col('high') == pl.col('open')) & 
              (pl.col('high') == pl.col('close')) &
              (pl.col('volume') == 0))
        )
        
        return df_clean
 
                      
    def get_price(self, symbol: str, from_date: Optional[str] = None, 
                  to_date: Optional[str] = None, interval: str = '1d',
                  postclean: bool = False) -> Optional[Union[float, pl.DataFrame]]:
        """
        Retrieves the price of a symbol.
        
        Args:
            symbol: Stock symbol
            from_date: Start date (Unix timestamp or 'YYYY-MM-DD' format). If None, returns latest price
            to_date: End date (Unix timestamp or 'YYYY-MM-DD' format). If None, uses today
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')
            postclean: If True, removes periods when market is closed
            
        Returns:
            Current price (float) if no dates, or Polars DataFrame with history if dates provided
        """
        # If no date is provided, return latest price
        if from_date is None and to_date is None:
            quote = self.get_quote(symbol)
            if quote and 'meta' in quote:
                return quote['meta'].get('regularMarketPrice')
            return None
        
        # Otherwise, retrieve price history
        url = f"{self.BASE_URL}/chart/{symbol}"
        params = {'interval': interval}
        
        # Handle dates
        if from_date:
            # Convert date if necessary
            if isinstance(from_date, str) and '-' in from_date:
                from_timestamp = int(datetime.strptime(from_date, '%Y-%m-%d').timestamp())
                params['period1'] = from_timestamp
            else:
                params['period1'] = from_date
        
        if to_date:
            if isinstance(to_date, str) and '-' in to_date:
                to_timestamp = int(datetime.strptime(to_date, '%Y-%m-%d').timestamp())
                params['period2'] = to_timestamp
            else:
                params['period2'] = to_date
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart']:
                result = data['chart']['result'][0]
                timestamps = result.get('timestamp', [])
                indicators = result.get('indicators', {})
                quote_data = indicators.get('quote', [{}])[0]
                
                # Create a Polars DataFrame
                df = pl.DataFrame({
                    'timestamp': timestamps,
                    'datetime': [datetime.fromtimestamp(ts) for ts in timestamps],
                    'open': quote_data.get('open', []),
                    'high': quote_data.get('high', []),
                    'low': quote_data.get('low', []),
                    'close': quote_data.get('close', []),
                    'volume': quote_data.get('volume', [])
                })
                
                # Clean closed periods if requested
                if postclean:
                    df = self._remove_closed_periods(df, symbol)
                
                return df
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None

# Usage example
if __name__ == "__main__":
    client = YahooFinanceClient()
    
    # Get latest Apple price
    price = client.get_price("AAPL")
    if price:
        print(f"Latest Apple price: ${price}")
    
    # Get trading hours and identify nights
    three_days_ago = (today - timedelta(days=3)).strftime('%Y-%m-%d')
    
    trading_hours = client.get_trading_hours("AAPL", from_date=three_days_ago, interval='1h')
    if trading_hours is not None:
        print(f"\n=== Trading hours vs nights (last 3 days) ===")
        print(trading_hours)
        
        market_hours = trading_hours.filter(pl.col('is_market_hours'))
        night_hours = trading_hours.filter(pl.col('is_night'))
        
        print(f"\nMarket open hours: {len(market_hours)}")
        print(f"Night/closed hours: {len(night_hours)}")
        
        # Display some night examples
        print(f"\nNight period examples:")
        print(night_hours.head(10))
    
    # Get trading days over the last 10 days
    from datetime import datetime, timedelta
    today = datetime.now()
    ten_days_ago = (today - timedelta(days=10)).strftime('%Y-%m-%d')
    
    trading_days = client.get_trading_days("AAPL", from_date=ten_days_ago)
    if trading_days is not None:
        print(f"\n=== Trading days over the last 10 days ===")
        print(trading_days)
        
        trading_count = trading_days.filter(pl.col('is_trading_day'))
        print(f"\nNumber of trading days: {len(trading_count)}")
    
    # Get trading periods
    trading_periods = client.get_trading_periods("AAPL")
    if trading_periods:
        print(f"\nTimezone: {trading_periods['timezone']}")
        print(f"Exchange: {trading_periods['exchangeTimezoneName']}")
    
    # Get history with cleaning of closed periods
    df = client.get_price("AAPL", from_date="2024-01-01", to_date="2024-01-31", 
                         interval="1d", postclean=True)
    if df is not None:
        print(f"\n{df}")
        print(f"\nNumber of trading days: {len(df)}")
    
    # Compare with/without cleaning
    df_raw = client.get_price("AAPL", from_date="2024-01-01", to_date="2024-01-31", 
                              interval="1d", postclean=False)
    df_clean = client.get_price("AAPL", from_date="2024-01-01", to_date="2024-01-31", 
                                interval="1d", postclean=True)
    
    if df_raw is not None and df_clean is not None:
        print(f"\nBefore cleaning: {len(df_raw)} rows")
        print(f"After cleaning: {len(df_clean)} rows")
        print(f"Rows removed: {len(df_raw) - len(df_clean)}")