import requests
import polars as pl
from typing import Optional, Dict, Any, Union
from datetime import datetime


class YahooFinanceClient:
    """Client basique pour interagir avec Yahoo Finance."""
    
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_quote(self, symbol: str, from_date: Optional[str] = None, 
                  to_date: Optional[str] = None, interval: str = '1d') -> Optional[Dict[str, Any]]:
        """
        Récupère les informations de cotation pour un symbole donné.
        
        Args:
            symbol: Le symbole boursier (ex: 'AAPL', 'MSFT')
            from_date: Date de début (timestamp Unix ou format 'YYYY-MM-DD'). Si None, retourne la dernière cotation
            to_date: Date de fin (timestamp Unix ou format 'YYYY-MM-DD'). Si None, utilise aujourd'hui
            interval: Intervalle des données ('1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')
            
        Returns:
            Dictionnaire contenant les données de cotation ou None en cas d'erreur
        """
        url = f"{self.BASE_URL}/chart/{symbol}"
        params = {'interval': interval}
        
        # Si aucune date n'est fournie, utiliser range pour la dernière cotation
        if from_date is None and to_date is None:
            params['range'] = '1d'
        else:
            # Gérer les dates
            if from_date:
                # Convertir la date si nécessaire
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
            print(f"Erreur lors de la requête: {e}")
            return None
    
    def get_market_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations générales sur le marché (timezone, heures d'ouverture officielles).
        
        Args:
            symbol: Le symbole boursier
            
        Returns:
            Dictionnaire contenant les informations sur le marché
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
                    # Heures d'ouverture/fermeture officielles
                    'regular': current_period.get('regular', {}),
                    'pre': current_period.get('pre', {}),
                    'post': current_period.get('post', {})
                }
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête: {e}")
            return None
    
    def get_trading_hours(self, symbol: str, from_date: str, to_date: Optional[str] = None, 
                         interval: str = '1h') -> Optional[pl.DataFrame]:
        """
        Récupère les heures de trading et identifie les périodes de nuit (marché fermé).
        
        Args:
            symbol: Le symbole boursier
            from_date: Date de début (format 'YYYY-MM-DD')
            to_date: Date de fin (format 'YYYY-MM-DD'), si None utilise aujourd'hui
            interval: Intervalle ('1h', '30m', '15m', '5m', '1m')
            
        Returns:
            DataFrame avec datetime, is_market_hours, is_night
        """
        # Récupérer les données intraday
        df = self.get_price(symbol, from_date=from_date, to_date=to_date, 
                           interval=interval, postclean=False)
        
        if df is None:
            return None
        
        # Extraire l'heure de chaque timestamp
        df = df.with_columns([
            pl.col('datetime').dt.hour().alias('hour'),
            pl.col('datetime').dt.weekday().alias('weekday'),  # 0=lundi, 6=dimanche
        ])
        
        # Identifier les heures de marché (approximation pour US markets: 9h30-16h EST)
        # Note: Pour être plus précis, il faudrait prendre en compte le timezone
        df = df.with_columns([
            # Marché ouvert si: jour de semaine (0-4) ET pendant les heures de trading
            ((pl.col('weekday') < 5) & 
             (pl.col('volume') > 0) &
             (pl.col('open').is_not_null())
            ).alias('is_market_hours'),
            
            # Nuit = pas pendant les heures de marché
            ~((pl.col('weekday') < 5) & 
              (pl.col('volume') > 0) &
              (pl.col('open').is_not_null())
            ).alias('is_night')
        ])
        
        return df.select(['datetime', 'hour', 'weekday', 'is_market_hours', 'is_night', 'volume'])
    
    def get_trading_days(self, symbol: str, from_date: str, to_date: Optional[str] = None) -> Optional[pl.DataFrame]:
        """
        Récupère les jours de trading effectifs sur une période.
        
        Args:
            symbol: Le symbole boursier
            from_date: Date de début (format 'YYYY-MM-DD')
            to_date: Date de fin (format 'YYYY-MM-DD'), si None utilise aujourd'hui
            
        Returns:
            DataFrame avec les dates de trading (datetime) et si le marché était ouvert
        """
        # Récupérer les données sur la période avec un interval journalier
        df = self.get_price(symbol, from_date=from_date, to_date=to_date, interval='1d', postclean=False)
        
        if df is None:
            return None
        
        # Identifier les jours de trading (où il y a eu de l'activité)
        df = df.with_columns([
            # Un jour de trading a des valeurs non-nulles et du volume > 0
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
        Retire les périodes où le marché est fermé (weekends, jours fériés).
        
        Cette méthode filtre automatiquement les lignes sans activité de marché.
        
        Args:
            df: DataFrame avec les données de prix
            symbol: Le symbole boursier (non utilisé actuellement, pour extension future)
            
        Returns:
            DataFrame nettoyé sans les périodes fermées
        """
        # Méthode simple : retirer les lignes où toutes les valeurs OHLC sont nulles ou identiques
        # Cela indique généralement une période de marché fermé
        
        # Filtrer les lignes où au moins une valeur OHLC est non-nulle
        df_clean = df.filter(
            (pl.col('open').is_not_null()) &
            (pl.col('high').is_not_null()) &
            (pl.col('low').is_not_null()) &
            (pl.col('close').is_not_null())
        )
        
        # Filtrer les lignes où il y a eu une variation de prix (pas de marché fermé)
        # Si high == low == open == close et volume == 0, c'est probablement fermé
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
        Récupère le prix d'un symbole.
        
        Args:
            symbol: Le symbole boursier
            from_date: Date de début (timestamp Unix ou format 'YYYY-MM-DD'). Si None, retourne le dernier prix
            to_date: Date de fin (timestamp Unix ou format 'YYYY-MM-DD'). Si None, utilise aujourd'hui
            interval: Intervalle des données ('1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')
            postclean: Si True, retire les périodes où le marché est fermé
            
        Returns:
            Le prix actuel (float) si pas de dates, ou DataFrame Polars avec historique si dates fournies
        """
        # Si aucune date n'est fournie, retourner le dernier prix
        if from_date is None and to_date is None:
            quote = self.get_quote(symbol)
            if quote and 'meta' in quote:
                return quote['meta'].get('regularMarketPrice')
            return None
        
        # Sinon, récupérer l'historique des prix
        url = f"{self.BASE_URL}/chart/{symbol}"
        params = {'interval': interval}
        
        # Gérer les dates
        if from_date:
            # Convertir la date si nécessaire
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
                
                # Créer un DataFrame Polars
                df = pl.DataFrame({
                    'timestamp': timestamps,
                    'datetime': [datetime.fromtimestamp(ts) for ts in timestamps],
                    'open': quote_data.get('open', []),
                    'high': quote_data.get('high', []),
                    'low': quote_data.get('low', []),
                    'close': quote_data.get('close', []),
                    'volume': quote_data.get('volume', [])
                })
                
                # Nettoyer les périodes fermées si demandé
                if postclean:
                    df = self._remove_closed_periods(df, symbol)
                
                return df
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête: {e}")
            return None

# Exemple d'utilisation
if __name__ == "__main__":
    client = YahooFinanceClient()
    
    # Récupérer le dernier prix d'Apple
    price = client.get_price("AAPL")
    if price:
        print(f"Dernier prix d'Apple: ${price}")
    
    # Récupérer les heures de trading et identifier les nuits
    three_days_ago = (today - timedelta(days=3)).strftime('%Y-%m-%d')
    
    trading_hours = client.get_trading_hours("AAPL", from_date=three_days_ago, interval='1h')
    if trading_hours is not None:
        print(f"\n=== Heures de trading vs nuits (derniers 3 jours) ===")
        print(trading_hours)
        
        market_hours = trading_hours.filter(pl.col('is_market_hours'))
        night_hours = trading_hours.filter(pl.col('is_night'))
        
        print(f"\nHeures de marché ouvert: {len(market_hours)}")
        print(f"Heures de nuit/fermé: {len(night_hours)}")
        
        # Afficher quelques exemples de nuits
        print(f"\nExemples de périodes de nuit:")
        print(night_hours.head(10))
    
    # Récupérer les jours de trading sur les 10 derniers jours
    from datetime import datetime, timedelta
    today = datetime.now()
    ten_days_ago = (today - timedelta(days=10)).strftime('%Y-%m-%d')
    
    trading_days = client.get_trading_days("AAPL", from_date=ten_days_ago)
    if trading_days is not None:
        print(f"\n=== Jours de trading sur les 10 derniers jours ===")
        print(trading_days)
        
        trading_count = trading_days.filter(pl.col('is_trading_day'))
        print(f"\nNombre de jours de trading: {len(trading_count)}")
    
    # Récupérer les périodes de trading
    trading_periods = client.get_trading_periods("AAPL")
    if trading_periods:
        print(f"\nFuseau horaire: {trading_periods['timezone']}")
        print(f"Exchange: {trading_periods['exchangeTimezoneName']}")
    
    # Récupérer l'historique avec nettoyage des périodes fermées
    df = client.get_price("AAPL", from_date="2024-01-01", to_date="2024-01-31", 
                         interval="1d", postclean=True)
    if df is not None:
        print(f"\n{df}")
        print(f"\nNombre de jours de trading: {len(df)}")
    
    # Comparer avec/sans nettoyage
    df_raw = client.get_price("AAPL", from_date="2024-01-01", to_date="2024-01-31", 
                              interval="1d", postclean=False)
    df_clean = client.get_price("AAPL", from_date="2024-01-01", to_date="2024-01-31", 
                                interval="1d", postclean=True)
    
    if df_raw is not None and df_clean is not None:
        print(f"\nAvant nettoyage: {len(df_raw)} lignes")
        print(f"Après nettoyage: {len(df_clean)} lignes")
        print(f"Lignes retirées: {len(df_raw) - len(df_clean)}")