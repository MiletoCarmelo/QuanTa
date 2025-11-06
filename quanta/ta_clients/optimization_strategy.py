from typing import Any, Dict, Tuple
import numpy as np
import polars as pl
import optuna
from quanta.utils.ta import SMA, RSI, MACD, BollingerBands, INDICATOR_CLASSES


# Configuration JSON avec structure logique
OPTIMIZATION_CONFIG = {
    "strategy_name": "simple_strategy",
    "indicators": {
        "SMA_short": {
            "class": "SMA",
            "params": {
                "period": {
                    "type": "int",
                    "low": 5,
                    "high": 50
                }
            }
        },
        "SMA_long": {
            "class": "SMA",
            "params": {
                "period": {
                    "type": "int",
                    "low": 20,
                    "high": 200
                }
            }
        },
        "RSI": {
            "class": "RSI",
            "params": {
                "period": {
                    "type": "int",
                    "low": 14,
                    "high": 21
                }
            }
        },
        "ATR": {
            "class": "ATR",
            "params": {
                "period": {
                    "type": "int",
                    "low": 10,    # ← Minimum (très réactif)
                    "high": 30      # ← Maximum (plus stable)
                }
            }
        },
        "BollingerBands": {
            "class": "BollingerBands",
            "params": {
                "period": {
                    "type": "int",
                    "low": 15,
                    "high": 30
                },
                "std_dev": {
                    "type": "float",
                    "low": 2.0,
                    "high": 3.0
                }
            }
        },
        "MACD": {
            "class": "MACD",
            "params": {
                "fast_period": {
                    "type": "int",
                    "low": 8,
                    "high": 16
                },
                "slow_period": {
                    "type": "int",
                    "low": 20,
                    "high": 30
                },
                "signal_period": {
                    "type": "int",
                    "low": 7,
                    "high": 12
                }
            }
        }
    },
    "strategy": {
        "stop_loss": {
            "type": "float",
            "low": 0.005,
            "high": 0.02,
            "log": True
        },
        "take_profit": {
            "type": "float",
            "low": 0.01,
            "high": 0.05,
            "log": True
        },
        "position_size": {
            "type": "float",
            "low": 0.2,
            "high": 0.8
        },
        "macd_hist_buy_threshold": {
            "type": "float",
            "low": -1.0,  # Élargi pour être moins restrictif (permet plus de signaux)
            "high": 0.5,
            "description": "Seuil MACD_hist pour éviter d'acheter pendant chute libre (doit être > cette valeur). Valeur plus basse = moins restrictif."
        },
        "macd_hist_sell_threshold": {
            "type": "float",
            "low": -0.5,
            "high": 1.0,  # Élargi pour être moins restrictif (permet plus de signaux)
            "description": "Seuil MACD_hist pour éviter de vendre pendant montée forte (doit être < cette valeur). Valeur plus haute = moins restrictif."
        }
    },
    "constraints": [
        
        # 1. basis contrainte : 
        {"condition": "SMA_long_period >= SMA_short_period + 50"}, # Assurer que la période courte est inférieure à la période longue et Forcer une VRAIE différence entre les SMA
        {"condition": "MACD_fast_period < MACD_slow_period"}, # (convention) C'est une convention en analyse technique, EMA rapide doit être inférieure à EMA lente. Le MACD (Moving Average Convergence Divergence) compare deux moyennes mobiles exponentielles (EMA) de différentes périodes pour identifier les tendances et les points de retournement potentiels dans le prix d'un actif.
        {"condition": "MACD_signal_period < MACD_slow_period"}, # (convention) Signal MACD doit être plus court que slow 
        
        # Ratio risk/reward minimum (bonne pratique)
        {"condition": "take_profit >= 1.2 * stop_loss"},  # Si votre parser supporte les expressions
        
        ## 2. ⚠️ Contraintes métier optionnelles (selon votre stratégie)
        
        # Relation entre indicateurs (si logique stratégique)
        {"condition": "RSI_period < BollingerBands_period"},    # RSI plus réactif que BB
        {"condition": "SMA_short_period < BollingerBands_period"},  # SMA court < BB pour réactivité
    
    ],
    "optimization": {
        "direction": "maximize",
        "n_trials": 1000,
        "n_jobs": 1,
        "timeout": 3600,
        "seed": 42
    }
}


STRATEGIES_IMPLEMENTED = {
    'simple_strategy': {
        'description': 'Stratégie simple basée sur les croisements de SMA et les niveaux de RSI.',
        'parameters': ['short_window', 'long_window', 'rsi_period', 'stop_loss', 'atr_period', 'take_profit', 'position_size', 'macd_hist_buy_threshold', 'macd_hist_sell_threshold'],
        'required_indicators': ['SMA_short', 'SMA_long', 'RSI', 'MACD', 'ATR'],
        'function': 'simple_strategy_fct'
    }
}


class BacktestStrategyCore:
    """Contient toutes les stratégies de trading implémentées."""
    
    @staticmethod
    def simple_strategy_fct(
            df: pl.DataFrame, 
            short_window: int, 
            long_window: int,
            rsi_period: int, 
            atr_period: int,
            stop_loss: float, 
            take_profit: float,
            position_size: float,
            macd_hist_buy_threshold: float = -0.5,   # Seuil pour éviter d'acheter pendant chute libre (défaut moins restrictif)
            macd_hist_sell_threshold: float = 0.5,  # Seuil pour éviter de vendre pendant montée forte (défaut moins restrictif)
            return_trades: bool = False
    ) -> dict:
        df = df.clone().drop_nulls()
        
        if len(df) == 0:
            return {"total_return": 0.0, "sharpe_ratio": 0.0, "cagr": 0.0, "max_drawdown": 0.0}
        
        # Calculer la moyenne de l'ATR pour le filtre de volatilité
        atr_mean = df[f'ATR{atr_period}'].mean()
    
        # Générer les signaux
        # 
        # AMÉLIORATION : Ajout de filtres pour éviter les signaux prématurés pendant les tendances fortes
        # - Ne pas VENDRE pendant une forte tendance haussière (momentum positif fort)
        # - Ne pas ACHETER pendant une forte tendance baissière (momentum négatif fort)
        # Utilisation de MACD pour détecter la force du momentum
        
        # Construire les conditions de base (moins restrictives pour plus de signaux)
        # Conditions assouplies : thresholds SMA réduits, RSI levels élargis, ATR filter assoupli
        buy_base_condition = (
            (pl.col(f'SMA{short_window}') < pl.col(f'SMA{long_window}') * 0.995) &  # Réduit de 0.98 à 0.995 (moins restrictif)
            (pl.col(f'RSI{rsi_period}') < 45) &  # Élargi de 40 à 45 (moins restrictif)
            (pl.col(f'ATR{atr_period}') > atr_mean * 0.6)  # Réduit de 0.8 à 0.6 (moins restrictif)
        )
        
        sell_base_condition = (
            (pl.col(f'SMA{short_window}') > pl.col(f'SMA{long_window}') * 1.005) &  # Réduit de 1.02 à 1.005 (moins restrictif)
            (pl.col(f'RSI{rsi_period}') > 55) &  # Réduit de 60 à 55 (moins restrictif)
            (pl.col(f'ATR{atr_period}') > atr_mean * 0.6)  # Réduit de 0.8 à 0.6 (moins restrictif)
        )
        
        # Ajouter le filtre MACD si disponible (évite les signaux pendant tendances fortes)
        # Les seuils sont maintenant optimisables via macd_hist_buy_threshold et macd_hist_sell_threshold
        if 'MACD_hist' in df.columns:
            # Ne pas acheter si momentum baissier très fort (chute libre)
            # MACD_hist doit être supérieur au seuil pour permettre l'achat
            buy_condition = buy_base_condition & (pl.col('MACD_hist') > macd_hist_buy_threshold)
            # Ne pas vendre si momentum haussier très fort (montée forte)
            # MACD_hist doit être inférieur au seuil pour permettre la vente
            sell_condition = sell_base_condition & (pl.col('MACD_hist') < macd_hist_sell_threshold)
        else:
            # Si MACD n'existe pas, utiliser les conditions de base
            buy_condition = buy_base_condition
            sell_condition = sell_base_condition
        
        df = df.with_columns([
            pl.when(
                # LONG (ACHETER) : Acheter quand prix est BAS (creux)
                # Condition : SMA court < SMA long (tendance baissière) + RSI survendu = bon moment pour ACHETER au creux
                # FILTRE ANTI-TENDANCE FORTE : Ne pas acheter si momentum baissier est très fort (évite chute libre)
                buy_condition
            ).then(1)  # Signal LONG = on ACHÈTE (BUY) - maintenant au creux ✓
            .when(
                # SHORT (VENDRE) : Vendre quand prix est HAUT (sommet)
                # Condition : SMA court > SMA long (tendance haussière) + RSI suracheté = bon moment pour VENDRE au sommet
                # FILTRE ANTI-TENDANCE FORTE : Ne pas vendre si momentum haussier est très fort (évite montée forte)
                sell_condition
            ).then(-1)  # Signal SHORT = on VEND (SELL) - maintenant au sommet ✓
            .otherwise(0)
            .alias('signal')
        ])
        
        
        # Calculer les rendements
        df = df.with_columns([
            pl.col('close').pct_change().alias('price_returns'),
            pl.col('signal').shift(1).fill_null(0).alias('signal_shifted')
        ])
        
        # Appliquer les stops correctement
        df = df.with_columns([
            pl.when(pl.col('signal_shifted') != 0)
            .then(
                # Calculer le rendement brut
                pl.col('price_returns') * pl.col('signal_shifted') * position_size
            )
            .otherwise(0.0)
            .alias('raw_returns')
        ])
        
        # Appliquer stop loss et take profit
        df = df.with_columns([
            pl.when(pl.col('raw_returns') < -stop_loss)
            .then(-stop_loss * position_size)  # Limiter la perte
            .when(pl.col('raw_returns') > take_profit)
            .then(take_profit * position_size)  # Limiter le gain
            .otherwise(pl.col('raw_returns'))
            .alias('strategy_returns')
        ])
        
        returns = df['strategy_returns'].drop_nulls()
        
        # DEBUG: Afficher les statistiques
        num_total_signals = (df['signal'] != 0).sum()
        print(f"  Signaux générés: {num_total_signals}")
        
        if len(returns) == 0 or num_total_signals == 0:
            return {"total_return": 0.0, "sharpe_ratio": 0.0, "cagr": 0.0, "max_drawdown": 0.0}
        
        # Juste après la création des signaux
        num_long_signals = (df['signal'] == 1).sum()
        num_short_signals = (df['signal'] == -1).sum()
        num_actual_trades = (df['signal_shifted'] != 0).sum()

        print(f"  📊 Signaux totaux: {num_total_signals} (Long: {num_long_signals}, Short: {num_short_signals})")
        print(f"  💼 Trades exécutés: {num_actual_trades}")

        
        total_return = returns.sum()
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252 * 390) if std_ret and std_ret > 0 else 0.0
        
        # 3. CAGR (Compound Annual Growth Rate)
        # Calculer la courbe de capital (cumulative returns)
        df = df.with_columns([
            (1 + pl.col('strategy_returns')).cum_prod().alias('cumulative_returns')
        ])
        
        cumulative = df['cumulative_returns'].to_list()
        final_value = cumulative[-1] if cumulative else 1.0
        initial_value = 1.0

        # CORRECTION : Utiliser les dates réelles
        if 'datetime' in df.columns:
            first_date = df['datetime'][0]
            last_date = df['datetime'][-1]
            
            # Calculer la différence en jours
            if hasattr(first_date, 'date'):  # Si c'est un datetime
                days_diff = (last_date - first_date).days
            else:  # Si c'est déjà un timedelta ou autre
                days_diff = (last_date - first_date).total_seconds() / 86400
            
            n_years = days_diff / 365.25
            
            print(f"  📅 Période: {first_date} → {last_date} ({days_diff} jours = {n_years:.2f} ans)")
        else:
            # Fallback si pas de colonne datetime
            print("  ⚠️  Pas de colonne 'datetime', estimation basée sur le nombre de périodes")
            n_periods = len(df)
            periods_per_year = 252 * 6.5  # Ajustez selon votre intervalle
            n_years = n_periods / periods_per_year

        # Protection contre division par zéro ou années trop petites
        if n_years > 0.01 and final_value > 0:  # Minimum 4 jours
            cagr = (final_value / initial_value) ** (1 / n_years) - 1
        else:
            print(f"  ⚠️  Période trop courte ({n_years:.4f} ans), CAGR non calculable")
            cagr = 0.0
        
        # 4. Max Drawdown
        cumulative_max = df['cumulative_returns'].cum_max()
        drawdown = (df['cumulative_returns'] - cumulative_max) / cumulative_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
        
        print(f"  📊 Trades: {num_actual_trades} | Return: {total_return:.4f} | Sharpe: {sharpe_ratio:.2f} | DD: {max_drawdown:.2%}")        
        
        result = {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe_ratio),
            "cagr": float(cagr),
            "max_drawdown": float(max_drawdown),
            "num_signals": int(num_total_signals)  # Pour debug
        }
        
        if return_trades:
            trades_df = df.filter(pl.col('signal_shifted') != 0).select([
                pl.col('datetime').alias('timestamp'),
                pl.col('close').alias('price'),
                pl.col('signal_shifted').alias('signal'),
                pl.lit(position_size).alias('position_size'),
                pl.col('strategy_returns').alias('pnl'),
                pl.col('cumulative_returns').alias('cumulative_capital')
            ])
            
            # Ajouter le type de trade (buy/sell)
            # 
            # ANALYSE DE LA LOGIQUE :
            # - signal_shifted = signal de la période précédente (shift(1))
            # - returns = price_returns * signal_shifted
            #   → signal=1 : on gagne si prix monte → position LONG → on ACHÈTE (BUY)
            #   → signal=-1 : on gagne si prix baisse → position SHORT → on VEND (SELL)
            # 
            # PROBLÈME POTENTIEL : La logique de génération des signaux peut être inversée :
            # - Signal 1 (LONG) : SMA court > SMA long + 2% ET RSI < 40
            #   → Tendance haussière mais RSI survendu = pullback = bon moment pour ACHETER → BUY ✓
            # - Signal -1 (SHORT) : SMA court < SMA long - 2% ET RSI > 60  
            #   → Tendance baissière mais RSI suracheté = rebond = bon moment pour VENDRE → SELL ✓
            #
            # Le mapping devrait être :
            # - signal = 1 → BUY (ACHETER pour position LONG)
            # - signal = -1 → SELL (VENDRE pour position SHORT)
            trades_df = trades_df.with_columns([
                pl.when(pl.col('signal') == 1)
                .then(pl.lit('BUY'))    # LONG signal = ACHETER
                .when(pl.col('signal') == -1)
                .then(pl.lit('SELL'))   # SHORT signal = VENDRE
                .otherwise(pl.lit('HOLD'))
                .alias('action')
            ])
            
            # Ajouter un numéro de position
            trades_df = trades_df.with_columns([
                pl.arange(0, len(trades_df)).alias('position_number')
            ])
            
            # Calculer la quantité (en % du capital)
            trades_df = trades_df.with_columns([
                (pl.col('position_size') * pl.col('cumulative_capital')).alias('quantity_usd')
            ])
            
            # Réorganiser les colonnes
            trades_df = trades_df.select([
                'timestamp',
                'position_number',
                'action',
                'price',
                'quantity_usd',
                'position_size',
                'pnl',
                'cumulative_capital'
            ])
        
            result["trades_df"] = trades_df

        return result
    

class StrategyClient:
    """
    Classe pour gérer la configuration d'optimisation des stratégies de trading.
    """

    def __init__(self):
        self.strategy_core = BacktestStrategyCore()

    def save_config(self, config: Dict[str, Any], output_path: str) -> None:
        """
        Sauvegarde la configuration d'optimisation dans un fichier JSON.

        Args:
            config (Dict[str, Any]): Configuration d'optimisation.
            output_path (str): Chemin du fichier de sortie.
        """
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)

    def get_seed_optimization_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration d'optimisation par défaut.
        
        Returns:
            Configuration par défaut
        """
        return OPTIMIZATION_CONFIG
                
    def get_seed_indicator_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration des indicateurs par défaut.

        Returns:
            Configuration par défaut
        """
        return INDICATOR_CLASSES
    
    def print_strategies(self):
        """Affiche les stratégies disponibles."""
        print("Stratégies disponibles :")
        for name, info in STRATEGIES_IMPLEMENTED.items():
            print(f"\n  {name}:")
            print(f"    Description: {info['description']}")
            print(f"    Indicateurs requis: {info['required_indicators']}")
            
    def get_strategy_fct(self, strategy_name: str):
        """
        Retourne la fonction de stratégie correspondant au nom donné.

        Args:
            strategy_name (str): Nom de la stratégie.

        Returns:
            Fonction de la stratégie.
        """
        if strategy_name in STRATEGIES_IMPLEMENTED:
            function_name = STRATEGIES_IMPLEMENTED[strategy_name]['function']
            return getattr(self.strategy_core, function_name)
        else:
            raise ValueError(f"Stratégie '{strategy_name}' non implémentée.")
        