from typing import Any, Dict, Tuple
import numpy as np
import polars as pl
import optuna
from quanta.utils.ta import SMA, RSI, MACD, BollingerBands, INDICATOR_CLASSES


# Configuration JSON with logical structure
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
                    "low": 10,    # ← Minimum (very reactive)
                    "high": 30      # ← Maximum (more stable)
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
            "low": -1.0,  # Widened to be less restrictive (allows more signals)
            "high": 0.5,
            "description": "MACD_hist threshold to avoid buying during free fall (must be > this value). Lower value = less restrictive."
        },
        "macd_hist_sell_threshold": {
            "type": "float",
            "low": -0.5,
            "high": 1.0,  # Widened to be less restrictive (allows more signals)
            "description": "MACD_hist threshold to avoid selling during strong rise (must be < this value). Higher value = less restrictive."
        }
    },
    "constraints": [
        
        # 1. Basic constraints: 
        {"condition": "SMA_long_period >= SMA_short_period + 50"}, # Ensure short period is less than long period and force a TRUE difference between SMAs
        {"condition": "MACD_fast_period < MACD_slow_period"}, # (convention) This is a convention in technical analysis, fast EMA must be less than slow EMA. MACD (Moving Average Convergence Divergence) compares two exponential moving averages (EMA) of different periods to identify trends and potential reversal points in asset price.
        {"condition": "MACD_signal_period < MACD_slow_period"}, # (convention) MACD signal must be shorter than slow 
        
        # Minimum risk/reward ratio (best practice)
        {"condition": "take_profit >= 1.2 * stop_loss"},  # If your parser supports expressions
        
        ## 2. ⚠️ Optional business constraints (depending on your strategy)
        
        # Relationship between indicators (if strategic logic)
        {"condition": "RSI_period < BollingerBands_period"},    # RSI more reactive than BB
        {"condition": "SMA_short_period < BollingerBands_period"},  # Short SMA < BB for reactivity
    
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
        'description': 'Simple strategy based on SMA crossovers and RSI levels.',
        'parameters': ['short_window', 'long_window', 'rsi_period', 'stop_loss', 'atr_period', 'take_profit', 'position_size', 'macd_hist_buy_threshold', 'macd_hist_sell_threshold'],
        'required_indicators': ['SMA_short', 'SMA_long', 'RSI', 'MACD', 'ATR', 'BollingerBands'],
        'function': 'simple_strategy_fct'
    }
}


class BacktestStrategyCore:
    """Contains all implemented trading strategies."""
    
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
            macd_hist_buy_threshold: float = -0.5,   # Threshold to avoid buying during free fall (less restrictive default)
            macd_hist_sell_threshold: float = 0.5,  # Threshold to avoid selling during strong rise (less restrictive default)
            return_trades: bool = False
    ) -> dict:
        df = df.clone().drop_nulls()
        
        if len(df) == 0:
            return {"total_return": 0.0, "sharpe_ratio": 0.0, "cagr": 0.0, "max_drawdown": 0.0}
        
        # Calculate ATR mean for volatility filter
        atr_mean = df[f'ATR{atr_period}'].mean()
    
        # Generate signals
        # 
        # IMPROVEMENT: Adding filters to avoid premature signals during strong trends
        # - Do not SELL during a strong bullish trend (strong positive momentum)
        # - Do not BUY during a strong bearish trend (strong negative momentum)
        # Using MACD to detect momentum strength
        
        # Build base conditions (less restrictive for more signals)
        # Relaxed conditions: reduced SMA thresholds, widened RSI levels, relaxed ATR filter
        buy_base_condition = (
            (pl.col(f'SMA{short_window}') < pl.col(f'SMA{long_window}') * 0.995) &  # Reduced from 0.98 to 0.995 (less restrictive)
            (pl.col(f'RSI{rsi_period}') < 45) &  # Widened from 40 to 45 (less restrictive)
            (pl.col(f'ATR{atr_period}') > atr_mean * 0.6)  # Reduced from 0.8 to 0.6 (less restrictive)
        )
        
        sell_base_condition = (
            (pl.col(f'SMA{short_window}') > pl.col(f'SMA{long_window}') * 1.005) &  # Reduced from 1.02 to 1.005 (less restrictive)
            (pl.col(f'RSI{rsi_period}') > 55) &  # Reduced from 60 to 55 (less restrictive)
            (pl.col(f'ATR{atr_period}') > atr_mean * 0.6)  # Reduced from 0.8 to 0.6 (less restrictive)
        )
        
        # Add MACD filter if available (avoids signals during strong trends)
        # Thresholds are now optimizable via macd_hist_buy_threshold and macd_hist_sell_threshold
        if 'MACD_hist' in df.columns:
            # Do not buy if bearish momentum is very strong (free fall)
            # MACD_hist must be greater than threshold to allow buying
            buy_condition = buy_base_condition & (pl.col('MACD_hist') > macd_hist_buy_threshold)
            # Do not sell if bullish momentum is very strong (strong rise)
            # MACD_hist must be less than threshold to allow selling
            sell_condition = sell_base_condition & (pl.col('MACD_hist') < macd_hist_sell_threshold)
        else:
            # If MACD doesn't exist, use base conditions
            buy_condition = buy_base_condition
            sell_condition = sell_base_condition
        
        df = df.with_columns([
            pl.when(
                # LONG (BUY): Buy when price is LOW (trough)
                # Condition: Short SMA < Long SMA (bearish trend) + oversold RSI = good time to BUY at the trough
                # ANTI-STRONG-TREND FILTER: Do not buy if bearish momentum is very strong (avoids free fall)
                buy_condition
            ).then(1)  # LONG signal = we BUY - now at the trough ✓
            .when(
                # SHORT (SELL): Sell when price is HIGH (peak)
                # Condition: Short SMA > Long SMA (bullish trend) + overbought RSI = good time to SELL at the peak
                # ANTI-STRONG-TREND FILTER: Do not sell if bullish momentum is very strong (avoids strong rise)
                sell_condition
            ).then(-1)  # SHORT signal = we SELL - now at the peak ✓
            .otherwise(0)
            .alias('signal')
        ])
        
        
        # Calculate returns
        df = df.with_columns([
            pl.col('close').pct_change().alias('price_returns'),
            pl.col('signal').shift(1).fill_null(0).alias('signal_shifted')
        ])
        
        # Apply stops correctly
        df = df.with_columns([
            pl.when(pl.col('signal_shifted') != 0)
            .then(
                # Calculate raw return
                pl.col('price_returns') * pl.col('signal_shifted') * position_size
            )
            .otherwise(0.0)
            .alias('raw_returns')
        ])
        
        # Apply stop loss and take profit
        df = df.with_columns([
            pl.when(pl.col('raw_returns') < -stop_loss)
            .then(-stop_loss * position_size)  # Limit loss
            .when(pl.col('raw_returns') > take_profit)
            .then(take_profit * position_size)  # Limit gain
            .otherwise(pl.col('raw_returns'))
            .alias('strategy_returns')
        ])
        
        returns = df['strategy_returns'].drop_nulls()
        
        # DEBUG: Display statistics
        num_total_signals = (df['signal'] != 0).sum()
        print(f"  Signals generated: {num_total_signals}")
        
        if len(returns) == 0 or num_total_signals == 0:
            return {"total_return": 0.0, "sharpe_ratio": 0.0, "cagr": 0.0, "max_drawdown": 0.0}
        
        # Right after signal creation
        num_long_signals = (df['signal'] == 1).sum()
        num_short_signals = (df['signal'] == -1).sum()
        num_actual_trades = (df['signal_shifted'] != 0).sum()

        print(f"  📊 Total signals: {num_total_signals} (Long: {num_long_signals}, Short: {num_short_signals})")
        print(f"  💼 Executed trades: {num_actual_trades}")

        
        total_return = returns.sum()
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252 * 390) if std_ret and std_ret > 0 else 0.0
        
        # 3. CAGR (Compound Annual Growth Rate)
        # Calculate capital curve (cumulative returns)
        df = df.with_columns([
            (1 + pl.col('strategy_returns')).cum_prod().alias('cumulative_returns')
        ])
        
        cumulative = df['cumulative_returns'].to_list()
        final_value = cumulative[-1] if cumulative else 1.0
        initial_value = 1.0

        # CORRECTION: Use actual dates
        if 'datetime' in df.columns:
            first_date = df['datetime'][0]
            last_date = df['datetime'][-1]
            
            # Calculate difference in days
            if hasattr(first_date, 'date'):  # If it's a datetime
                days_diff = (last_date - first_date).days
            else:  # If it's already a timedelta or other
                days_diff = (last_date - first_date).total_seconds() / 86400
            
            n_years = days_diff / 365.25
            
            print(f"  📅 Period: {first_date} → {last_date} ({days_diff} days = {n_years:.2f} years)")
        else:
            # Fallback if no datetime column
            print("  ⚠️  No 'datetime' column, estimation based on number of periods")
            n_periods = len(df)
            periods_per_year = 252 * 6.5  # Adjust according to your interval
            n_years = n_periods / periods_per_year

        # Protection against division by zero or too small years
        if n_years > 0.01 and final_value > 0:  # Minimum 4 days
            cagr = (final_value / initial_value) ** (1 / n_years) - 1
        else:
            print(f"  ⚠️  Period too short ({n_years:.4f} years), CAGR not calculable")
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
            "num_signals": int(num_total_signals)  # For debug
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
            
            # Add trade type (buy/sell)
            # 
            # LOGIC ANALYSIS:
            # - signal_shifted = signal from previous period (shift(1))
            # - returns = price_returns * signal_shifted
            #   → signal=1 : we profit if price goes up → LONG position → we BUY
            #   → signal=-1 : we profit if price goes down → SHORT position → we SELL
            # 
            # POTENTIAL ISSUE: Signal generation logic may be inverted:
            # - Signal 1 (LONG): Short SMA > Long SMA + 2% AND RSI < 40
            #   → Bullish trend but oversold RSI = pullback = good time to BUY → BUY ✓
            # - Signal -1 (SHORT): Short SMA < Long SMA - 2% AND RSI > 60  
            #   → Bearish trend but overbought RSI = bounce = good time to SELL → SELL ✓
            #
            # The mapping should be:
            # - signal = 1 → BUY (BUY for LONG position)
            # - signal = -1 → SELL (SELL for SHORT position)
            trades_df = trades_df.with_columns([
                pl.when(pl.col('signal') == 1)
                .then(pl.lit('BUY'))    # LONG signal = BUY
                .when(pl.col('signal') == -1)
                .then(pl.lit('SELL'))   # SHORT signal = SELL
                .otherwise(pl.lit('HOLD'))
                .alias('action')
            ])
            
            # Add position number
            trades_df = trades_df.with_columns([
                pl.arange(0, len(trades_df)).alias('position_number')
            ])
            
            # Calculate quantity (in % of capital)
            trades_df = trades_df.with_columns([
                (pl.col('position_size') * pl.col('cumulative_capital')).alias('quantity_usd')
            ])
            
            # Reorganize columns
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
    Class to manage trading strategy optimization configuration.
    """

    def __init__(self):
        self.strategy_core = BacktestStrategyCore()

    def save_config(self, config: Dict[str, Any], output_path: str) -> None:
        """
        Saves the optimization configuration to a JSON file.

        Args:
            config (Dict[str, Any]): Optimization configuration.
            output_path (str): Output file path.
        """
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)

    def get_seed_optimization_config(self) -> Dict[str, Any]:
        """
        Returns the default optimization configuration.
        
        Returns:
            Default configuration
        """
        return OPTIMIZATION_CONFIG
                
    def get_seed_indicator_config(self) -> Dict[str, Any]:
        """
        Returns the default indicator configuration.

        Returns:
            Default configuration
        """
        return INDICATOR_CLASSES
    
    def print_strategies(self):
        """Displays available strategies."""
        print("Available strategies:")
        for name, info in STRATEGIES_IMPLEMENTED.items():
            print(f"\n  {name}:")
            print(f"    Description: {info['description']}")
            print(f"    Required indicators: {info['required_indicators']}")
            
    def get_strategy_fct(self, strategy_name: str):
        """
        Returns the strategy function corresponding to the given name.

        Args:
            strategy_name (str): Strategy name.

        Returns:
            Strategy function.
        """
        if strategy_name in STRATEGIES_IMPLEMENTED:
            function_name = STRATEGIES_IMPLEMENTED[strategy_name]['function']
            return getattr(self.strategy_core, function_name)
        else:
            raise ValueError(f"Strategy '{strategy_name}' not implemented.")