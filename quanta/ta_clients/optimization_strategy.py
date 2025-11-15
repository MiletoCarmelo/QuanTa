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
                    "low": 15,
                    "high": 50  # Further reduced from 80 to minimize lagging influence
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
        },
        "EngulfingPattern": {
            "class": "EngulfingPattern",
            "params": {
                "separate_columns": {
                    "type": "bool",
                    "value": False
                }
            }
        },
        "ADX": {
            "class": "ADX",
            "params": {
                "period": {
                    "type": "int",
                    "low": 10,
                    "high": 20
                }
            }
        },
        # Optional momentum oscillators - can complement or replace RSI
        "Stochastic": {
            "class": "Stochastic",
            "params": {
                "k_period": {
                    "type": "int",
                    "low": 10,
                    "high": 20
                },
                "k_slow": {
                    "type": "int",
                    "low": 2,
                    "high": 5
                },
                "d_period": {
                    "type": "int",
                    "low": 2,
                    "high": 5
                }
            }
        },
        "WilliamsR": {
            "class": "WilliamsR",
            "params": {
                "period": {
                    "type": "int",
                    "low": 10,
                    "high": 20
                }
            }
        },
        "CCI": {
            "class": "CCI",
            "params": {
                "period": {
                    "type": "int",
                    "low": 10,
                    "high": 20
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
    },
    "signals": {
        "use_engulfing": {
            "type": "bool",
            "value": False,
            "description": "If True, Engulfing pattern is required for signal generation (Bullish for BUY, Bearish for SELL). Auto-detected if EngulfingPattern is in indicators."
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
        },
        "adx_threshold": {
            "type": "float",
            "low": 20.0,
            "high": 35.0,
            "description": "ADX threshold - only trade when ADX < this value (mean reversion works better in range markets). Lower = more restrictive (fewer trades in trends)."
        }
    },
    "constraints": [
        
        # 1. Basic constraints: 
        {"condition": "SMA_long_period >= SMA_short_period + 10"}, # Ensure short period is less than long period with minimal gap (reduced from +20 to +10 for more flexibility)
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
        "target": "sharpe_ratio",  # Metric to optimize (sharpe_ratio, total_return, cagr, max_drawdown)
        # "direction" is auto-determined based on target:
        #   - max_drawdown → minimize
        #   - sharpe_ratio, total_return, cagr → maximize
        # You can override by explicitly setting "direction": "minimize" or "maximize"
        "n_trials": 1000,
        "n_jobs": 1,
        "timeout": 3600,
        "seed": 42,
        # Optional: metrics to track (but not optimize)
        "track_metrics": ["total_return", "cagr", "max_drawdown"]
    }
}


STRATEGIES_IMPLEMENTED = {
    'simple_strategy': {
        'description': 'Simple strategy based on SMA crossovers and RSI levels with optional Engulfing pattern confirmation.',
        'parameters': ['short_window', 'long_window', 'rsi_period', 'stop_loss', 'atr_period', 'take_profit', 'position_size'],
        'signal_parameters': ['use_engulfing', 'macd_hist_buy_threshold', 'macd_hist_sell_threshold', 'adx_threshold'],
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
            adx_threshold: float = 25.0,  # ADX threshold - only trade when ADX < this value (mean reversion works better in range)
            use_engulfing: bool = False,  # If True, Engulfing pattern is required for signal
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
        
        # Build base conditions - RSI and ATR are the main drivers
        # SMA_long is now optional/very soft - focus on mean reversion signals
        # Optional oscillators (Stochastic, WilliamsR, CCI) can complement RSI
        
        # RSI condition (main oscillator)
        rsi_oversold = pl.col(f'RSI{rsi_period}') < 45
        rsi_overbought = pl.col(f'RSI{rsi_period}') > 55
        
        # Optional: Add Stochastic filter if available (complements RSI)
        stoch_col_k = None
        stoch_col_d = None
        for col in df.columns:
            if col.startswith('Stoch') and col.endswith('_K'):
                stoch_col_k = col
                stoch_col_d = col.replace('_K', '_D')
                break
        
        if stoch_col_k and stoch_col_k in df.columns:
            # Stochastic < 20 = oversold, > 80 = overbought
            stoch_oversold = pl.col(stoch_col_k) < 20
            stoch_overbought = pl.col(stoch_col_k) > 80
            # Combine with RSI (both must agree for stronger signal)
            rsi_oversold = rsi_oversold & stoch_oversold
            rsi_overbought = rsi_overbought & stoch_overbought
        
        # Optional: Add Williams %R filter if available
        willr_col = None
        for col in df.columns:
            if col.startswith('WilliamsR'):
                willr_col = col
                break
        
        if willr_col and willr_col in df.columns:
            # Williams %R < -80 = oversold, > -20 = overbought
            willr_oversold = pl.col(willr_col) < -80
            willr_overbought = pl.col(willr_col) > -20
            # Combine with RSI (both must agree for stronger signal)
            rsi_oversold = rsi_oversold & willr_oversold
            rsi_overbought = rsi_overbought & willr_overbought
        
        # Optional: Add CCI filter if available
        cci_col = None
        for col in df.columns:
            if col.startswith('CCI'):
                cci_col = col
                break
        
        if cci_col and cci_col in df.columns:
            # CCI < -100 = oversold, > 100 = overbought
            cci_oversold = pl.col(cci_col) < -100
            cci_overbought = pl.col(cci_col) > 100
            # Combine with RSI (both must agree for stronger signal)
            rsi_oversold = rsi_oversold & cci_oversold
            rsi_overbought = rsi_overbought & cci_overbought
        
        # BUY: Oversold conditions + volatility (mean reversion opportunity)
        buy_base_condition = (
            rsi_oversold &  # RSI (and optional oscillators) oversold
            (pl.col(f'ATR{atr_period}') > atr_mean * 0.6)  # Sufficient volatility
        )
        
        # SELL: Overbought conditions + volatility (mean reversion opportunity)
        sell_base_condition = (
            rsi_overbought &  # RSI (and optional oscillators) overbought
            (pl.col(f'ATR{atr_period}') > atr_mean * 0.6)  # Sufficient volatility
        )
        
        # Optional: Add very soft SMA_long filter if you want (can be disabled by making condition always True)
        # This is now truly optional - comment out or remove if not needed
        use_sma_filter = True  # Set to False to completely disable SMA_long influence
        if use_sma_filter:
            # Calculate SMA_long slope for optional trend context
            df = df.with_columns([
                ((pl.col(f'SMA{long_window}').diff() / pl.col(f'SMA{long_window}').shift(1).fill_null(1.0)) * 100)
                .alias('SMA_long_slope_pct')
            ])
            # Very permissive: only filter out extreme opposite trends
            buy_base_condition = buy_base_condition & (
                (pl.col(f'SMA{short_window}') < pl.col(f'SMA{long_window}') * 1.02) |  # Short can be up to 2% above long
                (pl.col('SMA_long_slope_pct') < 0.2)  # OR long SMA declining/flat (very soft)
            )
            sell_base_condition = sell_base_condition & (
                (pl.col(f'SMA{short_window}') > pl.col(f'SMA{long_window}') * 0.98) |  # Short can be up to 2% below long
                (pl.col('SMA_long_slope_pct') > -0.2)  # OR long SMA rising/flat (very soft)
            )
        
        # Add ADX filter if available (avoids signals during strong trends - mean reversion works better in range)
        # ADX measures trend strength: < 25 = weak trend (good for mean reversion), > 25 = strong trend (avoid)
        adx_period = None
        for col in df.columns:
            if col.startswith('ADX'):
                adx_period = int(col.replace('ADX', ''))
                break
        
        if adx_period and f'ADX{adx_period}' in df.columns:
            # Only trade when trend is weak (ADX < threshold) - mean reversion works better in range markets
            # Threshold is now optimizable via adx_threshold parameter
            buy_condition = buy_base_condition & (pl.col(f'ADX{adx_period}') < adx_threshold)
            sell_condition = sell_base_condition & (pl.col(f'ADX{adx_period}') < adx_threshold)
        else:
            buy_condition = buy_base_condition
            sell_condition = sell_base_condition
        
        # Add MACD filter if available (avoids signals during strong momentum)
        # Thresholds are now optimizable via macd_hist_buy_threshold and macd_hist_sell_threshold
        if 'MACD_hist' in df.columns:
            # Do not buy if bearish momentum is very strong (free fall)
            # MACD_hist must be greater than threshold to allow buying
            buy_condition = buy_condition & (pl.col('MACD_hist') > macd_hist_buy_threshold)
            # Do not sell if bullish momentum is very strong (strong rise)
            # MACD_hist must be less than threshold to allow selling
            sell_condition = sell_condition & (pl.col('MACD_hist') < macd_hist_sell_threshold)
        
        # Add Engulfing Pattern filter if enabled
        # Bullish Engulfing confirms buy signals, Bearish Engulfing confirms sell signals
        # IMPORTANT: Pattern is detected at the END of candle N, so we check pattern from previous candle (shift(1))
        # to generate signal at the BEGINNING of next candle (N+1)
        if use_engulfing:
            # Check which column format is used
            if 'Engulfing' in df.columns:
                # Single column format: 1 = bullish, -1 = bearish, 0 = none
                # Check pattern from previous candle (shift(1)) to open at beginning of current candle
                buy_condition = buy_condition & (pl.col('Engulfing').shift(1) == 1)
                sell_condition = sell_condition & (pl.col('Engulfing').shift(1) == -1)
            elif 'Engulfing_Bullish' in df.columns and 'Engulfing_Bearish' in df.columns:
                # Separate columns format
                # Check pattern from previous candle (shift(1)) to open at beginning of current candle
                buy_condition = buy_condition & (pl.col('Engulfing_Bullish').shift(1) == 1)
                sell_condition = sell_condition & (pl.col('Engulfing_Bearish').shift(1) == 1)
        
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
        """Displays available strategies with required and optional indicators."""
        print("Available strategies:")
        for name, info in STRATEGIES_IMPLEMENTED.items():
            print(f"\n  {name}:")
            print(f"    Description: {info['description']}")
            print(f"    Required indicators: {info['required_indicators']}")
            
            # Get optional indicators from OPTIMIZATION_CONFIG
            required_set = set(info['required_indicators'])
            all_indicators = set(OPTIMIZATION_CONFIG.get('indicators', {}).keys())
            optional_indicators = sorted(all_indicators - required_set)
            
            if optional_indicators:
                print(f"    Optional indicators: {optional_indicators}")
            
            # Display signal parameters
            signal_params = info.get('signal_parameters', [])
            if signal_params:
                print(f"    Signal parameters: {signal_params}")
            
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