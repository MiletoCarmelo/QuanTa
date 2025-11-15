import optuna
from optuna.samplers import TPESampler
from quanta.utils.ta import TAClient, INDICATOR_CLASSES
from quanta.ta_clients.optimization_strategy import STRATEGIES_IMPLEMENTED
from quanta.clients.yfinance import YahooFinanceClient
from quanta.ta_clients.optimization_strategy import StrategyClient
from quanta.utils.ta import TAClient
import polars as pl
from typing import Dict, Any, List, Callable, Tuple, Union
from pathlib import Path
from optuna.importance import get_param_importances
import re
import polars as pl
import numpy as np

class OptimizationClient:
    """Client for optimizing trading strategies with Optuna."""

    def __init__(self, optimization_config: Dict[str, Any]):
        """
        Initialize the optimization client.
        
        Args:
            optimization_config: Optimization configuration (dict or path to JSON)
        """
        
        # Extract indicator classes and optimization configuration
        self.optimization_config = optimization_config
        self.study = None
        
        # Validate that the strategy exists
        strategy_name = optimization_config.get("strategy_name")
        if not strategy_name:
            raise ValueError("'strategy_name' missing in config")
        
        if strategy_name not in STRATEGIES_IMPLEMENTED:
            raise ValueError(f"Strategy '{strategy_name}' unknown")
        
        # Validate that required indicators are properly configured
        required = STRATEGIES_IMPLEMENTED[strategy_name]['required_indicators']
        configured = set(optimization_config['indicators'].keys())
        
        # Automatically filter the config to keep required indicators + optional ones like EngulfingPattern
        # Optional indicators are those that can be used but are not required
        optional_indicators = {'EngulfingPattern'}  # Add other optional indicators here if needed
        
        filtered_indicators = {}
        # Add required indicators
        for ind_name in required:
            if ind_name in optimization_config['indicators']:
                filtered_indicators[ind_name] = optimization_config['indicators'][ind_name]
        
        # Add optional indicators if present
        for ind_name in optional_indicators:
            if ind_name in optimization_config['indicators']:
                filtered_indicators[ind_name] = optimization_config['indicators'][ind_name]
        
        # Check that we have all required indicators
        missing = set(required) - set(filtered_indicators.keys())
        if missing:
            raise ValueError(f"Missing indicators for {strategy_name}: {missing}")
        
        # Display ignored indicators (those that are neither required nor optional)
        ignored = configured - set(required) - optional_indicators
        if ignored:
            print(f"ℹ️  Ignored indicators (not used by '{strategy_name}'): {ignored}")
        
        self.optimization_config['indicators'] = filtered_indicators

        # NEW: Also filter constraints to keep only valid ones
        filtered_constraints = []
        for constraint in self.optimization_config.get('constraints', []):
            condition = constraint['condition']
            # Check if all constraint parameters are in required indicators
            is_valid = True
            for ind_name in configured:  # All indicators from original config
                if ind_name not in required and ind_name in condition:
                    is_valid = False
                    break
            
            if is_valid:
                filtered_constraints.append(constraint)
            else:
                print(f"ℹ️  Ignored constraint (indicator not used): {condition}")

        self.optimization_config['constraints'] = filtered_constraints

        print(f"✓ Strategy '{strategy_name}' initialized with indicators: {list(filtered_indicators.keys())}")
        

    def _parse_constraints(self) -> Dict[str, List[Tuple[str, Union[str, float]]]]:
        """
        Parse constraints to identify dependencies between parameters.
        Supports:
        - param1 < param2 (comparison between parameters)
        - param1 < 50 (comparison with constant)
        - 30 < param1 < 70 (to implement if necessary)
        
        Returns:
            Dict mapping each parameter to its constraints of type (operator, other_param_or_value)
            Example: {
                'SMA_long_period': [('<', 'SMA_short_period')],
                'RSI_period': [('>', 5), ('<', 25)]
            }
        """
        constraints_map = {}
        
        for constraint in self.optimization_config.get('constraints', []):
            condition = constraint['condition']
            
            # Parse constraint with regex
            # Supports: A < B, A > 50, 30 < A, etc.
            pattern = r'([\w.]+)\s*([<>=!]+)\s*([\w.]+)'
            match = re.match(pattern, condition)
            
            if match:
                left = match.group(1)
                operator = match.group(2)
                right = match.group(3)
                
                # Determine if left and right are parameters or constants
                def is_number(s):
                    try:
                        float(s)
                        return True
                    except ValueError:
                        return False
                
                left_is_param = not is_number(left)
                right_is_param = not is_number(right)
                
                # Reverse operator
                inverse_ops = {
                    '<': '>',
                    '>': '<',
                    '<=': '>=',
                    '>=': '<=',
                    '==': '=='
                }
                
                # Case 1: param < param (or param > param, etc.)
                if left_is_param and right_is_param:
                    if left not in constraints_map:
                        constraints_map[left] = []
                    constraints_map[left].append((operator, right))
                    
                    if right not in constraints_map:
                        constraints_map[right] = []
                    constraints_map[right].append((inverse_ops.get(operator, operator), left))
                
                # Case 2: param < 50 (parameter compared to constant)
                elif left_is_param and not right_is_param:
                    if left not in constraints_map:
                        constraints_map[left] = []
                    constraints_map[left].append((operator, float(right)))
                
                # Case 3: 30 < param (constant compared to parameter)
                elif not left_is_param and right_is_param:
                    if right not in constraints_map:
                        constraints_map[right] = []
                    # Reverse: 30 < param → param > 30
                    constraints_map[right].append((inverse_ops.get(operator, operator), float(left)))
        
        return constraints_map

    def _apply_constraint_to_range(
        self,
        param_name: str,
        low: Union[int, float],
        high: Union[int, float],
        suggested_values: Dict[str, Union[int, float]]
    ) -> Tuple[Union[int, float], Union[int, float]]:
        """
        Adjust parameter bounds based on constraints and already suggested values.
        
        Args:
            param_name: Parameter name
            low: Original lower bound
            high: Original upper bound
            suggested_values: Dict of already suggested values
        
        Returns:
            New bounds (low, high) respecting constraints
        """
        if not hasattr(self, '_constraints_map'):
            self._constraints_map = self._parse_constraints()
        
        constraints = self._constraints_map.get(param_name, [])
        
        for operator, other in constraints:
            # Determine if 'other' is a parameter or a value
            if isinstance(other, (int, float)):
                # It's a constant
                other_value = other
            elif other in suggested_values:
                # It's an already suggested parameter
                other_value = suggested_values[other]
            else:
                # Parameter not yet suggested, we can't apply this constraint
                continue
            
            # Apply constraint
            if operator == '<':
                # param < other → param must be < other
                if isinstance(low, int) and isinstance(other_value, (int, float)):
                    high = min(high, int(other_value) - 1)
                else:
                    high = min(high, other_value - 0.001)  # For floats
            elif operator == '>':
                # param > other → param must be > other
                if isinstance(low, int) and isinstance(other_value, (int, float)):
                    low = max(low, int(other_value) + 1)
                else:
                    low = max(low, other_value + 0.001)
            elif operator == '<=':
                if isinstance(low, int):
                    high = min(high, int(other_value))
                else:
                    high = min(high, other_value)
            elif operator == '>=':
                if isinstance(low, int):
                    low = max(low, int(other_value))
                else:
                    low = max(low, other_value)
            elif operator == '==':
                low = high = other_value
        
        # Ensure low <= high
        if low > high:
            # Impossible constraint to satisfy
            return (low, low - 1)
        
        return (low, high)

    def suggest_params_from_config(self, trial: optuna.Trial, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate parameter suggestions while dynamically respecting constraints.
        CORRECTED version: Suggestion THEN constraints.
        """
        params = {
            'indicators': {},
            'strategy': {},
            'signals': {}
        }
        
        suggested_values = {}
        
        # 1. Suggest ALL indicator parameters
        for indicator_name, indicator_config in config.get("indicators", {}).items():
            params['indicators'][indicator_name] = {}
            
            for param_name, param_config in indicator_config.get("params", {}).items():
                # Check if parameter has a fixed value (not optimizable)
                if "value" in param_config:
                    # Use fixed value, don't optimize
                    params['indicators'][indicator_name][param_name] = param_config["value"]
                    continue
                
                param_type = param_config["type"]
                full_param_name = f"{indicator_name}_{param_name}"
                
                if param_type == "int":
                    low = param_config["low"]
                    high = param_config["high"]
                    
                    # Adjust according to constraints ONLY between indicators
                    low, high = self._apply_constraint_to_range(
                        full_param_name, low, high, suggested_values
                    )
                    
                    if low > high:
                        value = param_config["low"]
                    else:
                        value = trial.suggest_int(
                            full_param_name,
                            int(low),
                            int(high),
                            step=param_config.get("step", 1)
                        )
                    
                    suggested_values[full_param_name] = value
                    
                elif param_type == "float":
                    low = param_config["low"]
                    high = param_config["high"]
                    
                    low, high = self._apply_constraint_to_range(
                        full_param_name, low, high, suggested_values
                    )
                    
                    if low > high:
                        value = param_config["low"]
                    else:
                        value = trial.suggest_float(
                            full_param_name,
                            low,
                            high,
                            log=param_config.get("log", False),
                            step=param_config.get("step", None)
                        )
                    
                    suggested_values[full_param_name] = value
                    
                elif param_type == "categorical":
                    value = trial.suggest_categorical(
                        full_param_name,
                        param_config["choices"]
                    )
                    suggested_values[full_param_name] = value
                
                params['indicators'][indicator_name][param_name] = value
        
        # 2. Suggest ALL strategy parameters (WITHOUT constraints for now)
        for param_name, param_config in config.get("strategy", {}).items():
            param_type = param_config["type"]
            
            if param_type == "int":
                value = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1)
                )
                
            elif param_type == "float":
                value = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                    step=param_config.get("step", None)
                )
                
            elif param_type == "categorical":
                value = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
            
            params['strategy'][param_name] = value
            suggested_values[param_name] = value
        
        # 3. Process signal parameters (conditions for opening/closing positions)
        for param_name, param_config in config.get("signals", {}).items():
            param_type = param_config.get("type")
            
            # Check if parameter has a fixed value
            if "value" in param_config:
                params['signals'][param_name] = param_config["value"]
                continue
            
            # Otherwise, suggest value if it has optimization range
            if param_type == "int" and "low" in param_config and "high" in param_config:
                value = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1)
                )
                params['signals'][param_name] = value
                suggested_values[param_name] = value
            elif param_type == "float" and "low" in param_config and "high" in param_config:
                value = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                    step=param_config.get("step", None)
                )
                params['signals'][param_name] = value
                suggested_values[param_name] = value
            elif param_type == "categorical" and "choices" in param_config:
                value = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
                params['signals'][param_name] = value
                suggested_values[param_name] = value
        
        return params
    
    def create_indicators_from_params(self, config: Dict[str, Any], params: Dict[str, Any]) -> List:
        """
        Automatically create indicators from parameters.
        
        Args:
            config: Complete JSON configuration
            params: Suggested parameters (with 'indicators' and 'strategy' structure)
        
        Returns:
            List of instantiated indicator objects
        """
        indicators = []
        
        # Create indicators from params (includes both optimizable and fixed parameters)
        for indicator_name, indicator_params in params['indicators'].items():
            indicator_config = config['indicators'][indicator_name]
            indicator_class_name = indicator_config['class']

            if indicator_class_name not in INDICATOR_CLASSES:
                raise ValueError(f"Indicator '{indicator_class_name}' not found in INDICATOR_CLASSES")
            
            indicator_class = INDICATOR_CLASSES[indicator_class_name]

            # Instantiate indicator with parameters
            try:
                # Try first with kwargs
                indicator = indicator_class(**indicator_params)
            except TypeError:
                # If it fails, try with positional args
                indicator = indicator_class(*indicator_params.values())
            
            indicators.append(indicator)
        
        return indicators
    
    def check_constraints(self, params: Dict[str, Any], constraints: list) -> bool:
        """
        Check if parameters respect constraints.
        
        Args:
            params: Suggested parameters
            constraints: List of constraints
        
        Returns:
            True if all constraints are respected
        """
        # Flatten params structure for evaluation
        flat_params = {}
        
        # Add indicator parameters
        for indicator_name, indicator_params in params['indicators'].items():
            for param_name, param_value in indicator_params.items():
                flat_params[f"{indicator_name}_{param_name}"] = param_value
        
        # Add strategy parameters
        flat_params.update(params['strategy'])
        
        # Add signal parameters
        flat_params.update(params.get('signals', {}))
        
        # Check constraints
        for constraint in constraints:
            condition = constraint["condition"]
            try:
                if not eval(condition, {}, flat_params):
                    return False
            except Exception as e:
                print(f"Error when evaluating constraint '{condition}': {e}")
                return False
        
        return True

    def _map_params_for_strategy(self, params: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
        """
        Map optimization parameters to parameters expected by strategy function.
        
        Args:
            params: Parameters suggested by Optuna (structure with 'indicators' and 'strategy')
            strategy_name: Strategy name
        
        Returns:
            Dictionary with mapped parameters for backtest function
        """
        
        if strategy_name not in STRATEGIES_IMPLEMENTED:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        strategy_info = STRATEGIES_IMPLEMENTED[strategy_name]
        expected_params = strategy_info['parameters']
        signal_params = strategy_info.get('signal_parameters', [])
        required_indicators = strategy_info['required_indicators']
        
        backtest_params = {}
        
        # Map signal parameters first (conditions for opening/closing positions)
        for signal_param in signal_params:
            if signal_param in params.get('signals', {}):
                backtest_params[signal_param] = params['signals'][signal_param]
        
        # Map indicator parameters
        for expected_param in expected_params:
            # Check if it's a strategy parameter (stop_loss, take_profit, etc.)
            if expected_param in params['strategy']:
                backtest_params[expected_param] = params['strategy'][expected_param]
                continue
            
            # Otherwise, search in required indicators
            param_found = False
            for indicator_name in required_indicators:
                if indicator_name in params['indicators']:
                    indicator_params = params['indicators'][indicator_name]
                    
                    # Map according to naming conventions
                    # For example: "short_window" → search "period" in "SMA_short"
                    if 'short' in expected_param.lower() and 'short' in indicator_name.lower():
                        if 'period' in indicator_params:
                            backtest_params[expected_param] = indicator_params['period']
                            param_found = True
                            break
                    elif 'long' in expected_param.lower() and 'long' in indicator_name.lower():
                        if 'period' in indicator_params:
                            backtest_params[expected_param] = indicator_params['period']
                            param_found = True
                            break
                    elif 'rsi' in expected_param.lower() and 'RSI' in indicator_name:
                        if 'period' in indicator_params:
                            backtest_params[expected_param] = indicator_params['period']
                            param_found = True
                            break
                    elif 'atr' in expected_param.lower() and 'ATR' in indicator_name:
                        if 'period' in indicator_params:
                            backtest_params[expected_param] = indicator_params['period']
                            param_found = True
                            break
            
            if not param_found:
                raise ValueError(f"Unable to map parameter '{expected_param}' for strategy '{strategy_name}'")
        
        # Automatically detect if EngulfingPattern is in indicators and override use_engulfing
        # Check in the optimization_config (which contains all configured indicators)
        if 'EngulfingPattern' in self.optimization_config.get('indicators', {}):
            # Override use_engulfing from signals if EngulfingPattern is present
            backtest_params['use_engulfing'] = True
        elif 'use_engulfing' not in backtest_params:
            # If not in signals and EngulfingPattern not present, default to False
            backtest_params['use_engulfing'] = False
        
        return backtest_params

    def objective_function_create(self, 
                  trial: optuna.Trial,
                  backtest_func: Callable,
                  df_init: pl.DataFrame,
                  strategy_name: str) -> float:
        """
        Objective function for optimization.
        Args:
            trial: Optuna trial object
            config: Complete JSON configuration
            backtest_func: Backtest function
            df_init: Initial DataFrame
        Returns:
            Sharpe Ratio (float)
        """
        
        # Suggest parameters
        params = self.suggest_params_from_config(trial, self.optimization_config)
        
        # Check constraints
        constraints = self.optimization_config.get("constraints", [])
        if not self.check_constraints(params, constraints):
            return float('-inf')  # Penalize trials that don't respect constraints
        # Create indicators
        indicators = self.create_indicators_from_params(self.optimization_config, params)
        
        # Apply indicators
        df = df_init.clone()
        ta_client = TAClient()
        df = ta_client.calculate_indicators(df, indicators)

        # Get strategy name from config
        if strategy_name is None or not isinstance(strategy_name, str):
            raise ValueError("The 'strategy_name' field must be defined in OPTIMIZATION_CONFIG and be a string")
        
        # Dynamically map parameters
        backtest_params = self._map_params_for_strategy(params, strategy_name)
        
        # Execute backtest
        try:
            # Use contextlib.redirect_stdout to suppress prints if verbose=False
            from contextlib import redirect_stdout
            from io import StringIO
            
            if not getattr(self, '_verbose', True):
                # Redirect stdout to a StringIO to suppress prints
                with redirect_stdout(StringIO()):
                    metrics = backtest_func(df, **backtest_params)
            else:
                metrics = backtest_func(df, **backtest_params)
            
            # Get target metric from config (default to sharpe_ratio for backward compatibility)
            opt_config = self.optimization_config.get("optimization", {})
            target_metric = opt_config.get("target", "sharpe_ratio")
            direction = opt_config.get("direction", "maximize")
            track_metrics = opt_config.get("track_metrics", ["total_return", "cagr", "max_drawdown"])
            
            # Determine default value based on direction
            # For maximize: use -inf (worst possible)
            # For minimize: use +inf (worst possible)
            default_value = float('-inf') if direction == "maximize" else float('inf')
            
            # Get the target metric value
            target_value = metrics.get(target_metric, default_value)
            
            # Record all tracked metrics in trial's user_attrs
            for metric_name in track_metrics:
                if metric_name in metrics:
                    trial.set_user_attr(metric_name, metrics.get(metric_name, 0.0))
            
            # Also record the target metric explicitly
            trial.set_user_attr(f"target_{target_metric}", target_value)
            
            if "trades_df" in metrics:
                trades_json = metrics["trades_df"].write_json()
                trial.set_user_attr("trades_json", trades_json)
                trial.set_user_attr("num_trades", len(metrics["trades_df"]))
                
            # Return appropriate default if value is None
            if target_value is None:
                return default_value
            return target_value
        
        except Exception as e:
            if getattr(self, '_verbose', True):
                print(f"Error during backtest: {e}")
            # Return worst possible value based on direction
            opt_config = self.optimization_config.get("optimization", {})
            direction = opt_config.get("direction", "maximize")
            return float('-inf') if direction == "maximize" else float('inf')
    
    def optimize(self, 
                 backtest_func: Callable,
                 df: pl.DataFrame,
                 verbose: bool = True) -> optuna.Study:
        """
        Launch optimization with provided configuration.
        
        Args:
            config: Optimization configuration (dict or path to JSON)
            backtest_func: Backtest function
            df: Initial DataFrame
            verbose: Display detailed information (config, results, progress bar AND strategy prints)
        
        Returns:
            Optuna Study with results
        """
        opt_config = self.optimization_config["optimization"]
        indic_config = self.optimization_config["indicators"]
        strat_config = self.optimization_config["strategy"]
        signals_config = self.optimization_config.get("signals", {})
        
        # Auto-determine direction based on target metric if not explicitly set
        target_metric = opt_config.get('target', 'sharpe_ratio')
        metrics_to_minimize = ['max_drawdown']  # Metrics that should be minimized
        metrics_to_maximize = ['sharpe_ratio', 'total_return', 'cagr']  # Metrics that should be maximized
        
        # Check if direction was explicitly set in original config
        original_opt_config = self.optimization_config.get("optimization", {})
        direction_was_set = 'direction' in original_opt_config
        
        # If direction is not explicitly set, auto-determine it
        if not direction_was_set:
            if target_metric in metrics_to_minimize:
                opt_config['direction'] = 'minimize'
            elif target_metric in metrics_to_maximize:
                opt_config['direction'] = 'maximize'
            else:
                # Default to maximize if unknown metric
                opt_config['direction'] = 'maximize'
                if verbose:
                    print(f"⚠️  Warning: Unknown target metric '{target_metric}', defaulting to 'maximize'")
        
        # Display configuration
        if verbose:
            print("="*70)
            print("OPTIMIZATION CONFIGURATION")
            print("="*70)
            print(f"\nIndicators to optimize:")
            for ind_name, ind_config in indic_config.items():
                print(f"  - {ind_name} ({ind_config['class']}): {list(ind_config['params'].keys())}")
            
            print(f"\nStrategy parameters:")
            for param_name in strat_config.keys():
                print(f"  - {param_name}")
            
            if signals_config:
                print(f"\nSignal parameters:")
                for param_name in signals_config.keys():
                    print(f"  - {param_name}")
            
            print(f"\nOptimization target: {target_metric}")
            direction_msg = f"Direction: {opt_config['direction']}"
            if not direction_was_set:
                direction_msg += " (auto-determined)"
            print(direction_msg)
            print(f"Number of trials: {opt_config['n_trials']}")
            print("="*70 + "\n")
        
        # Store verbose for use in objective_function_create
        self._verbose = verbose
        
        # Create objective function with config
        def objective(trial):
            strategy_name_val = self.optimization_config.get("strategy_name")
            if strategy_name_val is None or not isinstance(strategy_name_val, str):
                raise ValueError("The 'strategy_name' field must be defined in OPTIMIZATION_CONFIG and be a string")
            return self.objective_function_create(
                trial=trial,
                backtest_func=backtest_func,
                df_init=df,
                strategy_name=strategy_name_val
            )
        
        # Control Optuna logs
        import logging
        optuna_logger = logging.getLogger("optuna")
        
        # Create study
        self.study = optuna.create_study(
            direction=opt_config["direction"],
            study_name=opt_config.get("study_name", "trading_strategy_optimization"),
            sampler=TPESampler(seed=opt_config.get("seed", None)),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        )
        
        # Disable Optuna logs if verbose=False
        if not verbose:
            optuna_logger.setLevel(logging.WARNING)  # Only warnings and errors
        
        # Optimize
        # Always show progress bar (tqdm), but strategy prints and logs are controlled by verbose
        self.study.optimize(
            objective, 
            n_trials=opt_config["n_trials"],
            n_jobs=opt_config.get("n_jobs", 1),
            show_progress_bar=True,  # Always show tqdm
            timeout=opt_config.get("timeout", None)
        )
        
        # Restore log level after optimization
        if not verbose:
            optuna_logger.setLevel(logging.INFO)
        
        # Display results
        if verbose:
            self._print_results()
        
        return self.study
    
    def _print_results(self):
        """Display optimization results."""
        opt_config = self.optimization_config.get("optimization", {})
        target_metric = opt_config.get("target", "sharpe_ratio")
        
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS")
        print("="*70)
        print(f"\nBest {target_metric}: {self.study.best_value:.4f}")
        
        print(f"\nBest parameters:")
        print("\n  Indicators:")
        for key, value in self.study.best_params.items():
            if any(ind_name in key for ind_name in self.optimization_config['indicators'].keys()):
                print(f"    {key}: {value}")
        
        print("\n  Strategy:")
        for key, value in self.study.best_params.items():
            if key in self.optimization_config['strategy'].keys():
                print(f"    {key}: {value}")
        
        signals_config = self.optimization_config.get("signals", {})
        if signals_config:
            print("\n  Signals:")
            for key, value in self.study.best_params.items():
                if key in signals_config.keys():
                    print(f"    {key}: {value}")
                
        print(f"\nPerformance metrics:")
        for key, value in self.study.best_trial.user_attrs.items():
            print(f"    {key}: {value:.4f}")
            
    def save_results(self, output_dir: str = "."):
        """
        Save optimization results.
        
        Args:
            output_dir: Output directory
        """
        if self.study is None:
            print("No optimization has been performed")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save all trials
        trials_df = self.study.trials_dataframe()
        trials_df.to_csv(output_path / 'optimization_results.csv', index=False)
        
        # Save best config
        best_config = self.get_best_config()
        self.save_config(best_config, output_path / 'best_config.json')
        
        print(f"\nResults saved in '{output_dir}/'")
        print(f"  - optimization_results.csv")
        print(f"  - best_config.json")
    
    def get_best_config(self) -> Dict[str, Any]:
        """
        Get the best configuration found.
        
        Returns:
            Dictionary with best parameters
        """
        if self.study is None:
            return {}
        
        best_config = {
            'indicators': {},
            'strategy': {},
            'signals': {}
        }
        
        # Get only indicators required by strategy (not all configured ones)
        strategy_name = self.optimization_config.get("strategy_name")
        if strategy_name and strategy_name in STRATEGIES_IMPLEMENTED:
            required_indicators = STRATEGIES_IMPLEMENTED[strategy_name]['required_indicators']
        else:
            required_indicators = list(self.optimization_config['indicators'].keys())
        
        # Get signal parameters from config
        signals_config = self.optimization_config.get("signals", {})
        strategy_config = self.optimization_config.get("strategy", {})
        
        # Iterate through all parameters of best trial
        for key, value in self.study.best_params.items():
            # Check if it's an indicator parameter
            matched = False
            for ind_name in required_indicators:
                if key.startswith(ind_name + "_") or key == ind_name:
                    if ind_name not in best_config['indicators']:
                        best_config['indicators'][ind_name] = {}
                    param_name = key.replace(f"{ind_name}_", "")
                    best_config['indicators'][ind_name][param_name] = value
                    matched = True
                    break
            
            if not matched:
                # Check if it's a signal parameter
                if key in signals_config:
                    best_config['signals'][key] = value
                elif key in strategy_config:
                    # It's a strategy parameter
                    best_config['strategy'][key] = value
                else:
                    # Fallback: assume it's a strategy parameter
                    best_config['strategy'][key] = value
        
        return best_config
    
    def get_best_indicators(self) -> List:
        """
        Get indicators with best parameters.
        
        Returns:
            List of instantiated indicator objects
        """
        if self.study is None:
            return []
        
        best_config = self.get_best_config()
        params = {
            'indicators': best_config['indicators'],
            'strategy': best_config['strategy']
        }

        return self.create_indicators_from_params(self.optimization_config, params)

    def analyze_parameter_importance(self):
        """
        Analyze importance of each parameter in optimization.
        
        Returns:
            Dict with importance of each parameter
        """
        if self.study is None:
            print("No optimization has been performed")
            return {}
        
        from optuna.importance import get_param_importances
        import warnings
        
        # Check there are enough valid trials
        valid_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(valid_trials) < 2:
            print("⚠️  Not enough valid trials to calculate parameter importance")
            return {}
        
        # Count trials with inf or nan that will be omitted by Optuna
        import math
        omitted_count = 0
        for t in self.study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                try:
                    value = t.value
                    if value is None or (isinstance(value, (int, float)) and 
                                        (math.isinf(value) or math.isnan(value))):
                        omitted_count += 1
                except (AttributeError, TypeError):
                    omitted_count += 1
        
        importances = {}
        
        # Remove ALL warnings and Optuna logs about omitted trials (aggressive approach)
        import logging
        
        # Save Optuna logger level and set it to ERROR to suppress warnings
        optuna_logger = logging.getLogger("optuna")
        old_level = optuna_logger.level
        optuna_logger.setLevel(logging.ERROR)  # Only errors
        
        # Capture and suppress ALL Python warnings
        with warnings.catch_warnings():
            # Filter all warnings about omitted trials
            warnings.filterwarnings("ignore", message=".*omitted.*")
            warnings.filterwarnings("ignore", message=".*omitted in visualization.*")
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.simplefilter("ignore")
            
            try:
                # Try FANOVA first (best method)
                try:
                    importances = get_param_importances(
                        self.study, 
                        evaluator=optuna.importance.FanovaImportanceEvaluator()
                    )
                    print("\n" + "="*70)
                    print("PARAMETER IMPORTANCE (FANOVA)")
                    print("="*70)
                    
                    # Display omitted trials count only once
                    if omitted_count > 0:
                        print(f"ℹ️  Number of omitted trials: {omitted_count}")
                    
                    for param, importance in sorted(importances.items(), 
                                                key=lambda x: x[1], reverse=True):
                        status = "⚠️  USELESS" if importance < 0.01 else "✓ Useful"
                        print(f"{param:30s}: {importance:.4f} {status}")
                    return importances
                except Exception as e:
                    print(f"ℹ️  FANOVA not available ({type(e).__name__}), using default method")
                
                # Try tree-based method (more robust)
                try:
                    importances = get_param_importances(
                        self.study,
                        evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator()
                    )
                    print("\n" + "="*70)
                    print("PARAMETER IMPORTANCE (Mean Decrease Impurity)")
                    print("="*70)
                    
                    # Display omitted trials count only once
                    if omitted_count > 0:
                        print(f"ℹ️  Number of omitted trials: {omitted_count}")
                    
                    for param, importance in sorted(importances.items(), 
                                                key=lambda x: x[1], reverse=True):
                        status = "⚠️  USELESS" if importance < 0.01 else "✓ Useful"
                        print(f"{param:30s}: {importance:.4f} {status}")
                    return importances
                except Exception as e:
                    print(f"ℹ️  Mean Decrease Impurity not available ({type(e).__name__})")
                
                # Fallback: default method (always available)
                try:
                    importances = get_param_importances(self.study)
                    print("\n" + "="*70)
                    print("PARAMETER IMPORTANCE")
                    print("="*70)
                    
                    # Display omitted trials count only once
                    if omitted_count > 0:
                        print(f"ℹ️  Number of omitted trials: {omitted_count}")
                    
                    for param, importance in sorted(importances.items(), 
                                                key=lambda x: x[1], reverse=True):
                        status = "⚠️  USELESS" if importance < 0.01 else "✓ Useful"
                        print(f"{param:30s}: {importance:.4f} {status}")
                    return importances
                except Exception as e:
                    print(f"❌ Unable to calculate parameter importance: {e}")
                    return {}
            finally:
                # Restore Optuna logger level
                optuna_logger.setLevel(old_level)
    
    def get_useless_parameters(self, threshold: float = 0.01):
        """
        Identify parameters with little impact.
        
        Args:
            threshold: Threshold below which a parameter is considered useless
        
        Returns:
            List of useless parameters
        """
        from optuna.importance import get_param_importances
        
        if self.study is None:
            return []
        
        try:
            importances = get_param_importances(
                self.study,
                evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator()
            )
        except Exception as e:
            print(f"❌ Unable to calculate importance: {e}")
            return []
        
        # Separate parameters by type
        indicator_params = {}
        strategy_params = {}
        
        for param, imp in importances.items():
            # Check if it's a strategy parameter
            if param in self.optimization_config.get('strategy', {}).keys():
                strategy_params[param] = imp
            else:
                # Otherwise it's an indicator parameter
                indicator_params[param] = imp
        
        # Identify useless ones
        useless_indicators = {p: imp for p, imp in indicator_params.items() if imp < threshold}
        useless_strategy = {p: imp for p, imp in strategy_params.items() if imp < threshold}
        
        print("\n" + "="*70)
        print("LOW IMPACT PARAMETERS ANALYSIS")
        print("="*70)
        
        if useless_indicators:
            print(f"\n⚠️  Technical indicators with low impact (< {threshold}):")
            for param, imp in sorted(useless_indicators.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {param:30s} (importance: {imp:.4f})")
        else:
            print(f"\n✓ All technical indicators have significant impact")
        
        if useless_strategy:
            print(f"\n⚠️  Strategy parameters with low impact (< {threshold}):")
            for param, imp in sorted(useless_strategy.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {param:30s} (importance: {imp:.4f})")
                # Add explanation
                if param in ['stop_loss', 'take_profit']:
                    print(f"      → This may indicate that your strategy doesn't use them effectively")
                elif param == 'position_size':
                    print(f"      → Sharpe Ratio is independent of position size")
        else:
            print(f"\n✓ All strategy parameters have significant impact")
        
        return list(useless_indicators.keys()) + list(useless_strategy.keys())
    
    def plot_importance(self):
        """Visualize parameter importance."""
        if self.study is None:
            print("No optimization has been performed")
            return
        
        import optuna.visualization as vis
        import warnings
        import math
        import logging
        
        # Count omitted trials
        omitted_count = sum(1 for t in self.study.trials 
                          if t.state == optuna.trial.TrialState.COMPLETE
                          and (not hasattr(t, 'value') or t.value is None 
                               or (isinstance(t.value, (int, float)) and 
                                  (math.isinf(t.value) or math.isnan(t.value)))))
        
        # Save and modify Optuna logger level to suppress warnings
        optuna_logger = logging.getLogger("optuna")
        old_level = optuna_logger.level
        optuna_logger.setLevel(logging.ERROR)  # Only errors
        
        try:
            # Suppress Optuna warnings about omitted trials
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*omitted.*")
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.simplefilter("ignore")
                
                # Create importance graph
                fig = vis.plot_param_importances(self.study)
                fig.show()
        finally:
            # Restore Optuna logger level
            optuna_logger.setLevel(old_level)
        
        # Display count only once if necessary
        if omitted_count > 0:
            print(f"\nℹ️  Number of omitted trials: {omitted_count}")

    def get_trades_history(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Get detailed trade history with best parameters.
        
        Args:
            df: DataFrame with price data
        
        Returns:
            DataFrame with trade history
        """
        if self.study is None:
            print("⚠️ No optimization has been performed")
            return pl.DataFrame()
        
        # Get best parameters
        best_config = self.get_best_config()
        
        # Check that config is not empty
        if not best_config.get('indicators') and not best_config.get('strategy'):
            print("⚠️ Error: Unable to retrieve best configuration")
            return pl.DataFrame()
        
        params = {
            'indicators': best_config['indicators'],
            'strategy': best_config['strategy']
        }
        
        try:
            # Create indicators
            indicators = self.create_indicators_from_params(self.optimization_config, params)
            
            if not indicators:
                print("⚠️ Error: No indicators created")
                return pl.DataFrame()
            
            # Apply indicators
            df_with_indicators = df.clone()
            ta_client = TAClient()
            df_with_indicators = ta_client.calculate_indicators(df_with_indicators, indicators)
            
            # Check that DataFrame is not empty after calculating indicators
            if len(df_with_indicators) == 0:
                print("⚠️ Error: DataFrame empty after calculating indicators")
                return pl.DataFrame()
            
            # Get strategy function
            strategy_name = self.optimization_config.get("strategy_name")
            opt_strategy = StrategyClient()
            backtest_func = opt_strategy.get_strategy_fct(strategy_name)
            
            # Map parameters
            backtest_params = self._map_params_for_strategy(params, strategy_name)
            
            # Execute with return_trades=True
            results = backtest_func(df_with_indicators, **backtest_params, return_trades=True)
            
            # Check that results contains trades_df
            if not isinstance(results, dict):
                print(f"⚠️ Error: Strategy function did not return a dictionary, type: {type(results)}")
                return pl.DataFrame()
            
            trades_df = results.get("trades_df")
            if trades_df is None:
                print("⚠️ Error: 'trades_df' key is missing from result")
                print(f"   Available keys: {list(results.keys())}")
                return pl.DataFrame()
            
            if isinstance(trades_df, pl.DataFrame) and len(trades_df) == 0:
                print("⚠️ No trades generated with best parameters")
                return trades_df
            
            return trades_df
            
        except Exception as e:
            print(f"⚠️ Error retrieving trades: {e}")
            import traceback
            traceback.print_exc()
            return pl.DataFrame()

    def print_trades_summary(self, trades_df: pl.DataFrame):
        """
        Display trade summary.
        
        Args:
            trades_df: DataFrame of trades
        """
        if len(trades_df) == 0:
            print("No trades to display")
            return
        
        print("\n" + "="*70)
        print("TRADES SUMMARY")
        print("="*70)
        
        # General stats
        total_trades = len(trades_df)
        buy_trades = (trades_df['action'] == 'BUY').sum()
        sell_trades = (trades_df['action'] == 'SELL').sum()
        
        print(f"\n📊 Total number of trades: {total_trades}")
        print(f"  - Buys: {buy_trades}")
        print(f"  - Sells: {sell_trades}")
        
        # Performance stats
        winning_trades = (trades_df['pnl'] > 0).sum()
        losing_trades = (trades_df['pnl'] < 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\n💰 Performance:")
        print(f"  - Winning trades: {winning_trades} ({win_rate:.2f}%)")
        print(f"  - Losing trades: {losing_trades} ({100-win_rate:.2f}%)")
        
        avg_win = trades_df.filter(pl.col('pnl') > 0)['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df.filter(pl.col('pnl') < 0)['pnl'].mean() if losing_trades > 0 else 0
        
        print(f"  - Average gain: {avg_win:.4f}")
        print(f"  - Average loss: {avg_loss:.4f}")
        
        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss)
            print(f"  - Profit Factor: {profit_factor:.2f}")
        
        # Top 5 best and worst trades
        print(f"\n🏆 Top 5 best trades:")
        best_trades = trades_df.sort('pnl', descending=True).head(5)
        for row in best_trades.iter_rows(named=True):
            print(f"  {row['timestamp']}: {row['action']:4s} @ {row['price']:.2f}$ → PnL: {row['pnl']:.4f}")
        
        print(f"\n📉 Top 5 worst trades:")
        worst_trades = trades_df.sort('pnl').head(5)
        for row in worst_trades.iter_rows(named=True):
            print(f"  {row['timestamp']}: {row['action']:4s} @ {row['price']:.2f}$ → PnL: {row['pnl']:.4f}")

    def export_trades_to_csv(self, trades_df: pl.DataFrame, filename: str = "trades_history.csv"):
        """
        Export trades to CSV file.
        
        Args:
            trades_df: DataFrame of trades
            filename: Output file name
        """
        if len(trades_df) == 0:
            print("No trades to export")
            return
        
        trades_df.write_csv(filename)
        print(f"✅ Trades exported to {filename}")


# Usage example
if __name__ == "__main__":    
    # Define backtest function
    
    
    # Get data
    yh = YahooFinanceClient()
    df_init = yh.get_price("AAPL", from_date="2025-09-26", to_date="2025-10-04", 
                           interval="1m", postclean=True)
    
    # Load configuration
    opt_strategy = StrategyClient()
    opt_strategy.print_strategies()
    
    INDICATOR_CLASSES  = opt_strategy.get_seed_indicator_config()
    opt_client = OptimizationClient(INDICATOR_CLASSES)
    
    
    strat_name = "simple_strategy"
    opt_config = opt_strategy.get_seed_optimization_config()
    backtest_strategy = opt_strategy.get_strategy_fct(strat_name)
    
    
    # Launch optimization
    ta_client = TAClient()
    study = opt_client.optimize(opt_config, backtest_strategy, df_init, ta_client)
    
    # Save results
    opt_client.save_results("results")
    
    # Get best indicators
    best_indicators = opt_client.get_best_indicators()
    print(f"\nBest indicators: {best_indicators}")