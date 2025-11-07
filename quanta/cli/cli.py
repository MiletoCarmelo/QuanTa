"""Top-level Typer app wiring for the QuanTa CLI."""

from __future__ import annotations

import inspect
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import typer

from quanta.clients.chart import ChartClient
from quanta.clients.yfinance import YahooFinanceClient
from quanta.utils.ta import INDICATOR_CLASSES, TAClient

from . import cli_cache, cli_fetch, cli_plot, cli_ta
from .cli_help import APP_HELP, patch_click_help

app = typer.Typer(help=APP_HELP)
patch_click_help()

_SESSION: Optional["QuantaSession"] = None


def _cast_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            continue
    return raw


def _parse_params(pairs: List[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    implicit_used = False
    for pair in pairs:
        if "=" not in pair:
            if implicit_used:
                raise typer.BadParameter(
                    f"Parameter '{pair}' must use key=value (implicit 'period' already set)."
                )
            params["period"] = _cast_value(pair)
            implicit_used = True
            continue
        key, value = pair.split("=", 1)
        params[key] = _cast_value(value)
    return params


def _parse_indicator_spec(spec: str) -> Tuple[str, List[Dict[str, Any]]]:
    if ":" not in spec:
        return spec, [{}]
    name, raw_params = spec.split(":", 1)
    key_to_values: Dict[str, List[Any]] = {}
    current_key: Optional[str] = None
    for chunk in raw_params.split(","):
        if not chunk:
            continue
        if "=" not in chunk:
            if current_key is None:
                raise typer.BadParameter(f"Indicator chunk '{chunk}' must use key=value.")
            key_to_values.setdefault(current_key, []).append(_cast_value(chunk))
            continue
        key, value = chunk.split("=", 1)
        current_key = key
        key_to_values.setdefault(key, []).append(_cast_value(value))
    params_list: List[Dict[str, Any]] = []
    if not key_to_values:
        params_list.append({})
    else:
        keys = list(key_to_values.keys())
        for combo in product(*(key_to_values[k] for k in keys)):
            params_list.append({k: v for k, v in zip(keys, combo)})
    return name, params_list


class QuantaSession:
    """Aggregates the different clients and the cache."""

    def __init__(self):
        self.cache = cli_cache.DataFrameCache()
        self.market_client = YahooFinanceClient()
        self.ta_client = TAClient()
        self.chart_client = ChartClient()

    def build_indicator(self, name: str, params: Optional[Dict[str, Any]] = None):
        cls = INDICATOR_CLASSES.get(name)
        if cls is None:
            raise KeyError(name)
        params = params or {}
        
        # Check if we have positional parameters (comma-separated values)
        # This happens when user passes --param 20,False,close
        # We look for any parameter value that contains commas
        for key, value in list(params.items()):
            if isinstance(value, str) and "," in value:
                values_str = [v.strip() for v in value.split(",")]
                
                # Get parameter names from __init__ signature
                init_sig = inspect.signature(cls.__init__)
                param_names = []
                for param_name, param in init_sig.parameters.items():
                    if param_name == "self":
                        continue
                    param_names.append(param_name)
                
                # Only parse if we're dealing with the first parameter (usually "period")
                # and the number of comma-separated values matches the number of parameters
                if key == param_names[0] if param_names else False:
                    if len(values_str) <= len(param_names):
                        # Remove the original parameter
                        del params[key]
                        # Map values to parameter names
                        for i, val_str in enumerate(values_str):
                            if i < len(param_names):
                                param_name = param_names[i]
                                # Cast value according to parameter annotation
                                param_obj = init_sig.parameters[param_name]
                                try:
                                    # Try to get type from annotation
                                    if param_obj.annotation != inspect.Parameter.empty:
                                        ann = param_obj.annotation
                                        # Handle Union types (like bool from typing)
                                        if hasattr(ann, '__origin__'):
                                            # For Union types, check args
                                            if bool in getattr(ann, '__args__', ()):
                                                val = val_str.lower() in ("true", "1", "yes", "on")
                                            elif int in getattr(ann, '__args__', ()):
                                                val = int(val_str)
                                            elif float in getattr(ann, '__args__', ()):
                                                val = float(val_str)
                                            else:
                                                val = _cast_value(val_str)
                                        elif ann == bool:
                                            # Handle bool values
                                            val = val_str.lower() in ("true", "1", "yes", "on")
                                        elif ann == int:
                                            val = int(val_str)
                                        elif ann == float:
                                            val = float(val_str)
                                        else:
                                            val = _cast_value(val_str)
                                    else:
                                        val = _cast_value(val_str)
                                    params[param_name] = val
                                except (ValueError, TypeError):
                                    # Fallback to string if casting fails
                                    params[param_name] = val_str
                        break  # Only process the first comma-separated parameter
        
        return cls(**params)


def get_session() -> QuantaSession:
    global _SESSION
    if _SESSION is None:
        _SESSION = QuantaSession()
    return _SESSION


cli_cache.register(app, get_session)
cli_fetch.register(app, get_session)
cli_ta.register(app, get_session, _parse_params)
cli_plot.register(app, get_session, _parse_indicator_spec)


def run() -> None:
    """Entry point for `python -m quanta.cli`."""

    app()


__all__ = ["app", "run", "get_session", "QuantaSession"]
