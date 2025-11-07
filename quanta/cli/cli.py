"""Top-level Typer app wiring for the QuanTa CLI."""

from __future__ import annotations

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


def _parse_indicator_spec(spec: str) -> Tuple[str, Dict[str, Any]]:
    if ":" not in spec:
        return spec, {}
    name, raw_params = spec.split(":", 1)
    params = {}
    for chunk in raw_params.split(","):
        if not chunk:
            continue
        if "=" not in chunk:
            raise typer.BadParameter(f"Indicator chunk '{chunk}' must use key=value.")
        key, value = chunk.split("=", 1)
        params[key] = _cast_value(value)
    return name, params


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
