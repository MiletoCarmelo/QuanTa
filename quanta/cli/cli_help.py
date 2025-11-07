"""Helpers for Typer help output and shared command help strings."""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from typing import Dict, Optional, Union, get_origin, get_args

import click
import typer

APP_HELP = "QuanTa CLI for fetching data, computing indicators, and plotting charts."
CACHE_HELP = "Manage cached datasets persisted on disk."
FETCH_HELP = "Download OHLCV data from Yahoo Finance and cache it."
TA_HELP = "Apply an indicator to a cached DataFrame and persist the updated result."
PLOT_HELP = "Render a Plotly chart from a cached dataset."

CACHE_OPTION_HELP: Dict[str, str] = {
    "status": "Show cached datasets along with last update timestamps.",
    "clear": "Remove every cached dataset from disk.",
    "ticker": "Display each cached ticker with its date range and average rows per day.",
}

FETCH_OPTION_HELP: Dict[str, str] = {
    "symbol": "Ticker symbol to download.",
    "from_date": "Start date (YYYY-MM-DD).",
    "to_date": "End date (YYYY-MM-DD).",
    "interval": "Yahoo Finance interval, e.g. 1d, 1h.",
    "cache_key": "Storage name for the downloaded dataset.",
}

TA_OPTION_HELP: Dict[str, str] = {
    "indicator": "Indicator name (e.g. SMA, RSI, MACD).",
    "source_key": "Cache key to read the DataFrame from.",
    "target_key": "Cache key to store the enriched DataFrame (defaults to source).",
    "params": "Indicator parameter in key=value form. Repeat for multiple parameters.",
}

PLOT_OPTION_HELP: Dict[str, str] = {
    "cache_key": "Cached DataFrame to visualise.",
    "symbol": "Display name for the chart title.",
    "indicators": "Indicator spec (NAME or NAME:param=value,param2=value). Repeat for multiple indicators.",
    "max_bars": "Limit the number of rows displayed.",
    "x_axis_type": "row_nb or datetime, passed to ChartClient.",
}

_HELP_CTX: ContextVar[click.Context | None] = ContextVar("_HELP_CTX", default=None)


def patch_click_help() -> None:
    """Work around Typer 0.9 calling Click 8.2's make_metavar without ctx."""

    signature = inspect.signature(click.Parameter.make_metavar)
    expects_ctx = "ctx" in signature.parameters and len(signature.parameters) == 2
    if not expects_ctx:
        return
    if getattr(click.Parameter.make_metavar, "_quanta_patched", False):
        return

    original_make_metavar = click.Parameter.make_metavar

    def _patched_make_metavar(self, ctx=None):
        ctx = ctx or _HELP_CTX.get()
        ctx = ctx or click.get_current_context(silent=True)
        if ctx is None:
            ctx = click.Context(click.Command(name=self.name or "cli"))
        return original_make_metavar(self, ctx)

    _patched_make_metavar._quanta_patched = True
    click.Parameter.make_metavar = _patched_make_metavar

    import typer.rich_utils  # Imported lazily to avoid unnecessary dependency at import time.

    original_print_options_panel = typer.rich_utils._print_options_panel

    def _patched_print_options_panel(*, name, params, ctx, markup_mode, console):
        token = _HELP_CTX.set(ctx)
        try:
            return original_print_options_panel(
                name=name,
                params=params,
                ctx=ctx,
                markup_mode=markup_mode,
                console=console,
            )
        finally:
            _HELP_CTX.reset(token)

    typer.rich_utils._print_options_panel = _patched_print_options_panel

    try:
        from typer.core import TyperArgument
    except ImportError:  # pragma: no cover
        TyperArgument = None  # type: ignore[assignment]

    if TyperArgument is not None and not getattr(TyperArgument.make_metavar, "_quanta_patched", False):

        def _patched_typer_argument_make_metavar(self, ctx=None):
            ctx = ctx or click.get_current_context(silent=True)
            if ctx is None:
                ctx = click.Context(click.Command(self.name or "cli"))

            if self.metavar is not None:
                return self.metavar
            var = (self.name or "").upper()
            if not self.required:
                var = f"[{var}]"
            type_var = None
            try:
                sig = inspect.signature(self.type.get_metavar)
                if "ctx" in sig.parameters:
                    type_var = self.type.get_metavar(self, ctx)
                else:
                    type_var = self.type.get_metavar(self)
            except (TypeError, ValueError):  # pragma: no cover
                type_var = self.type.get_metavar(self)
            if type_var:
                var += f":{type_var}"
            if self.nargs != 1:
                var += "..."
            return var

        _patched_typer_argument_make_metavar._quanta_patched = True
        TyperArgument.make_metavar = _patched_typer_argument_make_metavar  # type: ignore[assignment]

    _patch_typer_option_flags()


def _is_bool_annotation(annotation: object) -> bool:
    if annotation in (bool, Optional[bool]):
        return True
    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]  # noqa: E721
        return len(args) == 1 and args[0] is bool
    return False


def _patch_typer_option_flags() -> None:
    import typer.main

    if getattr(typer.main, "_quanta_patched_get_click_param", False):
        return

    original_get_click_param = typer.main.get_click_param

    def _patched_get_click_param(param):
        click_param, convertor = original_get_click_param(param)
        if isinstance(click_param, click.Option) and click_param.is_flag:
            annotation = getattr(param, "annotation", None)
            if not _is_bool_annotation(annotation):
                click_param.is_flag = False
                click_param.is_bool_flag = False
                click_param.flag_value = None
                click_param._flag_needs_value = False
        return click_param, convertor

    _patched_get_click_param._quanta_patched = True
    typer.main.get_click_param = _patched_get_click_param
    typer.main._quanta_patched_get_click_param = True


__all__ = [
    "APP_HELP",
    "CACHE_HELP",
    "CACHE_OPTION_HELP",
    "FETCH_HELP",
    "TA_HELP",
    "PLOT_HELP",
    "FETCH_OPTION_HELP",
    "TA_OPTION_HELP",
    "PLOT_OPTION_HELP",
    "patch_click_help",
]
