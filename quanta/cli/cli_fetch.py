"""CLI bindings for market data downloads."""

from __future__ import annotations

from typing import Callable, Optional
from datetime import datetime, timedelta 
import polars as pl
import typer

from .cli_help import FETCH_HELP, FETCH_OPTION_HELP

date_today = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

def register(app: typer.Typer, get_session: Callable[[], "QuantaSession"]) -> None:
    """Attach fetch-related commands to the main Typer app."""

    @app.command("fetch", help=FETCH_HELP)
    def fetch_command(
        symbol: str = typer.Argument(..., help=FETCH_OPTION_HELP["symbol"]),
        from_date: str = typer.Option(
            ...,
            "--from",
            help=FETCH_OPTION_HELP["from_date"],
            is_flag=False,
        ),
        to_date: Optional[str] = typer.Option(
            date_today,
            "--to",
            help=FETCH_OPTION_HELP["to_date"],
            is_flag=False,
        ),
        interval: str = typer.Option(
            "1d",
            "--interval",
            "-i",
            help=FETCH_OPTION_HELP["interval"],
            is_flag=False,
        ),
        cache_key: Optional[str] = typer.Option(
            None,
            "--cache-key",
            "-k",
            help=FETCH_OPTION_HELP["cache_key"],
            is_flag=False,
        ),
    ) -> None:
        """Download OHLCV data from Yahoo Finance and cache it."""

        session = get_session()
        df = session.market_client.get_price(symbol, from_date=from_date, to_date=to_date, interval=interval)
        if df is None or not isinstance(df, pl.DataFrame):
            typer.secho("Unable to fetch historical prices with the provided parameters.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        target_key = cache_key or symbol
        session.cache.set(target_key, df, metadata={"interval": interval})
        typer.secho(
            f"Stored {len(df)} rows for {symbol} under cache key '{target_key}'.",
            fg=typer.colors.GREEN,
        )


__all__ = ["register"]
