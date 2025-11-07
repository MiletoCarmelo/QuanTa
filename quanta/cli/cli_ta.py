"""CLI bindings for technical analysis helpers."""

from __future__ import annotations

from typing import Callable, Dict, List

import typer

from quanta.utils.ta import INDICATOR_CLASSES

from .cli_help import TA_HELP, TA_OPTION_HELP


def register(
    app: typer.Typer,
    get_session: Callable[[], "QuantaSession"],
    parse_params: Callable[[List[str]], Dict[str, object]],
) -> None:
    """Attach TA-related commands to the main Typer app."""

    @app.command("ta", help=TA_HELP)
    def apply_indicator(
        indicator: str = typer.Argument(..., help=TA_OPTION_HELP["indicator"], metavar="LIST"),
        source_key: str = typer.Option(
            "prices",
            "--source",
            "-s",
            help=TA_OPTION_HELP["source_key"],
            is_flag=False,
        ),
        target_key: str | None = typer.Option(
            None,
            "--target",
            "-t",
            help=TA_OPTION_HELP["target_key"],
            is_flag=False,
        ),
        params: List[str] = typer.Option(
            [],
            "--param",
            "-p",
            help=TA_OPTION_HELP["params"],
            is_flag=False,
        ),
    ) -> None:
        """Apply an indicator to a cached DataFrame and persist the updated result."""

        session = get_session()
        try:
            df = session.cache.get(source_key)
        except FileNotFoundError as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

        try:
            indicator_obj = session.build_indicator(indicator, parse_params(params))
        except KeyError:
            available = ", ".join(sorted(INDICATOR_CLASSES.keys()))
            typer.secho(f"Unknown indicator '{indicator}'. Available: {available}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        enriched = session.ta_client.calculate_indicators(df, [indicator_obj])
        meta = session.cache.get_metadata(source_key)
        interval_meta = {"interval": meta["interval"]} if meta.get("interval") else None
        session.cache.set(target_key or source_key, enriched, metadata=interval_meta)
        typer.secho(
            f"Indicator '{indicator}' applied. Columns now available: {', '.join(indicator_obj.get_column_names())}",
            fg=typer.colors.GREEN,
        )


__all__ = ["register"]
