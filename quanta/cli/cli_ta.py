"""CLI bindings for technical analysis helpers."""

from __future__ import annotations

from datetime import datetime
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

        param_dict = parse_params(params)
        try:
            indicator_obj = session.build_indicator(indicator, param_dict)
        except KeyError:
            available = ", ".join(sorted(INDICATOR_CLASSES.keys()))
            typer.secho(f"Unknown indicator '{indicator}'. Available: {available}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        enriched = session.ta_client.calculate_indicators(df, [indicator_obj])
        target = target_key or source_key
        base_meta = session.cache.get_metadata(target)
        if not base_meta and target != source_key:
            base_meta = session.cache.get_metadata(source_key)

        indicators_meta = list(base_meta.get("indicators", []))
        indicator_entry = {
            "name": indicator,
            "params": param_dict,
            "columns": indicator_obj.get_column_names(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        indicators_meta = [
            entry
            for entry in indicators_meta
            if not (entry.get("name") == indicator_entry["name"] and entry.get("params") == indicator_entry["params"])
        ]
        indicators_meta.append(indicator_entry)

        metadata: Dict[str, object] = {"indicators": indicators_meta}
        if base_meta.get("interval"):
            metadata["interval"] = base_meta["interval"]

        session.cache.set(target, enriched, metadata=metadata)
        typer.secho(
            f"Indicator '{indicator}' applied. Columns now available: {', '.join(indicator_obj.get_column_names())}",
            fg=typer.colors.GREEN,
        )


__all__ = ["register"]
