"""CLI bindings for chart rendering."""

from __future__ import annotations

from typing import Callable, List

import typer

from quanta.utils.ta import INDICATOR_CLASSES

from .cli_help import PLOT_HELP, PLOT_OPTION_HELP


def register(
    app: typer.Typer,
    get_session: Callable[[], "QuantaSession"],
    parse_indicator_spec: Callable[[str], tuple[str, dict]],
) -> None:
    """Attach plotting commands to the main Typer app."""

    @app.command("plot", help=PLOT_HELP)
    def plot_command(
        cache_key: str = typer.Option(
            "prices",
            "--cache-key",
            "-k",
            help=PLOT_OPTION_HELP["cache_key"],
            is_flag=False,
        ),
        symbol_name: str = typer.Option(
            "Asset",
            "--symbol-name",
            "-n",
            help=PLOT_OPTION_HELP["symbol_name"],
            is_flag=False,
        ),
        indicators: List[str] = typer.Option(
            [],
            "--indicator",
            "-i",
            help=PLOT_OPTION_HELP["indicators"],
            is_flag=False,
        ),
        max_bars: int | None = typer.Option(
            None,
            "--max-bars",
            help=PLOT_OPTION_HELP["max_bars"],
            is_flag=False,
        ),
        x_axis_type: str = typer.Option(
            "row_nb",
            "--x-axis",
            help=PLOT_OPTION_HELP["x_axis_type"],
            is_flag=False,
        ),
    ) -> None:
        """Render a Plotly chart from a cached dataset."""

        session = get_session()
        try:
            df = session.cache.get(cache_key)
        except FileNotFoundError as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

        indicator_objs = []
        for spec in indicators:
            name, params = parse_indicator_spec(spec)
            try:
                indicator_objs.append(session.build_indicator(name, params))
            except KeyError:
                available = ", ".join(sorted(INDICATOR_CLASSES.keys()))
                typer.secho(f"Unknown indicator '{name}'. Available: {available}", fg=typer.colors.RED)
                raise typer.Exit(code=1)

        session.chart_client.plot(
            df,
            symbol=symbol_name,
            indicators=indicator_objs,
            max_bars=max_bars,
            x_axis_type=x_axis_type,
        )


__all__ = ["register"]
