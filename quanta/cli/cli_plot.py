"""CLI bindings for chart rendering."""

from __future__ import annotations

from typing import Callable, List

import typer

from quanta.utils.ta import INDICATOR_CLASSES, Volatility, RSI, ATR
from quanta.utils.trace import Line

from .cli_help import PLOT_HELP, PLOT_OPTION_HELP


def register(
    app: typer.Typer,
    get_session: Callable[[], "QuantaSession"],
    parse_indicator_spec: Callable[[str], tuple[str, List[dict]]],
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

        # Split comma-separated values in indicators list
        expanded_indicators = []
        for spec in indicators:
            # Support comma-separated values like "SMA50,RSI14,Volatility15"
            if "," in spec:
                expanded_indicators.extend([s.strip() for s in spec.split(",")])
            else:
                expanded_indicators.append(spec)
        
        indicator_objs = []
        traces = []
        
        for spec in expanded_indicators:
            # Check if this is an existing column name in the DataFrame
            if spec in df.columns:
                # Try to detect if this column is from a known indicator type
                # and create the indicator object so it's rendered in the correct subplot
                indicator_created = False
                
                # Check for Volatility columns (e.g., Volatility15, Volatility20)
                if spec.startswith('Volatility'):
                    try:
                        # Extract period from column name (e.g., "Volatility15" -> 15)
                        period_str = spec.replace('Volatility', '')
                        if period_str.isdigit():
                            period = int(period_str)
                            # Check if we can determine annualize from the data
                            # For now, create with default and let it use the existing column
                            indicator_obj = Volatility(period=period)
                            indicator_obj.name = spec  # Override name to match column
                            indicator_objs.append(indicator_obj)
                            indicator_created = True
                    except (ValueError, AttributeError):
                        pass
                
                # Check for RSI columns (e.g., RSI14, RSI20)
                elif spec.startswith('RSI'):
                    try:
                        period_str = spec.replace('RSI', '')
                        if period_str.isdigit():
                            period = int(period_str)
                            indicator_obj = RSI(period=period)
                            indicator_obj.name = spec  # Override name to match column
                            indicator_objs.append(indicator_obj)
                            indicator_created = True
                    except (ValueError, AttributeError):
                        pass
                
                # Check for ATR columns (e.g., ATR14)
                elif spec.startswith('ATR'):
                    try:
                        period_str = spec.replace('ATR', '')
                        if period_str.isdigit():
                            period = int(period_str)
                            indicator_obj = ATR(period=period)
                            indicator_obj.name = spec  # Override name to match column
                            indicator_objs.append(indicator_obj)
                            indicator_created = True
                    except (ValueError, AttributeError):
                        pass
                
                # If not a recognized indicator column, create a Line trace for overlay
                if not indicator_created:
                    traces.append(Line(spec, name=spec))
            else:
                # It's an indicator specification - parse and build
                name, params_list = parse_indicator_spec(spec)
                try:
                    for params in params_list:
                        indicator_objs.append(session.build_indicator(name, params))
                except KeyError:
                    available = ", ".join(sorted(INDICATOR_CLASSES.keys()))
                    available_cols = ", ".join(sorted([col for col in df.columns if col not in ['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]))
                    if available_cols:
                        typer.secho(
                            f"Unknown indicator/column '{spec}'. Available indicators: {available}\n"
                            f"Available columns: {available_cols}",
                            fg=typer.colors.RED
                        )
                    else:
                        typer.secho(f"Unknown indicator '{spec}'. Available: {available}", fg=typer.colors.RED)
                    raise typer.Exit(code=1)

        # Prepare traces (add candlesticks if we have custom traces)
        from quanta.utils.trace import Candlesticks
        if traces:
            # If we have column traces, add candlesticks
            final_traces = [Candlesticks()] + traces
        else:
            final_traces = None
        
        session.chart_client.plot(
            df,
            symbol=symbol_name,
            traces=final_traces,
            indicators=indicator_objs if indicator_objs else None,
            max_bars=max_bars,
            x_axis_type=x_axis_type,
        )


__all__ = ["register"]
