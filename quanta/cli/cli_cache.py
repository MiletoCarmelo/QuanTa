"""Cache management helpers and CLI bindings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import polars as pl
from polars.exceptions import ComputeError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

from .cli_help import CACHE_HELP, CACHE_OPTION_HELP

CONSOLE = Console()


@dataclass
class CacheRecord:
    key: str
    path: Path
    rows: int
    columns: List[str]
    updated_at: str
    interval: Optional[str] = None


class DataFrameCache:
    """Stores DataFrames both in-memory and on-disk (parquet + manifest)."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path.home() / ".quanta" / "cache"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.base_dir / "manifest.json"
        self._manifest = self._load_manifest()
        self._memory: Dict[str, pl.DataFrame] = {}

    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def _persist_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2))

    def _path_for(self, key: str) -> Path:
        safe_key = key.replace(" ", "_")
        return self.base_dir / f"{safe_key}.parquet"

    def set(self, key: str, df: pl.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> None:
        path = self._path_for(key)
        df.write_parquet(path)
        self._memory[key] = df
        info = (metadata or {}).copy()
        existing = self._manifest.get(key, {})
        interval = info.get("interval") or existing.get("interval") or self._infer_interval_label(df)
        indicators_meta = info.get("indicators")
        if indicators_meta is None:
            indicators_meta = existing.get("indicators", [])
        start_date, end_date, distinct_days, avg_rows = self._compute_date_stats(df)
        self._manifest[key] = {
            "path": str(path),
            "rows": len(df),
            "columns": df.columns,
            "updated_at": datetime.utcnow().isoformat(),
            "start_date": start_date,
            "end_date": end_date,
            "distinct_days": distinct_days,
            "avg_rows_per_day": avg_rows,
            "interval": interval,
            "indicators": indicators_meta,
        }
        self._persist_manifest()

    def _remove_file(self, path_str: str) -> None:
        path = Path(path_str)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def get(self, key: str) -> pl.DataFrame:
        if key in self._memory:
            return self._memory[key]
        path = self._path_for(key)
        if not path.exists():
            raise FileNotFoundError(f"No cache entry named '{key}'.")
        df = pl.read_parquet(path)
        self._memory[key] = df
        return df

    def list_records(self) -> List[CacheRecord]:
        records: List[CacheRecord] = []
        for key, meta in self._manifest.items():
            path = Path(meta["path"])
            records.append(
                CacheRecord(
                    key=key,
                    path=path,
                    rows=meta.get("rows", 0),
                    columns=meta.get("columns", []),
                    updated_at=meta.get("updated_at", "unknown"),
                    interval=meta.get("interval"),
                )
            )
        return records

    def list_keys(self) -> List[str]:
        return sorted(self._manifest.keys())

    def describe_keys(self) -> List[Dict[str, Any]]:
        dirty = False
        summaries: List[Dict[str, Any]] = []
        for key in sorted(self._manifest.keys()):
            meta = self._manifest.get(key, {})
            start = meta.get("start_date")
            end = meta.get("end_date")
            distinct = meta.get("distinct_days")
            avg = meta.get("avg_rows_per_day")
            interval = meta.get("interval")
            df: Optional[pl.DataFrame] = None
            if start is None or end is None:
                try:
                    df = self.get(key)
                except FileNotFoundError:
                    df = None
                if df is not None:
                    start, end, distinct, avg = self._compute_date_stats(df)
                    meta["start_date"] = start
                    meta["end_date"] = end
                    meta["distinct_days"] = distinct
                    meta["avg_rows_per_day"] = avg
                    dirty = True
            if interval is None:
                if df is None:
                    try:
                        df = self.get(key)
                    except FileNotFoundError:
                        df = None
                if df is not None:
                    interval = self._infer_interval_label(df)
                    if interval is not None:
                        meta["interval"] = interval
                        dirty = True
            summaries.append(
                {
                    "key": key,
                    "rows": meta.get("rows", 0),
                    "start_date": start,
                    "end_date": end,
                    "distinct_days": distinct,
                    "avg_rows_per_day": avg,
                    "updated_at": meta.get("updated_at"),
                    "interval": interval,
                }
            )
        if dirty:
            try:
                self._persist_manifest()
            except OSError:
                # Cache directory might be read-only; ignore persistence errors.
                pass
        return summaries

    def list_indicator_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for key, meta in self._manifest.items():
            for entry in meta.get("indicators", []) or []:
                records.append(
                    {
                        "key": key,
                        "name": entry.get("name", "unknown"),
                        "params": entry.get("params", {}),
                        "columns": entry.get("columns", []),
                        "updated_at": entry.get("updated_at", meta.get("updated_at", "unknown")),
                    }
                )
        return records

    def delete(self, key: str) -> bool:
        meta = self._manifest.pop(key, None)
        self._memory.pop(key, None)
        if meta is None:
            return False
        self._remove_file(meta["path"])
        self._persist_manifest()
        return True

    def clear_all(self) -> int:
        removed = 0
        for key, meta in list(self._manifest.items()):
            self._remove_file(meta["path"])
            self._memory.pop(key, None)
            removed += 1
        self._manifest.clear()
        self._persist_manifest()
        return removed

    def get_metadata(self, key: str) -> Dict[str, Any]:
        return self._manifest.get(key, {}).copy()

    def _select_datetime_series(self, df: pl.DataFrame) -> Tuple[Optional[str], Optional[pl.Series]]:
        for candidate in ("datetime", "date", "timestamp"):
            if candidate not in df.columns:
                continue
            converted = self._convert_to_datetime_series(df[candidate])
            if converted is not None:
                return candidate, converted
        return None, None

    def _convert_to_datetime_series(self, series: pl.Series) -> Optional[pl.Series]:
        dtype = series.dtype
        if dtype == pl.Datetime:
            return series
        if dtype == pl.Date:
            return series.cast(pl.Datetime)
        if dtype == pl.Utf8:
            for target in (pl.Datetime, pl.Date):
                try:
                    parsed = series.str.strptime(target, strict=False)
                except ComputeError:
                    parsed = None
                if parsed is None:
                    continue
                if parsed.null_count() < parsed.len():
                    if target == pl.Date:
                        parsed = parsed.cast(pl.Datetime)
                    return parsed
        return None

    @staticmethod
    def _format_timestamp(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return str(value)

    def _compute_date_stats(self, df: pl.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[float]]:
        _, series = self._select_datetime_series(df)
        if series is None:
            return None, None, None, None
        clean = series.drop_nulls()
        if clean.len() == 0:
            return None, None, None, None
        start = clean.min()
        end = clean.max()
        try:
            distinct_days = clean.dt.date().n_unique()
        except AttributeError:
            distinct_days = None
        avg_rows = None
        if distinct_days and distinct_days > 0:
            avg_rows = round(clean.len() / distinct_days, 2)
        return (
            self._format_timestamp(start),
            self._format_timestamp(end),
            distinct_days,
            avg_rows,
        )

    def _infer_interval_label(self, df: pl.DataFrame) -> Optional[str]:
        _, series = self._select_datetime_series(df)
        if series is None:
            return None
        values = series.drop_nulls().sort().to_list()
        if len(values) < 2:
            return None
        min_seconds: Optional[float] = None
        prev = values[0]
        for current in values[1:]:
            if current is None or prev is None:
                prev = current
                continue
            delta = (current - prev).total_seconds()
            if delta <= 0:
                prev = current
                continue
            if min_seconds is None or delta < min_seconds:
                min_seconds = delta
            prev = current
        if min_seconds is None:
            return None
        return self._format_interval_label(min_seconds)

    @staticmethod
    def _format_interval_label(seconds: float) -> str:
        units = [
            ("d", 86400),
            ("h", 3600),
            ("m", 60),
            ("s", 1),
        ]
        for suffix, size in units:
            value = seconds / size
            if value >= 1:
                if abs(value - round(value)) < 1e-6:
                    value = round(value)
                if suffix == "s" and value < 1:
                    return f"{seconds:.2f}s"
                return f"{value:g}{suffix}"
        return f"{seconds:.2f}s"


def register(app: typer.Typer, get_session: Callable[[], "QuantaSession"]) -> None:
    """Attach cache-related commands to the main Typer app."""

    @app.command("cache", help=CACHE_HELP)
    def cache_command(
        status: bool = typer.Option(
            False,
            "--status",
            "-s",
            help=CACHE_OPTION_HELP["status"],
            is_flag=True,
        ),
        clear: bool = typer.Option(
            False,
            "--clear",
            "-c",
            help=CACHE_OPTION_HELP["clear"],
            is_flag=True,
        ),
        ticker: bool = typer.Option(
            False,
            "--ticker",
            "-t",
            help=CACHE_OPTION_HELP["ticker"],
            is_flag=True,
        ),
        indicators: bool = typer.Option(
            False,
            "--indicators",
            "-i",
            help=CACHE_OPTION_HELP["indicators"],
            is_flag=True,
        ),
    ) -> None:
        """Perform cache maintenance operations."""

        session = get_session()
        cache = session.cache

        def _flag(value: object) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                # Click 8.3 + Typer 0.12 return None when a flag is provided.
                return True
            if isinstance(value, str):
                return value.lower() in {"1", "true", "yes", "on"}
            return bool(value)

        status_flag, clear_flag, ticker_flag, indicators_flag = (
            _flag(status),
            _flag(clear),
            _flag(ticker),
            _flag(indicators),
        )

        selected_flags = sum(int(flag) for flag in (status_flag, clear_flag, ticker_flag, indicators_flag))
        if selected_flags > 1:
            typer.secho("Please select only one of --status, --clear, --ticker, or --indicators.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        if selected_flags == 0:
            status_flag = True

        if ticker_flag:
            summaries = cache.describe_keys()
            if not summaries:
                typer.echo("No cached datasets yet. Use 'fetch' to pull market data.")
                raise typer.Exit()

            table = Table(
                show_header=True,
                header_style="bold",
                expand=True,
            )
            table.add_column("Key", style="bold", no_wrap=True)
            table.add_column("Interval", justify="center")
            table.add_column("Rows", justify="right")
            table.add_column("Days", justify="right")
            table.add_column("Avg/Day", justify="right")
            table.add_column("Start", overflow="fold")
            table.add_column("End", overflow="fold")
            table.add_column("Updated", overflow="fold")

            total_rows = 0
            total_days = 0

            for item in summaries:
                rows = item["rows"] or 0
                days = item["distinct_days"] or 0
                avg_rows = item["avg_rows_per_day"]
                avg_display = f"{avg_rows:.2f}" if avg_rows is not None else "-"
                days_display = str(days) if days else "-"
                start_display = item["start_date"] or "-"
                end_display = item["end_date"] or "-"
                updated_display = item["updated_at"] or "-"
                table.add_row(
                    item["key"],
                    item.get("interval") or "-",
                    str(rows),
                    days_display,
                    avg_display,
                    start_display,
                    end_display,
                    updated_display,
                )
                total_rows += rows
                total_days += days or 0

            panel = Panel(table, title="Tickers", border_style="cyan", expand=False)
            CONSOLE.print(panel)

            if total_days:
                overall_avg = total_rows / total_days
                CONSOLE.print(f"[bold]Average rows per day across cache:[/] {overall_avg:.2f}")
            return

        if clear_flag:
            removed = cache.clear_all()
            typer.secho(f"Cleared {removed} cached dataset(s).", fg=typer.colors.YELLOW)
            return

        if indicators_flag:
            indicator_records = cache.list_indicator_records()
            if not indicator_records:
                typer.echo("No indicators recorded yet. Use 'ta' to enrich a dataset.")
                raise typer.Exit()
            table = Table(show_header=True, header_style="bold", expand=True)
            table.add_column("Key", style="bold", no_wrap=True)
            table.add_column("Indicator")
            table.add_column("Params")
            table.add_column("Columns")
            table.add_column("Updated", overflow="fold")

            for entry in indicator_records:
                params_dict = entry.get("params") or {}
                params_display = ", ".join(f"{k}={v}" for k, v in sorted(params_dict.items())) or "-"
                columns_display = ", ".join(entry.get("columns") or []) or "-"
                table.add_row(
                    entry["key"],
                    entry.get("name", "-"),
                    params_display,
                    columns_display,
                    entry.get("updated_at", "-"),
                )

            panel = Panel(table, title="Indicators", border_style="magenta", expand=False)
            CONSOLE.print(panel)
            return

        # Status path (default)
        records = cache.list_records()
        if not records:
            typer.echo("No cached datasets yet. Use 'fetch' to pull market data.")
            raise typer.Exit()
        for record in records:
            cols_preview = ", ".join(record.columns[:5])
            if len(record.columns) > 5:
                cols_preview += ", ..."
            typer.echo(
                f"[{record.key}] rows={record.rows} cols={len(record.columns)} "
                f"({cols_preview}) updated={record.updated_at} interval={record.interval or '-'}"
            )


__all__ = ["CacheRecord", "DataFrameCache", "register"]
