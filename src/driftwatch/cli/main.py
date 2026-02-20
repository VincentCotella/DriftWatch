"""Main CLI application for DriftWatch.

Provides commands for drift checking and reporting.
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from driftwatch import Monitor
from driftwatch.core.report import DriftReport, DriftStatus

app = typer.Typer(
    name="driftwatch",
    help="🔍 DriftWatch - ML Model Drift Detection",
    add_completion=False,
)
console = Console()


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load dataframe from CSV or Parquet file.

    Args:
        path: Path to the file

    Returns:
        Loaded pandas DataFrame

    Raises:
        typer.BadParameter: If file format is not supported
    """
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    else:
        raise typer.BadParameter(
            f"Unsupported file format: {path.suffix}. Use .csv or .parquet"
        )


@app.command()
def check(
    ref: Annotated[
        Path,
        typer.Option(
            "--ref",
            "-r",
            help="Path to reference dataset (CSV or Parquet)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    prod: Annotated[
        Path,
        typer.Option(
            "--prod",
            "-p",
            help="Path to production dataset (CSV or Parquet)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    threshold_psi: Annotated[
        float, typer.Option("--threshold-psi", help="PSI threshold (default: 0.2)")
    ] = 0.2,
    threshold_ks: Annotated[
        float,
        typer.Option("--threshold-ks", help="KS p-value threshold (default: 0.05)"),
    ] = 0.05,
    threshold_chi2: Annotated[
        float,
        typer.Option(
            "--threshold-chi2", help="Chi-squared p-value threshold (default: 0.05)"
        ),
    ] = 0.05,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Save report to JSON file"),
    ] = None,
) -> None:
    """Check for drift between reference and production datasets.

    Example:
        driftwatch check --ref train.csv --prod prod.csv
        driftwatch check -r train.parquet -p prod.parquet --threshold-psi 0.15
    """
    console.print("[bold blue]🔍 DriftWatch - Drift Detection[/bold blue]\n")

    # Load datasets
    console.print(f"Loading reference data from [cyan]{ref}[/cyan]...")
    ref_df = load_dataframe(ref)
    console.print(
        f"✓ Loaded {len(ref_df):,} samples with {len(ref_df.columns)} features\n"
    )

    console.print(f"Loading production data from [cyan]{prod}[/cyan]...")
    prod_df = load_dataframe(prod)
    console.print(
        f"✓ Loaded {len(prod_df):,} samples with {len(prod_df.columns)} features\n"
    )

    # Create monitor
    console.print("Initializing monitor...")
    monitor = Monitor(
        reference_data=ref_df,
        thresholds={
            "psi": threshold_psi,
            "ks_pvalue": threshold_ks,
            "chi2_pvalue": threshold_chi2,
        },
    )

    # Run drift check
    console.print("Running drift detection...\n")
    report = monitor.check(prod_df)

    # Display results
    _display_report(report)

    # Save to file if requested
    if output:
        output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        console.print(f"\n✓ Report saved to [cyan]{output}[/cyan]")

    # Exit with appropriate code
    if report.status == DriftStatus.CRITICAL:
        raise typer.Exit(code=2)
    elif report.status == DriftStatus.WARNING:
        raise typer.Exit(code=1)
    else:
        raise typer.Exit(code=0)


@app.command()
def report(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Path to drift report JSON file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format (table or json)",
        ),
    ] = "table",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Save output to file"),
    ] = None,
) -> None:
    """Display a drift report from a JSON file.

    Example:
        driftwatch report drift_report.json
        driftwatch report drift_report.json --format json
        driftwatch report drift_report.json --format table --output report.txt
    """
    # Load report
    data = json.loads(input_file.read_text(encoding="utf-8"))

    # Reconstruct report (basic reconstruction)
    # In a real scenario, you'd have a from_dict method
    if format == "json":
        output_str = json.dumps(data, indent=2)
        if output:
            output.write_text(output_str, encoding="utf-8")
            console.print(f"✓ Report saved to [cyan]{output}[/cyan]")
        else:
            console.print(output_str)
    else:
        # Table format
        _display_dict_report(data)
        if output:
            # For table output to file, we'd need to capture the rich output
            console.print(
                "[yellow]Warning: Table output to file not yet implemented. Use --format json[/yellow]"
            )


def _display_report(report: DriftReport) -> None:
    """Display drift report with Rich formatting."""
    # Status
    status_colors = {
        DriftStatus.OK: "green",
        DriftStatus.WARNING: "yellow",
        DriftStatus.CRITICAL: "red",
    }
    color = status_colors.get(report.status, "white")
    console.print(f"[bold {color}]Status: {report.status.value}[/bold {color}]")

    if report.has_drift():
        console.print(
            f"[{color}]Drift Detected: {len(report.drifted_features())}/{len(report.feature_results)} features[/{color}]"
        )
        console.print(f"[{color}]Drift Ratio: {report.drift_ratio():.1%}[/{color}]\n")
    else:
        console.print("[green]No drift detected ✓[/green]\n")

    # Feature table
    console.print("[bold]Feature Analysis:[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Feature", style="cyan")
    table.add_column("Method")
    table.add_column("Score", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Status")

    for result in report.feature_results:
        status_str = "⚠️ DRIFT" if result.has_drift else "✓ OK"
        status_color = "red" if result.has_drift else "green"

        table.add_row(
            result.feature_name,
            result.method,
            f"{result.score:.4f}",
            f"{result.threshold:.4f}",
            f"[{status_color}]{status_str}[/{status_color}]",
        )

    console.print(table)


def _display_dict_report(data: dict) -> None:
    """Display drift report from dictionary data."""
    console.print(f"[bold]Status:[/bold] {data.get('status', 'UNKNOWN')}")

    if data.get("feature_results"):
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Feature", style="cyan")
        table.add_column("Method")
        table.add_column("Score", justify="right")
        table.add_column("Threshold", justify="right")
        table.add_column("Status")

        for result in data["feature_results"]:
            has_drift = result.get("has_drift", False)
            status_str = "⚠️ DRIFT" if has_drift else "✓ OK"
            status_color = "red" if has_drift else "green"

            table.add_row(
                result["feature_name"],
                result["method"],
                f"{result['score']:.4f}",
                f"{result['threshold']:.4f}",
                f"[{status_color}]{status_str}[/{status_color}]",
            )

        console.print(table)


@app.command()
def dashboard(
    port: Annotated[
        int,
        typer.Option(
            "--port", "-p", help="Port to run the dashboard on (default: 8501)"
        ),
    ] = 8501,
    browser: Annotated[
        bool,
        typer.Option(
            "--no-browser", help="Disable auto-opening of the browser", is_flag=True
        ),
    ] = True,
) -> None:
    """Launch the interactive DriftWatch Streamlit dashboard.

    Example:
        driftwatch dashboard
        driftwatch dashboard --port 8502
    """
    import subprocess
    import sys
    from pathlib import Path

    try:
        import streamlit  # noqa: F401
    except ImportError:
        console.print(
            "[bold red]❌ streamlit is not installed.[/bold red]\n"
            "Install it with: [cyan]pip install driftwatch\\[dashboard\\][/cyan]"
        )
        raise typer.Exit(code=1) from None

    app_path = Path(__file__).parent.parent / "dashboard" / "app.py"

    args = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        f"--server.port={port}",
        "--server.headless=false" if browser else "--server.headless=true",
    ]

    console.print(
        f"[bold blue]🔍 DriftWatch Dashboard[/bold blue] — "
        f"launching on [link=http://localhost:{port}]http://localhost:{port}[/link]"
    )
    subprocess.run(args, check=False)


if __name__ == "__main__":
    app()
