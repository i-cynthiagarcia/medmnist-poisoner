"""Flower server entry point for clean baseline runs."""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from pytorchexample.experiments.shared.server import run_server

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    run_server(grid, context)
