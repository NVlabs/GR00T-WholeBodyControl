import subprocess

import click


@click.command()
def teleop():
    """CLI for interacting with the osmo."""
    subprocess.run(["python", "gr00t/control/teleop/gui/main.py"])


@click.group()
def cli():
    """CLI for interacting with the osmo."""


cli.add_command(teleop)

if __name__ == "__main__":
    cli()
