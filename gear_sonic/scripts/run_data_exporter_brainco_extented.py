"""Compatibility entry point for the modular BrainCo data exporter.

The historical filename contains the ``extented`` typo and remains available
so existing launch commands do not break.
"""

if __package__:
    from .brainco_data_exporter import *  # noqa: F401,F403
    from .brainco_data_exporter.cli import main as cli_main
else:
    from brainco_data_exporter import *  # noqa: F401,F403
    from brainco_data_exporter.cli import main as cli_main


if __name__ == "__main__":
    cli_main()
