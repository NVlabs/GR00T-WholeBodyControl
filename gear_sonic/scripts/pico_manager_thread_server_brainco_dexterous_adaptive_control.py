"""Compatibility entry point for the modular Pico/BrainCo manager.

New code lives in :mod:`pico_manager_brainco_dexterous`. This wrapper keeps
existing launch commands and imports working.
"""

if __package__:
    from .pico_manager_brainco_dexterous import *  # noqa: F401,F403
    from .pico_manager_brainco_dexterous.cli import main
else:
    from pico_manager_brainco_dexterous import *  # noqa: F401,F403
    from pico_manager_brainco_dexterous.cli import main


if __name__ == "__main__":
    main()
