try:
    import robosuite.macros_private as macros
except ImportError:
    import robosuite.macros as macros


def change_simulation_timestep(timestep: float):
    macros.SIMULATION_TIMESTEP = timestep
