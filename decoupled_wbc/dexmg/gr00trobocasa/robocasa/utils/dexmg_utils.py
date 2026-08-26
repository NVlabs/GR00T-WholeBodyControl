class DexMGConfigHelper:
    """
    Helper class for multi-inheritance scenarios, specifically designed to support task configuration
    and environment interaction in robotic manipulation tasks.

    Example Usage:
        This class is intended to be used in multi-inheritance cases such as:
            class PnPCounterToSink(Kitchen, DexMGConfigHelper)
        which will automatically generate a new configuration class:
            class PnPCounterToSink_Config(MG_Config)

    Behavior:
    1. In the implementation of the MG_RoboSuiteHumanoidGeneric class:
        - `self.env` will be an instance of PnPCounterToSink, enabling access to:
            - `self.env.get_object()` to retrieve key objects.
            - `self.env.get_subtask_term_signals()` to obtain signal information.
    2. In the implementation of PnPCounterToSink_Config:
        - `self.task_config()` delegates to `PnPCounterToSink.task_config`, which defines
          subtask divisions based on objects and signals.

    Attributes:
        subclasses (list): A list storing tuples of subclass names and their respective classes.
    """

    class AttrDict(dict):
        def __getattr__(self, key):
            if key not in self:
                self[key] = (
                    DexMGConfigHelper.AttrDict()
                )  # Create a new AttrDict if key doesn't exist
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

        def to_dict(self):
            """Recursively converts AttrDict to a normal dictionary"""
            return {
                key: (value.to_dict() if isinstance(value, DexMGConfigHelper.AttrDict) else value)
                for key, value in self.items()
            }

    subclasses = []

    def __init_subclass__(cls, **kwargs):
        """
        Automatically registers each subclass of DexMGConfigHelper.

        Args:
            cls: The subclass being registered.
            kwargs: Additional keyword arguments.
        """
        super().__init_subclass__(**kwargs)
        DexMGConfigHelper.subclasses.append((cls.__name__, cls))

    def __init__(self):
        """
        Initialize a DexMGConfigHelper instance.
        Note: This is an abstract class and should not be instantiated directly.
        """
        pass

    # Misc functions copied from dexmimicgen.mimicgen.env_interfaces.robosuite_humanoid
    def get_grippers(self):
        """
        Get grippers for the environment.
        """
        return self.robots[0].gripper["right"], self.robots[0].gripper["left"]

    def get_object(self):
        """
        Retrieve a key object required for the task.
        This method must be implemented in subclasses.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError

    def get_subtask_term_signals(self):
        """
        Retrieve signals used to define subtask termination conditions.
        This method must be implemented in subclasses.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError

    @staticmethod
    def task_config():
        """
        Define the configuration for dividing a task into subtasks.
        This method must be implemented in subclasses.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError
