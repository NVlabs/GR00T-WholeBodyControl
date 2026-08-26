"""Config-driven neural network modules for actor-critic policy networks."""

import inspect

import torch.nn as nn
import torchvision.models as models


def get_norm(norm_type, dim):
    """Build a normalization layer from a string identifier.

    Args:
        norm_type: Normalization type string ("layer_norm") or None to skip.
        dim: Feature dimension for the normalization layer.

    Returns:
        An ``nn.Module`` normalization layer, or None if ``norm_type`` is None.
    """
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim)
    elif norm_type is None:
        return None
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")


class ResidualBlock(nn.Module):
    """Single pre-activation residual block: ``x + Linear -> Norm -> Act(x)``.

    Uses a skip connection around a linear-norm-activation sequence so gradients
    flow unimpeded through the identity path. The linear layer preserves
    dimensionality (``dim -> dim``).

    Args:
        dim: Feature dimension (input and output are the same size).
        norm_type: Normalization type applied after the linear layer.
        activation: Name of an ``nn.Module`` activation class (e.g. "SiLU").
    """

    def __init__(self, dim, norm_type="layer_norm", activation="SiLU"):
        super().__init__()
        layers = [nn.Linear(dim, dim)]
        norm = get_norm(norm_type, dim)
        if norm:
            layers.append(norm)
        layers.append(getattr(nn, activation)())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """Forward with additive residual connection.

        Args:
            x: Input tensor of shape ``(*, dim)``.

        Returns:
            Tensor of same shape with residual added.
        """
        return x + self.block(x)


class ResidualMLP(nn.Module):
    """MLP with stacked residual blocks between input and output projections.

    Architecture: ``Linear(in->hidden) -> Norm -> Act -> [ResidualBlock]*depth
    -> Linear(hidden->out)``. The residual blocks maintain gradient flow through
    deep networks, while the input/output projections handle dimension changes.

    Args:
        input_dim: Size of the input feature vector.
        hidden_dim: Width of all residual blocks.
        output_dim: Size of the output feature vector.
        depth: Number of stacked ``ResidualBlock`` layers.
        norm: Normalization type passed to each block.
        activation: Activation function name for all layers.
    """

    def __init__(
        self, input_dim, hidden_dim, output_dim, depth, norm="layer_norm", activation="SiLU"
    ):
        super().__init__()

        # Input projection
        input_layers = [nn.Linear(input_dim, hidden_dim)]
        norm_layer = get_norm(norm, hidden_dim)
        if norm_layer:
            input_layers.append(norm_layer)
        input_layers.append(getattr(nn, activation)())
        self.input_layer = nn.Sequential(*input_layers)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(hidden_dim, norm_type=norm, activation=activation)
                for _ in range(depth)
            ]
        )

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward through input projection, residual stack, and output projection.

        Args:
            x: Input tensor of shape ``(*, input_dim)``.

        Returns:
            Output tensor of shape ``(*, output_dim)``.
        """
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)


class BaseModule(nn.Module):
    """Config-driven network module that auto-builds layers from a dictionary spec.

    Resolves input/output dimensions from observation dictionaries or explicit
    overrides, then dispatches to a layer builder (MLP, CNN, GRU, ResidualMLP,
    ResNet) based on ``module_config_dict.layer_config.type``. Temporal
    dimensions are handled by flattening on input and reshaping on output.

    Args:
        obs_dim_dict: Mapping of observation name to its flat dimension.
            Falls back to ``env_config.robot.algo_obs_dim_dict`` if None.
        module_config_dict: Config with ``input_dim``, ``output_dim``, and
            ``layer_config`` keys that drive network construction.
        module_dim_dict: Mapping of named module outputs to their dimensions,
            used to resolve symbolic references in ``input_dim``/``output_dim``.
        env_config: Environment configuration (provides observation dims, camera
            settings, and robot action dims).
        algo_config: Algorithm configuration (stored but not used directly).
        process_output_dim: If True, replace ``"robot_action_dim"`` sentinel
            values in ``output_dim`` with the actual action dimension.
        input_dim: Explicit input dimension override (skips calculation).
        output_dim: Explicit output dimension override (skips calculation).
        num_input_temporal_dims: If set, input's last two dims are flattened
            (``temporal * feature -> input_dim``).
        num_output_temporal_dims: If set, output is reshaped to
            ``(*, num_output_temporal_dims, feature_per_step)``.
    """

    def __init__(
        self,
        obs_dim_dict=None,
        module_config_dict=None,
        module_dim_dict={},
        env_config=None,
        algo_config=None,
        process_output_dim=False,
        input_dim=None,
        output_dim=None,
        num_input_temporal_dims=None,
        num_output_temporal_dims=None,
    ):
        super().__init__()

        self.env_config = env_config
        self.algo_config = algo_config
        if obs_dim_dict is None:
            if env_config is not None:
                self.obs_dim_dict = env_config.robot.algo_obs_dim_dict
        else:
            self.obs_dim_dict = obs_dim_dict

        self.module_config_dict = module_config_dict
        if process_output_dim:
            self.module_config_dict = self._process_module_config(
                self.module_config_dict, self.env_config.robot.actions_dim
            )

        self.module_dim_dict = module_dim_dict

        if input_dim is None:
            self._calculate_input_dim()
        else:
            self.input_dim = input_dim
        if output_dim is None:
            self._calculate_output_dim()
        else:
            self.output_dim = output_dim
        self.num_input_temporal_dims = num_input_temporal_dims
        self.num_output_temporal_dims = num_output_temporal_dims
        if num_input_temporal_dims is not None:
            self.input_dim *= num_input_temporal_dims
        if num_output_temporal_dims is not None:
            self.output_dim *= num_output_temporal_dims

        self._build_network_layer(self.module_config_dict.layer_config)

    def _process_module_config(self, module_config_dict, num_actions):
        """Replace ``"robot_action_dim"`` sentinels with the actual action count.

        Args:
            module_config_dict: Module config containing an ``output_dim`` list.
            num_actions: Number of robot action dimensions to substitute.

        Returns:
            The mutated ``module_config_dict`` with sentinels replaced.
        """
        output_dim_list = module_config_dict["output_dim"]
        if isinstance(output_dim_list, int):
            output_dim_list = [output_dim_list]

        for idx, output_dim in enumerate(output_dim_list):
            if output_dim == "robot_action_dim":
                module_config_dict["output_dim"][idx] = num_actions
        return module_config_dict

    def _calculate_input_dim(self):
        """Calculate total input dimension by summing over ``module_config_dict["input_dim"]``.

        Each entry is resolved as an observation name (looked up in
        ``obs_dim_dict``), a numeric literal, or a named module output
        (looked up in ``module_dim_dict``). Sets ``self.input_dim``.
        """
        # calculate input dimension based on the input specifications
        input_dim = 0
        for each_input in self.module_config_dict["input_dim"]:
            if each_input in self.obs_dim_dict:
                # atomic observation type
                input_dim += self.obs_dim_dict[each_input]
            elif isinstance(each_input, int | float):
                # direct numeric input
                input_dim += each_input
            elif each_input in self.module_dim_dict:
                input_dim += self.module_dim_dict[each_input]
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown input type: {each_input}")

        self.input_dim = input_dim

    def _calculate_output_dim(self):
        """Calculate total output dimension by summing over ``module_config_dict["output_dim"]``.

        Each entry is resolved as a numeric literal or a named module output.
        Sets ``self.output_dim``.
        """
        output_dim = 0
        output_dim_list = self.module_config_dict["output_dim"]
        if isinstance(output_dim_list, int) or isinstance(output_dim_list, str):
            output_dim_list = [output_dim_list]

        for each_output in output_dim_list:
            if isinstance(each_output, int | float):
                output_dim += each_output
            elif each_output in self.module_dim_dict:
                output_dim += self.module_dim_dict[each_output]
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown output type: {each_output}")

        self.output_dim = output_dim

    def _build_network_layer(self, layer_config):
        """Dispatch to the appropriate layer builder based on ``layer_config["type"]``.

        Supported types: ``"MLP"``, ``"CNN"``, ``"GRU"``, ``"ResidualMLP"``,
        ``"ResNet"``. The built network is stored as ``self.module``.

        Args:
            layer_config: Dict with a ``"type"`` key and type-specific params.
        """
        if layer_config["type"] == "MLP":
            self._build_mlp_layer(layer_config)
        elif layer_config["type"] == "CNN":
            self._build_cnn_layer(layer_config)
        elif layer_config["type"] == "GRU":
            self._build_gru_layer(layer_config)
        elif layer_config["type"] == "ResidualMLP":
            self._build_residual_mlp_layer(layer_config)
        elif layer_config["type"] == "ResNet":
            self._build_resnet_layer(layer_config)
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_config['type']}")

    def _build_mlp_layer(self, layer_config):
        """Build a plain MLP with configurable hidden dims and activation.

        Architecture: ``input -> [hidden_i -> Act]* -> output``. No residual
        connections or normalization (use ``ResidualMLP`` for those).

        Args:
            layer_config: Dict with ``"hidden_dims"`` (list of ints) and
                ``"activation"`` (nn.Module class name).
        """
        layers = []
        hidden_dims = layer_config["hidden_dims"]
        output_dim = self.output_dim
        activation = getattr(nn, layer_config["activation"])()

        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(activation)

        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)

        self.module = nn.Sequential(*layers)

    def _build_cnn_layer(self, layer_config):
        """Build a CNN encoder from env camera config and layer specifications.

        Constructs conv/pool layers from ``layer_config["layers"]``, resolves
        input spatial dims and channel count from ``env_config.simulator``, and
        appends a flatten + linear projection to ``self.output_dim``.

        NOTE: Input is a flattened vision observation vector that gets reshaped
        to ``(width, height, channels)`` based on camera config. The assertion
        verifies the flat dimension matches the expected spatial size.

        Args:
            layer_config: Dict with ``"channel_dims"`` (list of ints),
                ``"activation"``, and ``"layers"`` (list of layer dicts with
                ``"type"`` of ``"conv"`` or ``"pool"``).
        """
        layers = []
        channel_dims = layer_config["channel_dims"]
        activation = getattr(nn, layer_config["activation"])()

        # Get input dimensions from env_config camera settings
        camera_config = self.env_config.simulator.config.cameras
        input_height = camera_config.camera_resolutions[0]
        input_width = camera_config.camera_resolutions[1]

        # Determine number of channels from camera types
        input_channels = 0
        for camera_type in camera_config.camera_types:
            if camera_type.get("rgb", False):
                input_channels += 3
            if camera_type.get("depth", False):
                input_channels += 1

        # If no channels found, default to 1
        if input_channels == 0:
            input_channels = 1

        vision_obs_dim = [input_width, input_height, input_channels]
        print("vision_obs_dim", vision_obs_dim)
        assert (
            vision_obs_dim[0] * vision_obs_dim[1] * vision_obs_dim[2]
            == self.obs_dim_dict["vision_obs"]
        )
        if len(vision_obs_dim) != 3:
            raise ValueError(
                f"vision_obs dimension should be (width, height, channels), got {vision_obs_dim}"
            )
        input_width, input_height, input_channels = vision_obs_dim

        # Get layer configurations
        layer_configs = layer_config.get("layers", [])
        use_batch_norm = layer_config.get("norm_config", {}).get("use_batch_norm", False)

        # Track spatial dimensions and channels
        current_height, current_width = input_height, input_width
        current_channels = input_channels
        conv_idx = 0  # Track which conv layer we're on for channel dimensions

        for layer_cfg in layer_configs:
            layer_type = layer_cfg["type"]

            if layer_type == "conv":
                # Get conv parameters
                kernel_size = layer_cfg.get("kernel_size", 3)
                stride = layer_cfg.get("stride", 1)
                padding = layer_cfg.get("padding", 1)

                # Determine output channels
                if conv_idx < len(channel_dims):
                    out_channels = channel_dims[conv_idx]
                else:
                    out_channels = self.output_dim

                # Add conv layer
                layers.append(
                    nn.Conv2d(
                        current_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )

                if use_batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(activation)

                # Update dimensions
                current_channels = out_channels
                current_height = (current_height - kernel_size + 2 * padding) // stride + 1
                current_width = (current_width - kernel_size + 2 * padding) // stride + 1
                conv_idx += 1

            elif layer_type == "pool":
                # Get pool parameters
                kernel_size = layer_cfg.get("kernel_size", 2)
                stride = layer_cfg.get("stride", 2)

                # Add pooling layer if dimensions allow
                if current_height >= kernel_size and current_width >= kernel_size:
                    layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride))
                    current_height = current_height // stride
                    current_width = current_width // stride

        # Add global average pooling if spatial dimensions are too small
        # if current_height * current_width > 1:
        #     # import ipdb; ipdb.set_trace()
        #     layers.append(nn.AdaptiveAvgPool2d(1))

        layers.append(nn.Flatten())

        layers.append(nn.Linear(current_channels * current_height * current_width, self.output_dim))

        self.module = nn.Sequential(*layers)

    def forward_without_hidden_state(self, input):
        """Forward pass for stateless modules (MLP, CNN, ResidualMLP, ResNet).

        Args:
            input: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Output tensor of shape ``(batch, output_dim)``.
        """
        return self.module(input)

    def forward_with_hidden_state(self, input, hidden_state):
        """Forward pass for recurrent modules (GRU).

        Args:
            input: Input tensor of shape ``(batch, seq_len, input_dim)``.
            hidden_state: Previous hidden state tensor.

        Returns:
            Tuple of (output, updated_hidden_state).
        """
        # import ipdb; ipdb.set_trace()
        output, hidden_state = self.module(input, hidden_state)
        return output, hidden_state

    # def forward(self, input, hidden_state=None):
    #     if hidden_state is None:
    #         return self.forward_without_hidden_state(input)
    #     else:
    #         return self.forward_with_hidden_state(input, hidden_state)

    def _build_gru_layer(self, layer_config):
        """Build a GRU recurrent layer.

        Args:
            layer_config: Dict with ``"hidden_dim"`` and ``"num_layers"``.
        """
        self.module = nn.GRU(
            input_size=self.input_dim,
            hidden_size=layer_config["hidden_dim"],
            num_layers=layer_config["num_layers"],
            batch_first=True,
        )

    def _build_resnet_layer(self, layer_config):
        """Build a torchvision ResNet backbone with global avg pool and linear head.

        Removes the original classification head (avgpool + fc) and replaces
        it with ``AdaptiveAvgPool2d(1) -> Flatten -> Linear(feat, output_dim)``.

        Args:
            layer_config: Dict with ``"resnet_type"`` (e.g. ``"resnet18"``),
                ``"pretrained"`` (bool), and ``"trainable"`` (bool, freezes
                backbone params when False).
        """
        print("Building ResNet layer")
        resnet_type = layer_config.get("resnet_type", "resnet18")  # Default to resnet18
        pretrained = layer_config.get("pretrained", True)
        trainable = layer_config.get("trainable", True)

        if resnet_type == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
        elif resnet_type == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
        elif resnet_type == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
        elif resnet_type == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
        elif resnet_type == "resnet152":
            resnet = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        resnet_features = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc

        if resnet_type in ["resnet18", "resnet34"]:
            resnet_feature_dim = 512
        else:  # resnet50, resnet101, resnet152
            resnet_feature_dim = 2048

        # Freeze ResNet parameters if not trainable
        if not trainable:
            for param in resnet_features.parameters():
                param.requires_grad = False

        # Add a final linear layer to match output_dim
        layers = [
            resnet_features,
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(resnet_feature_dim, self.output_dim),
        ]

        self.module = nn.Sequential(*layers)

    def _build_residual_mlp_layer(self, layer_config):
        """Build a ``ResidualMLP`` from layer config.

        Args:
            layer_config: Dict with ``"hidden_dim"``, ``"depth"``, and
                optional ``"norm"`` and ``"activation"``.
        """
        self.module = ResidualMLP(
            input_dim=self.input_dim,
            hidden_dim=layer_config["hidden_dim"],
            output_dim=self.output_dim,
            depth=layer_config["depth"],
            norm=layer_config.get("norm", "layer_norm"),
            activation=layer_config.get("activation", "SiLU"),
        )

    def forward(self, input, **kwargs):
        """Forward pass with automatic temporal dim handling.

        Accepts either a tensor or a dict keyed by observation name. When
        ``num_input_temporal_dims`` is set, flattens the last two dims before
        the network. When ``num_output_temporal_dims`` is set, reshapes the
        output to ``(*, num_output_temporal_dims, feature_per_step)``.

        Args:
            input: Tensor of shape ``(batch, input_dim)`` or
                ``(batch, temporal, feature)``, or a dict mapping observation
                names to tensors (first key in ``input_dim`` config is used).
            **kwargs: Passed through (unused by base; allows subclass compat).

        Returns:
            Output tensor of shape ``(batch, output_dim)`` or
            ``(batch, num_output_temporal_dims, feature_per_step)``.
        """
        if isinstance(input, dict):
            input_obs_key = self.module_config_dict["input_dim"][0]
            input = input[input_obs_key]
        if self.num_input_temporal_dims is not None:
            input = input.view(*input.shape[:-2], self.input_dim)
        output = self.module(input)
        if self.num_output_temporal_dims is not None:
            output = output.view(
                *output.shape[:-1],
                self.num_output_temporal_dims,
                self.output_dim // self.num_output_temporal_dims,
            )
        return output


class BaseModuleAux(BaseModule):
    """BaseModule extended with auxiliary loss computation.

    Wraps the parent forward pass so that, when requested, auxiliary loss
    functions are evaluated on the network output and returned alongside
    their coefficients. This enables the PPO aux-loss trainer to add
    regularization terms (e.g. action smoothness) without modifying the
    core module.

    When ``compute_aux_loss=True``, returns a dict with ``"action_mean"``,
    ``"aux_losses"``, and ``"aux_loss_coef"``. Otherwise returns a plain
    tensor identical to ``BaseModule.forward``.
    """

    def __init__(self, aux_loss_func={}, aux_loss_coef={}, **kwargs):
        """Initialize with auxiliary loss functions and their coefficients.

        Args:
            aux_loss_func: Mapping of loss name to either an ``nn.Module``
                instance or a Hydra config dict (instantiated via
                ``hydra.utils.instantiate``). Each function receives a dict
                with an ``"action_mean"`` key.
            aux_loss_coef: Mapping of loss name to its scalar coefficient,
                returned alongside computed losses for the trainer to weight.
            **kwargs: Forwarded to ``BaseModule.__init__``.
        """
        super().__init__(**kwargs)
        self.aux_loss_coef = aux_loss_coef

        # Instantiate loss functions
        self.aux_loss_func = nn.ModuleDict()
        for name, func_cfg in aux_loss_func.items():
            if isinstance(func_cfg, nn.Module):
                self.aux_loss_func[name] = func_cfg
            else:
                # Hydra instantiation
                from hydra.utils import instantiate

                self.aux_loss_func[name] = instantiate(func_cfg)

    def forward(self, input, compute_aux_loss=False, **kwargs):
        """Forward pass with optional auxiliary loss computation.

        Args:
            input: Input tensor or dict (same as ``BaseModule.forward``).
            compute_aux_loss: If True, evaluate all registered aux loss
                functions and return a dict instead of a plain tensor.
            **kwargs: Forwarded to ``BaseModule.forward``.

        Returns:
            When ``compute_aux_loss=False``: output tensor of shape
            ``(batch, output_dim)``.
            When ``compute_aux_loss=True``: dict with keys
            ``"action_mean"`` (tensor), ``"aux_losses"`` (dict of scalar
            tensors), and ``"aux_loss_coef"`` (dict of floats).
        """
        # Call parent forward to get output tensor
        output = super().forward(input, **kwargs)

        if not compute_aux_loss:
            return output

        # Build loss inputs dict
        loss_inputs = {"action_mean": output}

        # Compute each loss
        aux_losses = {}
        for name, func in self.aux_loss_func.items():
            aux_losses[name] = func(loss_inputs)

        return {
            "action_mean": output,
            "aux_losses": aux_losses,
            "aux_loss_coef": self.aux_loss_coef,
        }
