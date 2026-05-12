from __future__ import annotations

from loguru import logger
import numpy as np
import torch
from torch import nn

from gear_sonic.trl.utils import common


def set_fuzzy_config_params(config, candidate_keys, value):
    """Set matching config keys to a given value.

    Scans ``config`` for any key that appears in ``candidate_keys`` and
    overwrites its value.  Used to wire dynamically-computed dimensions
    (e.g. ``input_dim``, ``output_dim``) into encoder / decoder param
    dicts before instantiation.

    Args:
        config: Mutable mapping of parameter names to values.
        candidate_keys: Iterable of key names to look for in ``config``.
        value: Replacement value to assign to each matched key.

    Returns:
        The same ``config`` mapping with matching keys updated in-place.
    """
    for key in candidate_keys:
        if key in config:
            config[key] = value
    return config


class UniversalTokenModule(nn.Module):
    """SONIC-style action transform module (ATM) with FSQ token bottleneck.

    Implements the encoder → FSQ quantizer → decoder pipeline described in the
    SONIC paper.  Multiple named encoders (e.g. ``g1``, ``smpl``, ``teleop``)
    convert different flavors of tokenizer observations into a shared latent
    space.  An optional Finite Scalar Quantizer (FSQ) discretises the latent,
    producing a compact token representation.  One or more decoders then map
    the tokens (plus proprioception) back to joint-space actions.

    Token flow::

        tokenizer_obs ──► encoder(s) ──► [+ additive encoders]
                                      ──► [latent_residual (pre_quantization)]
                                      ──► FSQ quantizer
                                      ──► [+ latent_residual (post_quantization)]
                                      ──► decoder(s)
                                      ──► action_mean

    Latent residual modes allow an external HOI policy to inject corrections
    into the latent space without retraining the base ATM:

    * ``"post_quantization"`` - residual added to FSQ-quantized tokens (default).
    * ``"pre_quantization"`` - residual added *before* FSQ; the sum gets quantized.
    * ``"pre_quantization_replace"`` - latent is replaced entirely by the residual.

    Attributes:
        encoders: ``nn.ModuleDict`` of named encoder networks.
        quantizer: FSQ quantizer (``None`` when quantization is disabled).
        decoders: ``nn.ModuleDict`` of named decoder networks.
        aux_loss_func: ``nn.ModuleDict`` of auxiliary loss callables.
        token_dim: Dimensionality of a single token (equals ``num_fsq_levels``).
        max_num_tokens: Number of tokens produced per timestep.
        token_total_dim: ``token_dim * max_num_tokens`` - flat token size.
    """

    def __init__(
        self,
        env_config,
        algo_config,
        obs_dim_dict=None,
        module_dim_dict=None,
        proprioception_features=[],
        num_fsq_levels=5,
        fsq_level_list=16,
        max_num_tokens=None,
        down_t=2,
        num_future_frames=1,
        quantizer=None,
        encoders=None,
        decoders=None,
        aux_loss_func={},
        aux_loss_coef={},
        encoder_sample_probs=None,
        reencode_smpl_g1_recon=False,
        meta_action_dim=None,  # For hierarchical policies with split actions
        body_action_dim=None,  # For separate body decoder (e.g., 29 body joints)
        hand_action_dim=None,  # For separate hand decoder (e.g., 14 hand joints)
        freeze_encoders=False,
        freeze_decoders=False,
        freeze_quantizer=False,
        stiff_compliance_threshold=0.01,  # Threshold for stiff mode filtering
        optimize_encoders_ratio_for_CHIP=False,  # CHIP compliance training optimization
        active_encoders=None,  # Optional list of encoder names to activate (None = all)
        active_decoders=None,  # Optional list of decoder names to activate (None = all)
        **kwargs,  # noqa: ARG002
    ):
        """Initialise encoders, FSQ quantizer, decoders, and auxiliary losses.

        Args:
            env_config: Environment configuration object exposing
                ``obs.group_obs_dims``, ``obs.group_obs_names``, and
                ``robot.actions_dim``.
            algo_config: Algorithm configuration (stored but not directly
                accessed during construction).
            obs_dim_dict: Mapping from observation name to dimension.  Falls
                back to ``env_config.robot.algo_obs_dim_dict`` when ``None``.
            module_dim_dict: Optional extra dimension overrides for named
                intermediate features.
            proprioception_features: List of observation keys whose concatenation
                forms the proprioception input fed to every decoder.
            num_fsq_levels: Number of FSQ levels (equals ``token_dim``).
            fsq_level_list: Per-level codebook size.  An ``int`` is broadcast
                to ``[fsq_level_list] * num_fsq_levels``.
            max_num_tokens: Explicit token count per timestep.  When ``None``
                this is derived as ``max(1, num_future_frames // 2**down_t)``.
            down_t: Temporal downsampling factor for token count derivation.
            num_future_frames: Number of future motion frames in the tokenizer
                input window.
            quantizer: Hydra-instantiable config for the FSQ quantizer.
                ``None`` disables quantization (identity passthrough).
            encoders: Dict of encoder configs (Hydra DictConfig).  Each entry
                specifies ``inputs``, ``outputs``, ``params``, and optional
                ``additive_to`` / ``sub_encoders`` / ``mask`` / ``freeze``.
            decoders: Dict of decoder configs (Hydra DictConfig).  Each entry
                specifies ``inputs``, ``outputs``, ``conds``, ``params``, and
                ``has_temporal_dim``.
            aux_loss_func: Dict of auxiliary loss configs to instantiate as
                ``nn.Module`` callables (stored in ``self.aux_loss_func``).
            aux_loss_coef: Dict mapping loss name → scalar coefficient passed
                back to the trainer.
            encoder_sample_probs: Ordered dict mapping encoder name → sampling
                probability; defines the column order of ``encoder_index`` in
                tokenizer observations.
            reencode_smpl_g1_recon: When ``True``, re-encode the G1 kinematic
                decoder output to compute cycle-consistency auxiliary losses.
            meta_action_dim: Action dimensionality for hierarchical policies
                where the top-level action differs from the full joint count.
            body_action_dim: Dimensionality of body joints for split
                body / hand decoding (default 29).
            hand_action_dim: Dimensionality of hand joints for split
                body / hand decoding (default 14).
            freeze_encoders: Freeze all encoder parameters after init.
            freeze_decoders: Freeze all decoder parameters after init.
            freeze_quantizer: Freeze quantizer parameters after init.
            stiff_compliance_threshold: Absolute compliance value below which
                an environment is treated as "stiff" for auxiliary loss gating.
            optimize_encoders_ratio_for_CHIP: When ``True``, enforce one-hot
                encoder selection (CHIP training mode) instead of the default
                multi-hot selection where SMPL-native envs activate both SMPL
                and G1 encoders.
            active_encoders: Optional list of encoder names to instantiate and
                run.  All encoders are active when ``None``.
            active_decoders: Optional list of decoder names to run during
                forward.  All decoders are run when ``None``.
            **kwargs: Absorbed for forward-compatibility.
        """
        super().__init__()

        # Store freeze flags
        self.freeze_encoders = freeze_encoders
        self.freeze_decoders = freeze_decoders
        self.stiff_compliance_threshold = stiff_compliance_threshold
        self.freeze_quantizer = freeze_quantizer
        self.optimize_encoders_ratio_for_CHIP = optimize_encoders_ratio_for_CHIP

        if self.optimize_encoders_ratio_for_CHIP:
            logger.info(
                "[CHIP Mode] optimize_encoders_ratio_for_CHIP=True: "
                "Native encoder selection is now one-hot (no G1 auto-activation for SMPL). "
                "G1 latents for SMPL-native envs computed only in aux losses when compliance=0."
            )

        # Cache for last encoded tokens (for external access, e.g., by callbacks)
        # These are populated during forward() and can be read afterwards
        self._last_encoded_tokens = None  # dict[encoder_name -> Tensor]
        self._last_encoded_latents = None  # dict[encoder_name -> Tensor] (pre-quantization)

        self.env_config = env_config
        self.algo_config = algo_config
        self.encoders_cfg = encoders
        self.decoders_cfg = decoders
        self.encoder_sample_probs = encoder_sample_probs
        self.reencode_smpl_g1_recon = reencode_smpl_g1_recon
        self.tokenizer_obs_dims = self.env_config.obs.group_obs_dims["tokenizer"]
        self.tokenizer_obs_names = self.env_config.obs.group_obs_names["tokenizer"]
        self.actions_dim = self.env_config.robot.actions_dim
        self.meta_action_dim = (
            meta_action_dim  # Can be different from actions_dim for hierarchical policies
        )

        if obs_dim_dict is None:
            obs_dim_dict = getattr(env_config.robot, "algo_obs_dim_dict", {})

        if module_dim_dict is None:
            module_dim_dict = {}

        self.obs_dim_dict = obs_dim_dict
        self.module_dim_dict = module_dim_dict
        self.proprioception_features = proprioception_features
        self.num_future_frames = num_future_frames

        # Initialize auxiliary loss functions (nn.ModuleDict so sub-module buffers
        # are part of the model state_dict and visible to DDP)
        self.aux_loss_func = nn.ModuleDict()
        for name, loss_func in aux_loss_func.items():
            self.aux_loss_func[name] = common.custom_instantiate(loss_func, _resolve=False)
        self.aux_loss_coef = aux_loss_coef

        module_input_candidate_params = ["input_dim"]

        module_output_candidate_params = ["output_dim"]

        """
        Initialize quantizer
        """
        if isinstance(fsq_level_list, int):
            fsq_level_list = [fsq_level_list] * num_fsq_levels
        if quantizer is not None:
            self.quantizer = common.custom_instantiate(
                quantizer, levels=fsq_level_list, _resolve=False
            )
        else:
            self.quantizer = None
        self.num_fsq_levels = num_fsq_levels
        self.fsq_level_list = fsq_level_list

        # quantizer levels determine the embedding dimension
        self.token_dim = self.num_fsq_levels
        self.down_t = down_t  # default
        if max_num_tokens is not None:
            self.max_num_tokens = max_num_tokens
        else:
            self.max_num_tokens = max(1, self.num_future_frames // (2**self.down_t))
        self.token_total_dim = self.token_dim * self.max_num_tokens
        logger.info(
            f"Motion Encoder and Quantizer initialized with embedding dim: {self.token_total_dim} (num_tokens={self.max_num_tokens}, token_dim={self.token_dim})"  # noqa: E501
        )

        """
        Initialize motion encoders dynamically
        """
        self.encoders = nn.ModuleDict()
        self.sub_encoders = {}
        self.encoder_input_features = {}
        self.encoder_mask_features = {}
        self.encoders_to_iterate = []
        self.additive_encoders = {}  # Map: base_encoder_name -> list of additive encoder names

        # First pass: identify additive encoders
        for encoder_name, encoder_config in self.encoders_cfg.items():
            additive_to = encoder_config.get("additive_to", None)
            if additive_to is not None:
                if additive_to not in self.additive_encoders:
                    self.additive_encoders[additive_to] = []
                self.additive_encoders[additive_to].append(encoder_name)

        # New config structure - initialize all encoders from config
        for encoder_name, encoder_config in self.encoders_cfg.items():
            # Skip non-active encoders early (avoids KeyError on missing obs dims)
            if active_encoders is not None and encoder_name not in active_encoders:
                continue

            sub_encoder_only = encoder_config.get("sub_encoder_only", False)
            is_additive = encoder_config.get("additive_to", None) is not None
            # Don't iterate additive encoders independently - they're called within their base encoder
            if not sub_encoder_only and not is_additive:
                self.encoders_to_iterate.append(encoder_name)
            self.sub_encoders[encoder_name] = encoder_config.get("sub_encoders", [])

            input_features = encoder_config.get("inputs", [])
            output_features = encoder_config.get("outputs", [])
            input_feature_dim = sum([self.tokenizer_obs_dims[key][-1] for key in input_features])
            if len(output_features) > 0:
                output_feature_dim = sum(
                    [self.tokenizer_obs_dims[key][-1] for key in output_features]
                )
            else:
                output_feature_dim = self.token_dim
            self.encoder_input_features[encoder_name] = input_features
            self.encoder_mask_features[encoder_name] = list(encoder_config.get("mask", []))

            if len(self.sub_encoders[encoder_name]) > 0:
                continue

            encoder_params = encoder_config["params"].copy()
            # Set dynamic parameters
            set_fuzzy_config_params(
                encoder_params, module_input_candidate_params, input_feature_dim
            )
            set_fuzzy_config_params(
                encoder_params, module_output_candidate_params, output_feature_dim
            )

            # Instantiate encoder
            encoder = common.custom_instantiate(encoder_params, _resolve=False)
            self.encoders[encoder_name] = encoder

            # Per-encoder freeze support
            if encoder_config.get("freeze", False):
                for param in encoder.parameters():
                    param.requires_grad = False
                logger.info(f"Froze encoder: {encoder_name}")

            logger.info(f"Initialized {encoder_name} encoder with input features: {input_features}")

        # Log additive encoder relationships
        if self.additive_encoders:
            for base_encoder, additive_list in self.additive_encoders.items():
                logger.info(
                    f"Additive encoders for '{base_encoder}': {additive_list} (outputs will be summed)"
                )

        """
        Initialize motion decoders dynamically
        """
        self.decoders = nn.ModuleDict()
        self.decoder_input_features = {}
        self.decoder_output_features = {}
        self.decoder_output_feature_dims = {}
        self.decoder_cond_features = {}  # For root-disentangled decoders (external_cond)
        self.decoder_mask_features = {}

        # Support custom meta_action_dim for hierarchical policies
        meta_action_dim = getattr(self, "meta_action_dim", None) or self.actions_dim

        # Compute proprioception dim from actual features (not hardcoded actor_obs)
        proprioception_dim = sum(
            self.obs_dim_dict[key]
            for key in self.proprioception_features
            if key in self.obs_dim_dict
        ) or self.obs_dim_dict.get("actor_obs", 0)

        decoder_feature_dims_map = {
            "proprioception": proprioception_dim,
            "token": self.token_dim,
            "token_flattened": self.token_dim * self.max_num_tokens,
            "action": self.actions_dim,
            "meta_action": meta_action_dim,  # For hierarchical policies
            "hand_action": hand_action_dim or 14,  # 14 hand joints default
            "body_action": body_action_dim or 29,  # 29 body joints default
        }
        for k, v in self.tokenizer_obs_dims.items():
            if k not in decoder_feature_dims_map:
                decoder_feature_dims_map[k] = v[-1]
        self.decoder_feature_dims_map = decoder_feature_dims_map

        for decoder_name, decoder_config in self.decoders_cfg.items():
            # Skip non-active decoders early (avoids KeyError on missing obs dims)
            if active_decoders is not None and decoder_name not in active_decoders:
                continue

            decoder_params = decoder_config["params"].copy()
            input_features = decoder_config.get("inputs", [])
            output_features = decoder_config.get("outputs", [])
            cond_features = decoder_config.get("conds", [])
            has_temporal_dim = decoder_config.has_temporal_dim
            input_feature_dim = 0
            for key in input_features:
                input_feature_dim += self.decoder_feature_dims_map[key]
            output_feature_dim = 0
            self.decoder_output_feature_dims[decoder_name] = {}
            for key in output_features:
                if key in decoder_feature_dims_map:
                    feature_dim = decoder_feature_dims_map[key]
                else:
                    assert has_temporal_dim
                    feature_dim = self.tokenizer_obs_dims[key][-1]
                self.decoder_output_feature_dims[decoder_name][key] = feature_dim
                output_feature_dim += feature_dim
            set_fuzzy_config_params(
                decoder_params, module_input_candidate_params, input_feature_dim
            )
            set_fuzzy_config_params(
                decoder_params, module_output_candidate_params, output_feature_dim
            )

            # For decoders with conds: compute external_cond_dim from cond feature dimensions.
            if cond_features:
                external_cond_dim = sum(self.tokenizer_obs_dims[key][-1] for key in cond_features)
                decoder_params["external_cond_dim"] = external_cond_dim
                logger.info(
                    f"Decoder '{decoder_name}' has conds={cond_features} "
                    f"with external_cond_dim={external_cond_dim}"
                )
            self.decoder_cond_features[decoder_name] = cond_features
            self.decoder_mask_features[decoder_name] = list(decoder_config.get("mask", []))

            # Instantiate decoder
            decoder = common.custom_instantiate(decoder_params, _resolve=False)
            self.decoders[decoder_name] = decoder
            self.decoder_input_features[decoder_name] = input_features
            self.decoder_output_features[decoder_name] = output_features

            logger.info(
                f"Initialized {decoder_name} decoder with input features: {input_features} and output features: {output_features}"  # noqa: E501
            )

        # Filter active encoders/decoders if specified (for kinematic-only training)
        if active_encoders is not None:
            self.encoders_to_iterate = [e for e in self.encoders_to_iterate if e in active_encoders]
            logger.info(f"Active encoders filtered to: {self.encoders_to_iterate}")
        if active_decoders is not None:
            # Keep all decoders in ModuleDict (for checkpoint compat), but only iterate active ones
            self._active_decoders = set(active_decoders)
            logger.info(f"Active decoders filtered to: {list(self._active_decoders)}")
        else:
            self._active_decoders = None

        # Apply freeze logic
        if self.freeze_encoders:
            for encoder in self.encoders.values():
                for param in encoder.parameters():
                    param.requires_grad = False
            logger.info(f"Froze encoders: {list(self.encoders.keys())}")

        if self.freeze_decoders:
            for decoder in self.decoders.values():
                for param in decoder.parameters():
                    param.requires_grad = False
            logger.info(f"Froze decoders: {list(self.decoders.keys())}")

        if self.freeze_quantizer and self.quantizer is not None:
            for param in self.quantizer.parameters():
                param.requires_grad = False
            logger.info("Froze quantizer")

        # Variable frame support: enabled when any encoder or decoder has mask observations
        self.variable_frames_enabled = any(
            feats
            for feats in (
                *self.encoder_mask_features.values(),
                *self.decoder_mask_features.values(),
            )
        )
        if self.variable_frames_enabled:
            # Cache frame indices for mask creation (avoids allocation every forward)
            self.register_buffer(
                "_frame_indices",
                torch.arange(self.num_future_frames).unsqueeze(0),
                persistent=False,
            )

    def _create_frame_and_token_masks(self, tokenizer_obs):
        """Create frame and token masks from num_frames observation.

        Returns (frame_mask, token_mask) or (None, None) if disabled.
            frame_mask: [B*seq, max_frames] bool, True=valid
            token_mask: [B*seq, max_tokens] bool, True=valid
        """
        if "command_num_frames" not in tokenizer_obs:
            return None, None

        num_frames = tokenizer_obs["command_num_frames"]  # [B, seq, 1]
        num_frames_flat = num_frames.reshape(-1, 1)  # [B*seq, 1]
        frame_mask = self._frame_indices < num_frames_flat  # [B*seq, max_frames]

        frames_per_token = 2**self.down_t
        token_frame_mask = frame_mask.reshape(-1, self.max_num_tokens, frames_per_token)
        token_mask = token_frame_mask.all(dim=-1)  # [B*seq, max_tokens]

        return frame_mask, token_mask

    def parse_tokenizer_obs(self, input_data):
        """Split the flat tokenizer observation tensor into a named dict.

        The tokenizer observation is stored as a single concatenated vector in
        ``input_data["tokenizer"]``.  This method reshapes each slice back to
        its original multi-dimensional form using the dimension metadata from
        ``env_config``.

        Args:
            input_data: Dict containing ``"tokenizer"`` key with shape
                ``(..., total_tokenizer_dim)``.

        Returns:
            Dict mapping observation name to tensor with shape
            ``(*batch_dims, *obs_dims)`` for each registered tokenizer
            observation.
        """
        tokenizer_obs = input_data["tokenizer"]
        tokenizer_obs_dict = {}
        index = 0
        for name in self.tokenizer_obs_names:
            dims = tuple(self.tokenizer_obs_dims[name])
            all_dim = np.prod(self.tokenizer_obs_dims[name])
            tokenizer_obs_dict[name] = tokenizer_obs[..., index : index + all_dim].reshape(
                tokenizer_obs.shape[:-1] + dims
            )
            index += all_dim
        assert index == tokenizer_obs.shape[-1], f"{index=}, {tokenizer_obs.shape[-1]=}"
        return tokenizer_obs_dict

    def create_encoder_masks(self, tokenizer_obs):
        """Create encoder masks for each encoder.

        When optimize_encoders_ratio_for_CHIP=False (legacy):
            - encoder_index is multi-hot: SMPL-native envs have both SMPL=1 and G1=1
            - encoder_masks["g1"] includes G1-native AND SMPL-native envs
            - encoder_masks["g1_has_smpl"] identifies which G1-masked envs are SMPL-native

        When optimize_encoders_ratio_for_CHIP=True (CHIP mode):
            - encoder_index is one-hot: each env has exactly one native encoder
            - encoder_masks are disjoint (no overlap)
            - encoder_masks["g1_has_smpl"] is empty (G1-native envs don't have SMPL)
            - G1 latents for SMPL-native envs are computed separately in aux losses
        """
        if len(self.encoders) == 1:
            return {list(self.encoders.keys())[0]: None}

        encoder_masks = {}
        encoder_index = tokenizer_obs["encoder_index"]

        for i, encoder_name in enumerate(self.encoder_sample_probs.keys()):
            encoder_masks[encoder_name] = encoder_index[..., i].bool().flatten()

        # Re-organized creation of intersection encoder masks.
        # NOTE: No need to use compliance encoders in this version -- directly use the normal encoders

        # For encoder pairs, create combined masks reflecting shared samples.
        # The pairs and their intersection masks:
        encoder_mask_combinations = [
            # (key1, key2, intersection_mask_names)
            ("g1", "smpl", [("g1_has_smpl", "smpl", "g1")]),
            (
                "teleop",
                "smpl",
                [("teleop_has_smpl", "smpl", "teleop"), ("smpl_has_teleop", "teleop", "smpl")],
            ),
            ("g1", "teleop", [("g1_has_teleop", "teleop", "g1")]),
            ("teleop", "g1", [("teleop_has_g1", "g1", "teleop")]),
            ("g1", "soma", [("g1_has_soma", "soma", "g1")]),
        ]
        for key1, key2, mask_defs in encoder_mask_combinations:
            if key1 in encoder_masks and key2 in encoder_masks:
                for mask_name, mask_src, mask_cond in mask_defs:
                    encoder_masks[mask_name] = encoder_masks[mask_src][encoder_masks[mask_cond]]
        return encoder_masks

    def assemble_all_tokens(self, encoded_tokens, encoder_masks, batch_size, seq_len):
        """Scatter per-encoder tokens into a single batch-aligned tensor.

        Each encoder processes only the environments assigned to it (via
        ``encoder_masks``).  This method writes each encoder's output back to
        the correct rows of a zero-initialised buffer, then reshapes to
        ``(batch, seq, num_tokens, token_dim)``.

        Args:
            encoded_tokens: Dict mapping encoder name to tensor of shape
                ``(num_masked, num_tokens, token_dim)``.
            encoder_masks: Dict mapping encoder name to a boolean index tensor
                of length ``batch*seq``.  ``None`` means all samples.
            batch_size: Batch dimension ``B``.
            seq_len: Sequence dimension ``S``.

        Returns:
            Tensor of shape ``(B, S, num_tokens, token_dim)`` with each row
            filled by its assigned encoder's output.
        """
        first_token = encoded_tokens[list(encoded_tokens.keys())[0]]
        if len(encoded_tokens) == 1:
            return first_token.view(batch_size, seq_len, *first_token.shape[1:])

        all_tokens = torch.zeros(
            (batch_size * seq_len,) + first_token.shape[1:],
            dtype=first_token.dtype,
            device=first_token.device,
        )
        for encoder_name in encoded_tokens.keys():
            all_tokens[encoder_masks[encoder_name]] = encoded_tokens[encoder_name]
        all_tokens = all_tokens.view(batch_size, seq_len, *all_tokens.shape[1:])
        return all_tokens

    def _encode_single(self, encoder_name, tokenizer_obs, encoder_mask=None, frame_mask=None):
        """Encode using a single encoder without quantization or additive composition.
        This is the core encoding logic used by both regular and additive encoders.

        Args:
            encoder_name: Name of the encoder to use
            tokenizer_obs: Dictionary of tokenizer observations
            encoder_mask: Optional mask for selecting specific samples
            frame_mask: Optional [B*seq, max_frames] bool mask for variable frame support

        Returns:
            latent: The encoded latent representation
        """  # noqa: D205
        encoder = sub_encoders = None
        if encoder_name in self.encoders:
            encoder = self.encoders[encoder_name]
        else:
            sub_encoders = [
                self.encoders[sub_encoder_name]
                for sub_encoder_name in self.sub_encoders[encoder_name]
            ]

        input_features = self.encoder_input_features[encoder_name]

        obs_list = [tokenizer_obs[key] for key in input_features]
        # Only handle dimension broadcasting for the "smpl" encoder_name and for obs with last dim 9 or 12
        if "smpl" in encoder_name:
            max_ndim = max(obs.ndim for obs in obs_list)
            # Check if any obs needs broadcasting: (last dim is 9 or 12 and ndim < max_ndim)
            need_broadcast = any(
                (obs.shape[-1] in (9, 12) and obs.ndim < max_ndim) for obs in obs_list
            )
            if need_broadcast:
                aligned_obs_list = []
                # Use the first obs of max_ndim as the temporal reference
                target_obs = next(o for o in obs_list if o.ndim == max_ndim)
                temporal_dim = target_obs.shape[-2]
                for obs in obs_list:
                    if obs.shape[-1] in (9, 12) and obs.ndim < max_ndim:
                        # Add missing dimension(s)
                        while obs.ndim < max_ndim:
                            obs = obs.unsqueeze(-2)  # noqa: PLW2901
                        # Expand temporal dimension
                        expand_shape = list(obs.shape)
                        expand_shape[-2] = temporal_dim
                        obs = obs.expand(*expand_shape)  # noqa: PLW2901
                    aligned_obs_list.append(obs)
                encoder_input = torch.cat(aligned_obs_list, dim=-1)
            else:
                encoder_input = torch.cat(obs_list, dim=-1)
        else:
            encoder_input = torch.cat(obs_list, dim=-1)
        encoder_input = encoder_input.view(-1, *encoder_input.shape[2:])

        # Apply encoder_mask to both input and frame_mask
        frame_mask_enc = None
        if encoder_mask is not None:
            encoder_input = encoder_input[encoder_mask]
            if frame_mask is not None:
                frame_mask_enc = frame_mask[encoder_mask]
        elif frame_mask is not None:
            frame_mask_enc = frame_mask

        # encode using the provided encoder (with gradients enabled for end-to-end training)
        if encoder is not None:
            if frame_mask_enc is not None and self.encoder_mask_features.get(encoder_name):
                latent = encoder(encoder_input, frame_mask=frame_mask_enc)
            else:
                latent = encoder(encoder_input)
        else:
            latent = encoder_input
            for sub_encoder in sub_encoders:
                latent = sub_encoder(latent)

        return latent

    def encode(
        self, encoder_name, tokenizer_obs, encoder_mask=None, no_quantization=False, frame_mask=None
    ):
        """Encode using the specified encoder, including any additive encoders.

        Args:
            encoder_name: Name of the base encoder to use
            tokenizer_obs: Dictionary of tokenizer observations
            encoder_mask: Optional mask for selecting specific samples
            no_quantization: If True, skip quantization and return raw latent
            frame_mask: Optional [B*seq, max_frames] bool mask for variable frame support

        Returns:
            If no_quantization: latent tensor
            Otherwise: (encoded_tokens, latent) tuple
        """
        # Get base encoder latent
        latent = self._encode_single(
            encoder_name, tokenizer_obs, encoder_mask, frame_mask=frame_mask
        )

        # Add contributions from additive encoders
        if encoder_name in self.additive_encoders:
            for additive_encoder_name in self.additive_encoders[encoder_name]:
                additive_latent = self._encode_single(
                    additive_encoder_name, tokenizer_obs, encoder_mask, frame_mask=frame_mask
                )
                latent = latent + additive_latent

        if no_quantization:
            return latent

        # quantize using quantizer
        if self.quantizer is not None:
            quantized_codes, _ = self.quantizer(latent)
            encoded_tokens = quantized_codes.contiguous()
        else:
            encoded_tokens = latent
        return encoded_tokens, latent

    def decode(self, decoder_name, decode_input_dict, token_mask=None):
        """Run a single named decoder and split its output by feature.

        Concatenates the decoder's declared input features from
        ``decode_input_dict``, passes them through the decoder network
        (optionally supplying ``external_cond`` and ``token_mask``), then
        slices the output back into a per-feature dict.

        Args:
            decoder_name: Key into ``self.decoders`` and associated metadata.
            decode_input_dict: Dict containing at least the keys declared in
                the decoder's ``inputs`` and ``conds`` config entries.
                Common keys include ``"token"``, ``"token_flattened"``, and
                ``"proprioception"``.
            token_mask: Optional ``(B*seq, max_tokens)`` bool mask for
                variable-length token sequences.  Only forwarded to decoders
                that declare mask features.

        Returns:
            Dict mapping output feature name to the corresponding slice of the
            decoder output tensor, e.g.
            ``{"action": Tensor(..., action_dim)}``.
        """
        decoder = self.decoders[decoder_name]
        input_features = self.decoder_input_features[decoder_name]
        output_feature_dims = self.decoder_output_feature_dims[decoder_name]
        cond_features = self.decoder_cond_features.get(decoder_name, [])
        decoder_input = torch.cat([decode_input_dict[key] for key in input_features], dim=-1)

        # Build optional kwargs for the decoder call
        kwargs = {}
        if cond_features:
            kwargs["external_cond"] = torch.cat(
                [decode_input_dict[key] for key in cond_features], dim=-1
            )
        if token_mask is not None and self.decoder_mask_features.get(decoder_name):
            kwargs["token_mask"] = token_mask
        output = decoder(decoder_input, **kwargs)

        # parse output
        output_dict = {}
        index = 0
        for key, dim in output_feature_dims.items():
            output_dict[key] = output[..., index : index + dim]
            index += dim
        assert index == output.shape[-1], f"{index=}, {output.shape[-1]=}"

        return output_dict

    def forward(  # noqa: D417
        self,
        input_data,
        compute_aux_loss=False,
        return_dict=False,
        latent_residual=None,
        latent_residual_mode="post_quantization",
        **kwargs,  # noqa: ARG002
    ):
        """Run the full encode → quantize → decode pipeline.

        Parses tokenizer observations, routes each environment to its assigned
        encoder, optionally applies an external latent residual, then decodes
        to joint-space actions.  When ``compute_aux_loss=True``, also computes
        all registered auxiliary losses (e.g. G1-SMPL alignment, cycle
        consistency) and returns a rich result dict.

        Args:
            input_data: Dict of named observation tensors, all with leading
                shape ``(B, S, ...)``.  Must contain at minimum ``"actor_obs"``
                (for batch/seq inference) and ``"tokenizer"`` (flat tokenizer
                obs of shape ``(B, S, total_tokenizer_dim)``).
            compute_aux_loss: When ``True``, evaluate all auxiliary loss
                functions and include them in the returned dict.
            return_dict: When ``True`` and ``compute_aux_loss=False``, return
                the full output dict instead of just ``action_mean``.
            latent_residual: Optional additive correction in latent token
                space.  Shape ``(B, token_total_dim)`` where
                ``token_total_dim = max_num_tokens * token_dim``.  Allows HOI
                policies to steer motion without modifying the base ATM.
            latent_residual_mode: Controls *when* ``latent_residual`` is
                applied:

                * ``"post_quantization"`` - add after FSQ (default, residual
                  stays continuous).
                * ``"pre_quantization"`` - add before FSQ (residual gets
                  quantized together with the encoder latent).
                * ``"pre_quantization_replace"`` - replace the encoder latent
                  entirely with ``latent_residual`` before quantization.
            **kwargs: Passed through; currently unused.

        Returns:
            When ``compute_aux_loss=True`` or ``return_dict=True``: a dict
            with keys:

            * ``"action_mean"`` - joint targets, shape ``(B, S, action_dim)``.
            * ``"aux_losses"`` - dict of scalar loss tensors (empty when
              ``compute_aux_loss=False``).
            * ``"aux_loss_coef"`` - per-loss coefficient dict from config.
            * ``"decoded_outputs"`` - raw decoder output dicts keyed by
              decoder name.
            * ``"tokenizer_obs"`` - parsed tokenizer observation dict.
            * ``"encoder_masks"`` - per-encoder boolean index tensors.
            * ``"encoded_tokens"`` - post-quantization tokens per encoder.
            * ``"encoded_latents"`` - pre-quantization latents per encoder.
            * ``"encoders_cfg"`` / ``"decoders_cfg"`` - config references.

            When ``compute_aux_loss=False`` and ``return_dict=False``:
            ``action_mean`` tensor directly.
        """
        # parse tokenizer obs
        batch_size, seq_len = input_data["actor_obs"].shape[:2]
        tokenizer_obs = self.parse_tokenizer_obs(input_data)
        proprioception_input = torch.cat(
            [input_data[key] for key in self.proprioception_features], dim=-1
        )

        # Reshape residual if provided: (batch, token_total_dim) -> (batch, 1, max_num_tokens, token_dim)
        residual_reshaped = None
        if latent_residual is not None:
            residual_reshaped = latent_residual.view(
                batch_size, 1, self.max_num_tokens, self.token_dim
            )

        # Variable frame masks
        frame_mask, token_mask = self._create_frame_and_token_masks(tokenizer_obs)

        # encode motion using all available encoders
        encoder_masks = self.create_encoder_masks(tokenizer_obs)
        encoded_tokens = {}
        encoded_latents = {}
        if latent_residual is not None and latent_residual_mode in [
            "pre_quantization",
            "pre_quantization_replace",
        ]:
            # PRE-QUANTIZATION MODES: add or replace residual before FSQ
            # Reshape residual to (batch*seq, num_tokens, token_dim) for masking
            residual_flat = latent_residual.view(
                batch_size * seq_len, self.max_num_tokens, self.token_dim
            )

            for encoder_name in self.encoders_to_iterate:
                encoder_mask = encoder_masks[encoder_name]

                # Get raw latent (including additive encoders) without quantization
                latent = self.encode(
                    encoder_name,
                    tokenizer_obs,
                    encoder_mask,
                    no_quantization=True,
                    frame_mask=frame_mask,
                )
                # Apply same mask to residual before adding
                if encoder_mask is not None:
                    masked_residual = residual_flat[encoder_mask]
                else:
                    masked_residual = residual_flat

                # Apply residual before quantization (only if we have samples)
                if latent.shape[0] > 0:
                    if latent_residual_mode == "pre_quantization":
                        # Add residual to encoder latent
                        latent = latent + masked_residual
                    elif latent_residual_mode == "pre_quantization_replace":
                        # Replace encoder latent with residual (zero out encoder)
                        latent = masked_residual
                    else:
                        raise ValueError(f"Unknown latent_residual_mode: {latent_residual_mode}")

                # Now quantize
                if self.quantizer is not None:
                    quantized_codes, _ = self.quantizer(latent)
                    encoded_tokens[encoder_name] = quantized_codes.contiguous()
                else:
                    encoded_tokens[encoder_name] = latent
                encoded_latents[encoder_name] = latent
        else:
            # STANDARD MODE: encode normally
            for encoder_name in self.encoders_to_iterate:
                encoded_tokens[encoder_name], encoded_latents[encoder_name] = self.encode(
                    encoder_name,
                    tokenizer_obs,
                    encoder_masks[encoder_name],
                    frame_mask=frame_mask,
                )
        all_tokens = self.assemble_all_tokens(encoded_tokens, encoder_masks, batch_size, seq_len)

        # POST-QUANTIZATION MODE: add residual after FSQ tokens (default)
        if latent_residual is not None and latent_residual_mode == "post_quantization":
            all_tokens = all_tokens + residual_reshaped

        # Cache tokens for external access (e.g., by callbacks)
        self._last_encoded_tokens = {k: v.detach().cpu() for k, v in encoded_tokens.items()}
        self._last_encoded_latents = {k: v.detach().cpu() for k, v in encoded_latents.items()}

        # Cache flattened full latent on device for reward computation (token smoothness)
        # all_tokens is the post-quantization token that gets sent to decoder
        # Shape: (batch, seq, num_tokens, token_dim) -> (batch, seq, latent_dim)
        self._last_full_latent_flat = all_tokens.detach().view(*all_tokens.shape[:-2], -1)

        # decode action and motion
        decode_input_dict = {
            "token": all_tokens,
            "token_flattened": all_tokens.view(*all_tokens.shape[:-2], -1),
            "proprioception": proprioception_input,
        }
        decode_input_dict.update(tokenizer_obs)

        decoded_outputs = {}
        decoders_to_run = (
            self._active_decoders if self._active_decoders is not None else self.decoders.keys()
        )
        for decoder_name in decoders_to_run:
            decoded_outputs[decoder_name] = self.decode(
                decoder_name, decode_input_dict, token_mask=token_mask
            )

        # Support "body_action", "meta_action", and "action" outputs (g1_dyn may not exist in kinematic-only mode)
        if "g1_dyn" in decoded_outputs:
            g1_dyn_out = decoded_outputs["g1_dyn"]
            if "body_action" in g1_dyn_out:
                action_mean = g1_dyn_out["body_action"]
            elif "meta_action" in g1_dyn_out:
                action_mean = g1_dyn_out["meta_action"]
            else:
                action_mean = g1_dyn_out["action"]

            # Concatenate hand decoder output if present
            if "hand_dyn" in decoded_outputs and "hand_action" in decoded_outputs["hand_dyn"]:
                hand_action = decoded_outputs["hand_dyn"]["hand_action"]
                action_mean = torch.cat([action_mean, hand_action], dim=-1)
        else:
            action_mean = None

        # compute aux losses
        if compute_aux_loss:
            aux_losses = {}

            # Initialize all paired latents
            reencoded_smpl_g1_latents = None
            paired_g1_smpl_latents = None
            paired_compliance_latents = None
            original_g1_latents_for_reencode = None

            # Determine encoder names
            smpl_encoder_name = "smpl" if "smpl" in self.encoders_to_iterate else None
            teleop_encoder_name = "teleop" if "teleop" in self.encoders_to_iterate else None

            # =========================================================================
            # STIFF-MODE OPTIMIZATION: Check for stiff samples FIRST before computing
            # expensive G1 latents. G1-SMPL loss and cycle consistency loss ONLY apply
            # in stiff mode (compliance ≈ 0). Skip computation if all samples are compliant.
            # =========================================================================
            has_stiff_samples = False
            compliance_values = None
            smpl_latents = None
            smpl_mask = None

            if smpl_encoder_name is not None and "g1" in encoded_latents:
                smpl_mask = encoder_masks.get(smpl_encoder_name)

                if smpl_mask is not None and smpl_mask.sum() > 0:
                    smpl_latents = encoded_latents[smpl_encoder_name]

                    # Extract compliance values for these envs FIRST
                    if "compliance" in tokenizer_obs:
                        compliance_flat = tokenizer_obs["compliance"].view(
                            -1, tokenizer_obs["compliance"].shape[-1]
                        )
                        compliance_values = compliance_flat[smpl_mask]

                        # Check if ANY samples are stiff
                        is_stiff = (compliance_values.abs() < self.stiff_compliance_threshold).all(
                            dim=-1
                        )
                        has_stiff_samples = is_stiff.any().item()
                    else:
                        # No compliance info means all samples are "stiff" (default behavior)
                        has_stiff_samples = True

            # Only compute stiff-mode losses if there are stiff samples
            if has_stiff_samples and smpl_mask is not None:
                # Compute G1 latents for the same envs as SMPL (for proper pairing)
                # OPTIMIZATION: Detach here since all downstream losses use g1 as
                # fixed target (detach_g1_target=True). This avoids redundant gradient
                # computation through the g1 encoder for these latents.
                g1_latents_for_smpl = self.encode(
                    encoder_name="g1",
                    tokenizer_obs=tokenizer_obs,
                    encoder_mask=smpl_mask,
                    no_quantization=True,
                ).detach()

                paired_g1_smpl_latents = {
                    "g1": g1_latents_for_smpl,
                    "smpl": smpl_latents,
                    "compliance": compliance_values,
                }

                # Store original G1 latents for cycle consistency loss (already detached)
                original_g1_latents_for_reencode = g1_latents_for_smpl

                # Cycle consistency: re-encode decoded G1 motion (only if stiff samples exist)
                if self.reencode_smpl_g1_recon:
                    reencoded_smpl_g1_latents = self.encode(
                        encoder_name="g1",
                        tokenizer_obs=decoded_outputs["g1_kin"],
                        encoder_mask=smpl_mask,
                        no_quantization=True,
                    )

            # Compute paired teleop-SMPL latents for TeleopSmplComplianceLatentLoss
            # NOTE: This applies to ALL compliance values (not just stiff), so no stiff check
            if (
                teleop_encoder_name is not None
                and smpl_latents is not None
                and smpl_mask is not None
            ):
                # Compute teleop latents for the same envs as SMPL
                # OPTIMIZATION: Detach here since TeleopSmplComplianceLatentLoss uses
                # teleop as teacher (detach_teleop_target=True). Only SMPL learns.
                teleop_latents_for_smpl = self.encode(
                    encoder_name=teleop_encoder_name,
                    tokenizer_obs=tokenizer_obs,
                    encoder_mask=smpl_mask,
                    no_quantization=True,
                ).detach()

                paired_compliance_latents = {
                    "teleop": teleop_latents_for_smpl,
                    "smpl": smpl_latents,
                }

            loss_inputs = {
                "input_data": input_data,
                "tokenizer_obs": tokenizer_obs,
                "encoded_tokens": encoded_tokens,
                "encoded_latents": encoded_latents,
                "encoder_masks": encoder_masks,
                "decoded_outputs": decoded_outputs,
                "action_mean": action_mean,
                "encoders_cfg": self.encoders_cfg,
                "frame_mask": (
                    frame_mask.view(batch_size, seq_len, -1) if frame_mask is not None else None
                ),
                "token_mask": (
                    token_mask.view(batch_size, seq_len, -1) if token_mask is not None else None
                ),
                "decoders_cfg": self.decoders_cfg,
                "reencoded_smpl_g1_latents": reencoded_smpl_g1_latents,
                # New fields for compliance-aware losses
                "paired_g1_smpl_latents": paired_g1_smpl_latents,
                "paired_compliance_latents": paired_compliance_latents,
                "original_g1_latents_for_reencode": original_g1_latents_for_reencode,
            }

            for loss_name, loss_func in self.aux_loss_func.items():
                aux_losses[loss_name] = loss_func(loss_inputs)

            output = {
                "action_mean": action_mean,
                "aux_losses": aux_losses,
                "aux_loss_coef": self.aux_loss_coef,
                # Exposed for recon trainer (existing RL callers ignore these keys)
                "decoded_outputs": decoded_outputs,
                "tokenizer_obs": tokenizer_obs,
                "encoder_masks": encoder_masks,
                "encoded_tokens": encoded_tokens,
                "encoded_latents": encoded_latents,
                "encoders_cfg": self.encoders_cfg,
                "decoders_cfg": self.decoders_cfg,
            }
        elif return_dict:
            output = {
                "action_mean": action_mean,
                "aux_losses": {},
                "aux_loss_coef": self.aux_loss_coef,
                "decoded_outputs": decoded_outputs,
                "tokenizer_obs": tokenizer_obs,
                "encoder_masks": encoder_masks,
                "encoded_tokens": encoded_tokens,
                "encoded_latents": encoded_latents,
                "encoders_cfg": self.encoders_cfg,
                "decoders_cfg": self.decoders_cfg,
            }
        else:
            output = action_mean
        return output

    def get_token_info(self):
        """Return a summary of the FSQ token configuration.

        Returns:
            Dict with keys:

            * ``"token_dim"`` - dimensionality of one token.
            * ``"total_dim"`` - ``token_dim * max_num_tokens``.
            * ``"num_levels"`` - number of FSQ levels.
            * ``"level_list"`` - per-level codebook sizes, or ``None`` if the
              quantizer does not expose a ``levels`` attribute.
            * ``"model_available"`` - always ``True``.
        """
        return {
            "token_dim": self.token_dim,
            "total_dim": self.token_total_dim,
            "num_levels": self.num_fsq_levels,
            "level_list": (
                list(self.quantizer.levels) if hasattr(self.quantizer, "levels") else None
            ),
            "model_available": True,
        }

    def forward_with_external_tokens(  # noqa: D417
        self, input_data: dict, external_tokens: torch.Tensor, **kwargs  # noqa: ARG002
    ) -> torch.Tensor:
        """Forward pass with externally provided tokens (bypasses encoder).

        This is used when an external model (e.g., kinematic diffusion) provides
        pre-computed FSQ tokens, allowing the encoder to be bypassed while still
        using the decoder for action generation.

        Args:
            input_data: Dict with 'actor_obs' for proprioception
            external_tokens: Pre-computed FSQ tokens from kinematic diffusion
                            Shape: (B, 2, 32) or (B, seq, 2, 32)

        Returns:
            action_mean: (B, action_dim) or (B, seq, action_dim)
        """
        # Get proprioception input
        proprioception_input = torch.cat(
            [input_data[key] for key in self.proprioception_features], dim=-1
        )

        # Handle different input shapes
        if external_tokens.dim() == 3:
            # (B, 2, 32) -> add seq dim -> (B, 1, 2, 32)
            external_tokens = external_tokens.unsqueeze(1)

        # external_tokens: (B, seq, 2, 32) or (B, seq, num_tokens, token_dim)
        batch_size = external_tokens.shape[0]
        seq_len = external_tokens.shape[1]

        # Flatten tokens for decoder: (B, seq, 2, 32) -> (B, seq, 64)
        token_flattened = external_tokens.view(batch_size, seq_len, -1)

        # Build decode input dict
        decode_input_dict = {
            "token": external_tokens,
            "token_flattened": token_flattened,
            "proprioception": proprioception_input,
        }

        # Decode actions using g1_dyn decoder
        g1_dyn_out = self.decode("g1_dyn", decode_input_dict)
        if "body_action" in g1_dyn_out:
            action_mean = g1_dyn_out["body_action"]
        elif "meta_action" in g1_dyn_out:
            action_mean = g1_dyn_out["meta_action"]
        else:
            action_mean = g1_dyn_out["action"]

        # Concatenate hand decoder output if present
        if "hand_dyn" in self.decoders:
            hand_dyn_out = self.decode("hand_dyn", decode_input_dict)
            if "hand_action" in hand_dyn_out:
                action_mean = torch.cat([action_mean, hand_dyn_out["hand_action"]], dim=-1)

        # Squeeze seq dim if it was added
        if action_mean.shape[1] == 1:
            action_mean = action_mean.squeeze(1)

        return action_mean

    def get_example_input(self, encoder_name, batch_size=1, device="cpu"):
        """Generate example input for ONNX export with specific encoder.

        Args:
            encoder_name: Name of the encoder to use
            batch_size: Batch size for the example input
            device: Device to create tensors on

        Returns:
            Tensor with shape (batch_size, feature_dim) including encoder features and proprioception
        """
        if encoder_name not in self.encoders:
            raise ValueError(
                f"Encoder '{encoder_name}' not found. Available encoders: {list(self.encoders.keys())}"
            )

        # Calculate total input feature dimension for the specified encoder
        encoder_input_features = self.encoder_input_features[encoder_name]
        total_feature_dim = 0

        for feature_name in encoder_input_features:
            if feature_name in self.tokenizer_obs_dims:
                feature_dims = self.tokenizer_obs_dims[feature_name]
                total_feature_dim += torch.prod(torch.tensor(feature_dims)).item()

        # Add proprioception dimension
        proprioception_dim = self.obs_dim_dict.get("actor_obs", 0)
        total_feature_dim += proprioception_dim

        # Create a single 2D tensor (B, F) including both encoder features and proprioception
        example_input = torch.randn(batch_size, total_feature_dim, device=device)

        return example_input

    def get_all_example_input(self, batch_size=1, device="cpu"):
        """Generate example inputs covering all tokenizer observations.

        Constructs random tensors whose shapes match the full tokenizer
        observation space plus proprioception.  Intended for ONNX tracing
        or shape debugging.

        Args:
            batch_size: Number of examples in the batch dimension.
            device: Target device for the returned tensors (e.g. ``"cpu"``
                or ``"cuda"``).

        Returns:
            Tuple of:

            * ``example_input`` - flat tensor of shape
              ``(batch_size, total_feature_dim)`` where
              ``total_feature_dim = tokenizer_feature_dim + proprioception_dim``.
            * ``example_dict`` - dict with keys ``"tokenizer"`` and
              ``"proprioception"``, each a random tensor with the
              corresponding shape.
        """
        # Calculate total input feature dimension for the specified encoder
        total_feature_dim = 0
        for feature_name in self.tokenizer_obs_names:
            feature_dims = self.tokenizer_obs_dims[feature_name]
            print(  # noqa: T201
                f"{feature_name}: {feature_dims}, start: {total_feature_dim} end: {total_feature_dim + torch.prod(torch.tensor(feature_dims)).item()}"  # noqa: E501
            )
            total_feature_dim += torch.prod(torch.tensor(feature_dims)).item()
        tokenizer_feature_dim = total_feature_dim

        # Add proprioception dimension
        proprioception_dim = self.obs_dim_dict.get("actor_obs", 0)
        total_feature_dim += proprioception_dim

        print(f"tokenizer_feature_dim: {tokenizer_feature_dim}")  # noqa: T201
        print(f"proprioception_dim: {proprioception_dim}")  # noqa: T201

        # Create a single 2D tensor (B, F) including both encoder features and proprioception
        example_input = torch.randn(batch_size, total_feature_dim, device=device)

        example_dict = {
            "tokenizer": torch.randn(batch_size, tokenizer_feature_dim, device=device),
            "proprioception": torch.randn(batch_size, proprioception_dim, device=device),
        }

        return example_input, example_dict
