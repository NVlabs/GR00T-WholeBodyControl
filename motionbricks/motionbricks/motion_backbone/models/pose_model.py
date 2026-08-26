from motionbricks.vqvae.neural_modules import vqvae
from motionbricks.motion_backbone.neural_modules.pose_backbone import pose_backbone_network
import torch as t
import os
import logging
from typing import Callable, Optional, Union, Dict

import torch
from pytorch_lightning import LightningModule
from motionbricks.motionlib.core.motion_reps import MotionRepBase
import numpy as np
from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints, LocalRootGlobalJoints
from motionbricks.helper.data_training_util import sample_motion_segments_from_motion_clips
from motionbricks.helper.data_training_util import sample_keyframes, extract_feature_from_motion_rep

log = logging.getLogger(__name__)


class MotionModel(LightningModule):
    """ @brief: Pose model that predicts discrete pose tokens given root motion and optional pose constraints.

    @input local_root_values: [batch, numFrames, root_dim]             # required
    @input pose_cond: [batch, numConstrainFrames=[0, 10], featdim]     # optional
    @input has_pose_cond: [batch, numConstrainFrames=[0, 10]] (int)    # required; where poses are provided
    @input num_tokens: [batch, 1]                                      # required
    @input pose_tokens: [batch, numTokens, num_pose_heads]             # required; could be all masked
    @input text_embedding: [batch, numTokens, text_embedding_dim]      # optional

    @output pose_logits: [batch, numTokens, num_pose_heads, num_codes]
    """

    def __init__(self,
                 pose_vqvae_network: vqvae.VQVAE,
                 root_vqvae_network: Union[None, vqvae.VQVAE],
                 backbone_network: pose_backbone_network,
                 motion_rep: MotionRepBase,
                 optimizer: Callable[[list], torch.optim.Optimizer] = None,
                 scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 args: Dict = None,
                 # the other key args here: most of them for compatibility only and won't be used at all
                 **kwargs):
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler

        self._supporting_networks = {'pose_net': pose_vqvae_network, 'root_net': root_vqvae_network}
        self.backbone_net = backbone_network

        self.motion_rep: GlobalRootGlobalJoints = motion_rep
        self.global_motion_rep: GlobalRootGlobalJoints = motion_rep.dual_rep.global_motion_rep
        self.local_motion_rep: LocalRootGlobalJoints = motion_rep.dual_rep.local_motion_rep

        self.DEFAULT_NUM_JOINTS = self.motion_rep.num_joints
        self._args = args
        self.one_logger_callback = kwargs.get("one_logger_callback", None)
        self.callbacks = []

        self._load_vqvae_models()

        if device is not None:
            self.pose_net = self._supporting_networks['pose_net'].to(device)
            self.root_net = self._supporting_networks['root_net'].to(device) \
                if root_vqvae_network is not None else None
            self.backbone_net = self.backbone_net.to(device)

        self._supporting_networks['pose_net'].requires_grad_ = False
        self._supporting_networks['pose_net'] = self._supporting_networks['pose_net'].eval()
        if self._supporting_networks['root_net'] is not None:
            self._supporting_networks['root_net'].requires_grad_ = False
            self._supporting_networks['root_net'] = self._supporting_networks['root_net'].eval()

    def _load_vqvae_models(self):
        """ @brief: load the vqvae models and init the backbone's input embedding.
        """
        self._vqvae_model_loaded = False
        vqvae_model_ckpt_path = self._args.vqvae_model_ckpt_path
        if os.path.exists(vqvae_model_ckpt_path):
            vqvae_model_weights = t.load(vqvae_model_ckpt_path)['state_dict']
            with t.no_grad():
                for sub_network in ['pose_net', 'root_net']:
                    if self._supporting_networks[sub_network] is None:
                        continue
                    for key, val in self._supporting_networks[sub_network].state_dict().items():
                        src = vqvae_model_weights[sub_network + '.' + key]
                        if val.shape != src.shape:
                            src = src.reshape(val.shape)
                        val.copy_(src)
                self._vqvae_model_loaded = True

            if not self.backbone_net.initted:  # the backbone's input embeddings are not initted; init with vqvae weights
                pose_codebook = self._supporting_networks['pose_net'].get_codebook()
                root_codebook = self._supporting_networks['root_net'].get_codebook() \
                    if self._supporting_networks['root_net'] is not None else None
                self.backbone_net.init_embedding_from_codebooks(pose_codebook, root_codebook)
        else:
            print(f"No VQVAE model checkpoint path available; Assuming the vqvae weights are intergrated in the model")

    def configure_optimizers(self):
        if self.one_logger_callback is not None:
            self.one_logger_callback.on_optimizer_init_start()
        optimizer = self.optimizer(self.parameters())
        if self.one_logger_callback is not None:
            self.one_logger_callback.on_optimizer_init_end()
        if not self.scheduler:
            return optimizer

        lt_kwargs = dict(self.scheduler.keywords.pop("lt_kwargs", {}))
        lt_kwargs["scheduler"] = self.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lt_kwargs}

    def set_callbacks(self, callbacks: list):
        self.callbacks = callbacks

    def configure_callbacks(self):
        return self.callbacks

    def inference_step(self, batch, batch_idx, requires_grad=False, meta_info: Dict = {}):
        if not hasattr(self, "_printed_inference_warning"):
            self._printed_inference_warning = True
            print("Warning: Pose model does not have an explicit inference step. Reusing training step.")
        with t.no_grad():
            return self.training_step(batch, batch_idx, use_outside_training=True, meta_info=meta_info)

    def training_step(self, batch, batch_idx, use_outside_training: bool = False, meta_info: Dict = {}):
        """ @brief: the training step takes input a normalized motion representation.
            batch.keys() -> dict_keys(['motion', 'motion_len', 'motion_pad_mask', 'batch_size'])
        """
        assert self.training or use_outside_training, \
            "It's possible to use training_step on evaluation set. But otherwise self.training should be true."

        # step 0: data preparations for model training; generate both global and local motions that could be used later
        batch_size, device = batch['batch_size'], batch['motion'].device
        raw_global_motions, motion_lengths, _ = \
            batch.pop('motion'), batch.pop('motion_len'), batch.pop('motion_pad_mask')
        if self.backbone_net.ACCEPT_TEXT_EMB_INPUT:
            _, _, _ = batch.pop('text'), batch.pop('text_len'), batch.pop('text_pad_mask')
        augmented_batch_size = int(batch_size * self._args['batchsize_mul_factor'])

        num_token_position = np.random.choice(np.arange(self._args['min_tokens'], self._args['max_tokens'] + 1))
        batch['num_tokens'] = t.full([augmented_batch_size, 1], num_token_position).to(device)
        num_frames = num_token_position * self.backbone_net.get_num_frames_per_token()

        valid_samples_id = (motion_lengths >= num_frames + 1)  # 1 additional frame for global-local convertion
        num_invalid_samples = batch_size - valid_samples_id.sum()
        if num_invalid_samples > batch_size // 2:
            return None  # don't have enough valid data samples, skipping this batch

        sample_info = {}
        global_motions = sample_motion_segments_from_motion_clips(raw_global_motions, motion_lengths,
                                                                  num_frames,
                                                                  self.args['batchsize_mul_factor'], info=sample_info,
                                                                  motion_rep=self.global_motion_rep)
        actual_batch_size = int(batch_size * self.args['batchsize_mul_factor'])
        batch['text_embeddings'] = None if (not self.backbone_net.ACCEPT_TEXT_EMB_INPUT) else \
            batch.pop('text_feat')[sample_info['chosen_ids']].view([augmented_batch_size, -1])

        # step 3: prepares the global & local input motions to the pose and root vqvae models
        first_frame_heading_angle = t.rand(actual_batch_size).to(device) * np.pi * 2.0 \
            if not self.motion_rep.compute_kwargs['removing_heading'] else 0.0
        global_motions = self.global_motion_rep.change_first_heading(
            global_motions, first_frame_heading_angle, is_normalized=True, to_normalize=True
        )  # note this `change_first_heading` also moves the first frame to the origin
        local_motions = self.motion_rep.dual_rep.global_to_local(
            global_motions, is_normalized=True, to_normalize=True,
            lengths=t.full([actual_batch_size], global_motions.shape[1]).to(device)
        )
        local_motions, global_motions = \
            local_motions[:, :num_frames, :], global_motions[:, :num_frames, :]  # drop last velocity padding frame
        batch['local_motions'], batch['global_motions'] = local_motions, global_motions

        # step 1: generate the pose_tokens; note we also mask / perturb the code indices in training
        # the shape of @input_tokens is [batch, numTokens, num_pose_heads]
        if 'groundtruth_pose_tokens' not in batch:
            self.move_supporting_nets_to_device(device)
            with t.no_grad():
                assert self._supporting_networks['pose_net'].motion_rep.name in ['local', 'global']
                pose_net_input = batch['local_motions'] \
                    if self._supporting_networks['pose_net'].motion_rep.name == 'local' else batch['global_motions']
                pose_tokens = \
                    self._supporting_networks['pose_net'].encode_into_idx(pose_net_input, fetch_overall_indices=False)
                batch['groundtruth_pose_tokens'] = pose_tokens

        num_pose_heads, _ = self.backbone_net.get_num_heads()
        batch['groundtruth_pose_tokens'] = \
            batch['groundtruth_pose_tokens'].view([augmented_batch_size, num_token_position, num_pose_heads])
        batch['focused_token_mask'], batch['masked_token_mask'], batch['incorrect_token_mask'], \
            batch['correct_token_mask'], batch['mask_percentage'], batch['incorrect_percentage'] = \
            self._get_token_masks(batch)
        batch['input_tokens'] = self._generate_tokens_for_training(batch)

        # step 2: the local_root_values at each frame location, has the shape [batch, numFrames, 5]
        if self._args['cond_root_feature_is_from_motion_rep'] == 'local':
            batch['local_root_values'] = extract_feature_from_motion_rep(batch['local_motions'], self.local_motion_rep,
                                                                        self._args['cond_root_feature'])
        else:
            assert self._args['cond_root_feature_is_from_motion_rep'] == 'global'
            batch['local_root_values'] = extract_feature_from_motion_rep(batch['global_motions'], self.global_motion_rep,
                                                                        self._args['cond_root_feature'])

        # step 3: the pose cond and whether a pose condition is provided. NOTE: we are using the dense cond during
        # training since it's easier to construct. In inference you could also provide the sparse condition which be
        # automatically processed in the @net.forward.
        # The dense pose_cond shape: [batch, numframes, featdim], and has_poses_cond shape: [batch, numframes] (bool)
        batch['pose_cond'], batch['has_pose_cond'], batch['text_embeddings'],  batch['has_text_embeddings'] = \
            self._sample_the_local_pose_conditions(batch['local_motions'], num_frames, batch['text_embeddings'])

        # step 4: core network forward and loss calculation
        model_outputs = self.backbone_net(batch['input_tokens'], batch['local_root_values'],
                                   batch['pose_cond'], batch['has_pose_cond'], batch['num_tokens'],
                                   batch['text_embeddings'], batch['has_text_embeddings'])

        losses = self.loss(batch, model_outputs)

        if use_outside_training:  # use in an evaluation process
            meta_info['losses'] = losses
            return None

        for key, val in losses.items():
            self.log(f"loss/train_{key}", val, on_step=True,
                     on_epoch=True, sync_dist=True, batch_size=batch["batch_size"])
        return losses['loss']

    def loss(self, batch: Dict, model_output_batch: Dict):
        """ @brief: calculate the cross-entropy loss between the predicted pose tokens and the groundtruth.
        """
        batch_size, num_positions = \
            batch['groundtruth_pose_tokens'].shape[0], model_output_batch['pose_logits'].shape[1]
        pose_logits = model_output_batch['pose_logits']
        groundtruth_tokens = batch['groundtruth_pose_tokens']

        # only consider the tokens that is focused and not masked
        IGNORE_TOKEN_IDS = -100
        target_tokens = t.where(batch['focused_token_mask'].view([batch_size, num_positions, -1]),
                                groundtruth_tokens, IGNORE_TOKEN_IDS)

        losses = {}
        pose_loss = t.nn.functional.cross_entropy(pose_logits.reshape([-1, pose_logits.shape[-1]]),
                                                  target_tokens.reshape([-1]).long(), ignore_index=IGNORE_TOKEN_IDS)

        losses['pose_loss'] = pose_loss
        losses['loss'] = pose_loss

        return losses

    @property
    def args(self):
        return self._args

    def _get_token_masks(self, batch: dict):
        """ @brief: this is the function which generate the token masks during training. It considers both the
            perturbation masks where the tokens are randomly flipped to a different token, as well the mask for the
            tokens that are replaced with [MASK] token for the network to predict.

            `focused_token_mask`: indicates all the tokens that is either perturbed, masked, or actually are the correct
                ones. This is generated with cosine scheduling.
        """
        pose_tokens = batch['groundtruth_pose_tokens']
        batch_size, device = pose_tokens.shape[0], pose_tokens.device
        num_pose_heads, _ = self.backbone_net.get_num_heads()
        num_token_positions = pose_tokens.shape[1]

        # step 1: calculate the number of tokens for each focused/mask/incorrect/correct types.
        # focus = masked + correct + incorrect; The non-focused tokens are always assumed to be known correct tokens.
        # The % of focused tokens are generated with the cosine schedule here so that more tokens are sampled.
        focus_mask_probs = t.cos(t.pi * 0.5 * (t.zeros([batch_size, 1], device=device).float().uniform_(0, 1)))
        incorrect_probs = t.zeros([batch_size, 1], device=device).uniform_(self._args['incorrect_token_ratio_min'],
                                                                           self._args['incorrect_token_ratio_max'])
        num_all_tokens_per_sample = num_pose_heads * num_token_positions
        num_focus_tokens = (num_all_tokens_per_sample * focus_mask_probs).int()   # [batch, 1]
        num_masked_tokens = t.floor(num_focus_tokens * self._args['masked_token_ratio']).int()
        num_incorrect_tokens = t.floor((num_focus_tokens - num_masked_tokens) * incorrect_probs).int()
        # num_correct_tokens = num_focus_tokens - num_masked_tokens - num_incorrect_tokens

        # step 2: sampling and generate the token masks
        random_order = t.rand((batch_size, num_all_tokens_per_sample), device=device).argsort(dim=-1)
        focused_token_mask = random_order < num_focus_tokens
        masked_token_mask = random_order < num_masked_tokens
        incorrect_token_mask = t.logical_and(random_order >= num_masked_tokens,
                                             random_order < num_masked_tokens + num_incorrect_tokens)
        correct_token_mask = t.logical_and(random_order >= num_masked_tokens + num_incorrect_tokens,
                                           random_order < num_focus_tokens)

        # step 4: the mask / incorrect level
        mask_percentage = masked_token_mask.sum(dim=-1, keepdims=True) / num_all_tokens_per_sample
        incorrect_percentage = incorrect_token_mask.sum(dim=-1, keepdims=True) / num_all_tokens_per_sample

        return focused_token_mask, masked_token_mask, incorrect_token_mask, \
            correct_token_mask, mask_percentage, incorrect_percentage

    def _generate_tokens_for_training(self, batch: Dict):
        """ @brief: based on the mask and the number of steps, change the tokens accordingly.
        """
        batch_size, num_positions_in_a_sample = \
            batch['local_motions'].shape[0], batch['groundtruth_pose_tokens'].shape[1]

        # incorrect tokens
        input_tokens = batch['groundtruth_pose_tokens'].clone()
        num_codes_pose_vqvae, _ = self.backbone_net.get_num_codes(include_aug_tokens=False)
        random_pose_tokens = t.randint_like(input_tokens, high=num_codes_pose_vqvae, low=0)
        input_tokens = t.where(batch['incorrect_token_mask'].view([batch_size, num_positions_in_a_sample, -1]),
                               random_pose_tokens, input_tokens)

        # masked tokens
        mask_tokens = t.ones_like(input_tokens) * self.backbone_net.POSE_MASK_ID  # the mask ids for the pose
        input_tokens = t.where(batch['masked_token_mask'].view([batch_size, num_positions_in_a_sample, -1]),
                               mask_tokens, input_tokens)

        return input_tokens

    @property
    def supporting_nets(self):
        return self._supporting_networks

    def move_supporting_nets_to_device(self, device):
        assert self._supporting_networks['root_net'] is None, "The root net is not supported in this version."
        if next(self._supporting_networks['pose_net'].named_parameters())[1].device != device:
            self._supporting_networks['pose_net'] = self._supporting_networks['pose_net'].to(device)
        if self._supporting_networks['pose_net'].training:  # just make sure they are in eval mode
            self._supporting_networks['pose_net'] = self._supporting_networks['pose_net'].eval()

    def _construct_keyframe_prob(self, max_num_keyframes: int = None, no_keyframe_prob: float = 0.0):
        """ @brief: generate the prob to sample each keyframe during training.
        """
        scheduled_max_num_keyframes = max_num_keyframes * \
            (self.trainer.global_step / self.args['keyframe_num_warmup_steps'])
        scheduled_max_num_keyframes = int(max(1, min(scheduled_max_num_keyframes, max_num_keyframes)))

        # construct the probability now; p_keyframe is the same for all keyframes=[1, max_num_keyframes]
        # and p_keyframe=0 is the no-keyframe probability
        prob_num_keyframes = [1.0 if i > 0 and i <= scheduled_max_num_keyframes else 0.0
                              for i in range(max_num_keyframes + 1)]
        prob_no_keyframe = no_keyframe_prob
        prob_num_keyframes[0] = sum(prob_num_keyframes) / (1 - prob_no_keyframe) * prob_no_keyframe
        prob_num_keyframes = np.array(prob_num_keyframes)
        prob_num_keyframes /= prob_num_keyframes.sum()
        return prob_num_keyframes

    def _sample_the_local_pose_conditions(self, local_motions, num_frames: int, text_embedding: t.Tensor = None):
        """ @brief: generate the local pose cond during training. Note the sampling for starting frames, ending frames
        and middle frames are done separately.
        """
        NUM_START_FRAMES = NUM_END_FRAMES = self.backbone_net.get_num_frames_per_token()
        NUM_MIDDLE_FRAMES = num_frames - NUM_START_FRAMES - NUM_END_FRAMES
        assert NUM_MIDDLE_FRAMES > 0, "The number of frames should be larger than the number of start and end frames."

        local_pose = \
            extract_feature_from_motion_rep(local_motions, self.local_motion_rep, self._args['local_pose_feature'])
        local_pose_component = {'start': local_pose[:, :NUM_START_FRAMES],
                                'end': local_pose[:, -NUM_END_FRAMES:],
                                'middle': local_pose[:, NUM_START_FRAMES:-NUM_END_FRAMES]}

        component_has_pose_cond, component_pose_cond = {}, {}
        for component in ['start', 'end', 'middle']:
            prob_num_keyframes = self._construct_keyframe_prob(
                self._args[f'max_num_{component}_keyframes'],
                no_keyframe_prob=self._args[f'no_{component}_keyframe_prob']
            )
            component_has_pose_cond[component], component_pose_cond[component] = \
                sample_keyframes(local_pose_component[component],
                                 self._args[f'max_num_{component}_keyframes'], prob_num_keyframes)
        has_pose_cond = t.cat([component_has_pose_cond['start'],
                               component_has_pose_cond['middle'], component_has_pose_cond['end']], dim=1)
        pose_cond = t.cat([component_pose_cond['start'],
                           component_pose_cond['middle'], component_pose_cond['end']], dim=1)

        if text_embedding is not None:
            assert self.backbone_net.ACCEPT_TEXT_EMB_INPUT, "The model does not accept text embedding as input."
            has_text_emb = t.rand([text_embedding.shape[0], 1], device=text_embedding.device) < \
                self._args.get('prob_provide_text_emb', 0.2)
            # provide text emb if end keyframes are not there
            # has_text_emb = t.logical_or(has_text_emb, component_has_pose_cond['end'].sum(dim=1, keepdims=True) < 1)
            text_embedding = t.where(has_text_emb, text_embedding, t.zeros_like(text_embedding))
        else:
            text_embedding = has_text_emb = None

        return pose_cond, has_pose_cond, text_embedding, has_text_emb # dense condition and the has_target

    @property
    def vqvae_model_loaded(self):
        return self._vqvae_model_loaded
