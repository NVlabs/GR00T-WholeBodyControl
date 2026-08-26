import torch as t
import torch
from torch import nn
from typing import Dict, Optional
import numpy as np
from motionbricks.motion_backbone.neural_modules.position_embedding import PositionEmbedding
from motionbricks.motionlib.core.motion_reps import MotionRepBase
from motionbricks.motion_backbone.neural_modules.mlp import FCBlock as mlp
from motionbricks.helper.data_training_util import convert_sparse_cond_to_dense_cond_if_needed
from functools import cached_property
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

class pose_backbone_network(nn.Module):
    def __init__(self, motion_rep: MotionRepBase, args: Dict = None):
        """ @brief: pose backbone network for motion generation.
        """
        super().__init__()
        self.motion_rep = motion_rep
        self.global_motion_rep = motion_rep.dual_rep.global_motion_rep
        self.local_motion_rep = motion_rep.dual_rep.local_motion_rep

        self._args = args
        self._build_transformer_backbone()
        self._build_input_embeddings_projections()
        self.register_buffer('initted', torch.Tensor([False]).to(dtype=t.bool))  # if need to initialize the codebooks

    def init_embedding_from_codebooks(self, pose_codebook: t.Tensor, root_codebook: t.Tensor):
        # make sure the codebook has the expected shapes
        if not self._args['pose_vqvae'].get('has_codebook', True):
            self.initted[:] = True
            print("No codebook for pose vqvae, skipping initialization.")
        else:
            assert self._args.pose_vqvae.num_heads == pose_codebook.shape[0] and \
                self.get_num_codes(include_aug_tokens=False)[0] == pose_codebook.shape[1] and \
                self._args.pose_vqvae.code_dim == pose_codebook.shape[0] * pose_codebook.shape[2], \
                'Pose codebook shape mismatch.'
            pose_token_dim = pose_codebook.shape[2]

            # init the input pose codebook
            pose_codebook = t.cat([pose_codebook, pose_codebook.mean(dim=1, keepdim=True)], dim=1)  # add [mask] token
            pose_codebook = pose_codebook.reshape([-1, pose_token_dim])  # change dim into [numHeads * num_codes, dim]
            with t.no_grad():
                self._pose_token_emb.weight[:] = pose_codebook

            self.initted[:] = True
            print("Successfully initialized the embeddings from vqvae codebook embeddings.")

    def _build_transformer_backbone(self):
        """ @brief: the transformer backbone.
        """
        encoder_layer = nn.TransformerEncoderLayer(d_model=self._args['n_embd'],
                                                   nhead=self._args['n_head'], batch_first=True)
        self._transformer_model = nn.TransformerEncoder(encoder_layer, num_layers=self._args['n_layers'])

    def _build_input_embeddings_projections(self):
        """ @brief: build the embeddings for the inputs
        """
        # step 1: the projection & embedding matrix for input to the transformer; the input includes
        # 1) pose tokens, 2) local root values, 3) pose values, 4) num_of_tokens
        num_pose_heads, _ = self.get_num_heads()
        num_pose_tokens, _ = self.get_num_codes(include_aug_tokens=False)
        if self._args['pose_vqvae'].get('has_codebook', True):
            pose_token_dim = self._args['pose_vqvae']['code_dim'] // num_pose_heads
        else:
            pose_token_dim = self._args['n_embd'] // num_pose_heads

        self._pose_token_emb = t.nn.Embedding(num_pose_heads * self.NUM_WITH_AUG_POSE_TOKENS, pose_token_dim)
        self._proj_pose_token_emb = mlp(num_layers=self._args['pose_token_mlp_num_layers'],
                                        layer_width=self._args['pose_feat_width'],
                                        size_in=num_pose_heads * pose_token_dim,
                                        size_out=self._args['pose_feat_width'])  # 1) pose tokens

        num_frames_per_token = self.get_num_frames_per_token()
        if self._args['cond_root_feature_is_from_motion_rep'] == 'local':
            self._args['local_root_dim'] = extract_feature_from_motion_rep(t.zeros([1, 1, 1000]), self.local_motion_rep,
                                                                          self._args['cond_root_feature']).shape[2]
        else:
            self._args['local_root_dim'] = extract_feature_from_motion_rep(t.zeros([1, 1, 1000]), self.global_motion_rep,
                                                                          self._args['cond_root_feature']).shape[2]
        self._proj_local_root_values = t.nn.Linear(self._args['local_root_dim'] * num_frames_per_token,
                                                   self._args['root_feat_width'])  # 2) local root values

        self._args['local_pose_dim'] = extract_feature_from_motion_rep(t.zeros([1, 1, 1000]), self.local_motion_rep,
                                                                      self._args['local_pose_feature']).shape[2]
        self._proj_local_pose = t.nn.Linear(self._args['local_pose_dim'],
                                            self._args['pose_feat_width'] // num_frames_per_token)  # 3) pose values
        assert self._args['pose_feat_width'] % self.get_num_frames_per_token() == 0, \
            "pose_feat_width needs to be divisible by num_frames_per_token"

        self._proj_num_valid_positions = t.nn.Embedding(self._args['max_tokens'] - self._args['min_tokens'] + 1,
                                                        self._args['token_length_feat_width'])  # 4) num of tokens

        # step 2: the proj to merge all input features into the input and positional emb
        self._proj_input = t.nn.Sequential(
            t.nn.Linear(self._args['pose_feat_width'] + self._args['root_feat_width'] +
                        self._args['token_length_feat_width'], self._args['n_embd']), t.nn.ReLU()
        )
        self._position_emb = PositionEmbedding(seq_length=self._args['max_tokens'],
                                               dim=self._args['n_embd'])  # the std for this fixed position emb is 0.5

        # step 3: the output logit projection matrix
        self._proj_pose_output_logit = t.nn.Linear(self._args['n_embd'], num_pose_heads * num_pose_tokens)

        # step 4: (optional)
        if self.ACCEPT_TEXT_EMB_INPUT:
            self._proj_text_embeddings = mlp(num_layers=self._args['pose_token_mlp_num_layers'],
                                             layer_width=self._args['n_embd'],
                                             size_in=self._args['text_emb_dim'], size_out=self._args['n_embd'])

    def forward(self, pose_tokens: t.Tensor, local_root_values: t.Tensor,
                pose_cond: t.Tensor, has_pose_cond: t.Tensor, num_tokens: t.Tensor,
                text_embeddings: t.Tensor = None, has_text_embeddings: t.Tensor = None):
        """
        @input pose_tokens: [batch, numTokens, num_pose_heads]             # required; could be all masked
        @input local_root_values: [batch, numFrames, 4]                    # required
        @input pose_cond: [batch, numConstrainFrames=[0, 10], featdim]     # optional
        @input has_poses_cond: [batch, numConstrainFrames=[0, 10]] (int)   # required; where poses are provided
        @input num_tokens: [batch, 1]                                      # required
        @input text_embedding: [batch, numTokens, text_embedding_dim]      # optional
        """
        batch_size, num_positions, num_pose_heads = pose_tokens.shape
        device = pose_tokens.device
        num_frames_per_token = self.get_num_frames_per_token()

        # step 1: generate the input embeddings for each term
        local_root_values = local_root_values.reshape([batch_size, num_positions,
                                                       num_frames_per_token * self._args['local_root_dim']])
        root_embedding = self._proj_local_root_values(local_root_values)  # [batch, num_positions, feat_dim]

        # step 2: the pose embeddings, merged from both pose token embedding and the pose value embeddings
        dense_pose_cond, dense_has_pose_cond = \
            convert_sparse_cond_to_dense_cond_if_needed(pose_cond, has_pose_cond, num_positions * num_frames_per_token)
        pose_cond_embedding = self._proj_local_pose(dense_pose_cond)  # [batch, numFrames, feat_dim]
        pose_cond_embedding = pose_cond_embedding.view([batch_size, num_positions * num_frames_per_token,
                                                        self._args['pose_feat_width'] // num_frames_per_token])

        pose_token_id_offsets = \
            torch.arange(num_pose_heads).view([1, 1, num_pose_heads]).to(device=device) * self.NUM_WITH_AUG_POSE_TOKENS
        pose_tokens = pose_tokens + pose_token_id_offsets   # [batch, num_positions, num_heads]
        pose_token_embedding = self._pose_token_emb(pose_tokens)  # [batch, num_positions, num_heads, feat_dim]
        pose_token_embedding = self._proj_pose_token_emb(pose_token_embedding.view([batch_size, num_positions, -1]))
        pose_token_embedding = pose_token_embedding.view([batch_size, num_positions * num_frames_per_token,
                                                          self._args['pose_feat_width'] // num_frames_per_token])

        dense_has_pose_cond = dense_has_pose_cond[:, :, None].float()  # [batch, numFrames, 1]
        pose_embedding = pose_cond_embedding * dense_has_pose_cond + pose_token_embedding * (1 - dense_has_pose_cond)
        pose_embedding = pose_embedding.view([batch_size, num_positions, self._args['pose_feat_width']])

        # step 3: the number of token embeddings [batch, num_positions, embedding_dim]
        num_token_embedding = self._proj_num_valid_positions(
            num_tokens.reshape([batch_size, 1]) - self._args['min_tokens']).expand([-1, num_positions, -1]
        )

        # step 4: merge the embeddings and apply positional emb; final shape [batch, num_positions, n_embd]
        position_ids = t.arange(num_positions).to(device=device)
        position_emb = self._position_emb.embed[position_ids].view([1, num_positions, self._args['n_embd']])
        input_embeddings = \
            self._proj_input(t.cat([pose_embedding, root_embedding, num_token_embedding], dim=-1)) + position_emb

        # step 5: the text input
        token_mask = t.arange(num_positions).to(device) < num_tokens.view([batch_size, -1])
        if self.ACCEPT_TEXT_EMB_INPUT and text_embeddings is not None:
            text_embeddings = self._proj_text_embeddings(text_embeddings)[:, None, :]
            input_embeddings = t.concat([text_embeddings, input_embeddings], dim=1)  # [batch, 1+num_positions, n_embd]
            mask = t.zeros([batch_size, 1 + num_positions], device=device).bool()  # zeros mean allowing
            mask[:, :1] = ~has_text_embeddings.bool()
            mask[:, 1:] = ~token_mask.bool()
        else:
            mask = ~token_mask.bool()

        # The shape of the 2D attn_mask is torch.Size([256, 83]), but should be (83, 83).
        output = self._transformer_model(input_embeddings, src_key_padding_mask=mask)[:, -num_positions:]
        pred_logits = self._proj_pose_output_logit(output).reshape([batch_size, num_positions, num_pose_heads, -1])

        return {'pose_logits': pred_logits}

    def get_num_codes(self, include_aug_tokens: bool):
        num_codes_per_head = {}

        for vqvae in ['pose_vqvae', 'root_vqvae']:  # calculate nb_code per head from overall nb_codes
            nb_code = self._args[vqvae]['nb_code']
            len_nb_code = len(np.array(self._args[vqvae]['nb_code']).reshape([-1]))
            if len_nb_code > 1:  # likely a fsq config
                assert np.all(np.array(nb_code) == nb_code[0]), \
                    "fsq config should have the same number of codes for each head"
                num_codes_per_head[vqvae] = nb_code[0]
            else:
                num_heads = self._args[vqvae]['num_heads']
                code_dim = self._args[vqvae]['code_dim']
                if self._args[vqvae].get('has_codebook', True):
                    assert code_dim % num_heads == 0, "code dim cannot be divided by the number of heads."
                    num_codes_per_head[vqvae] = int(round(2 ** (np.log2(nb_code) / num_heads)))
                    assert num_codes_per_head[vqvae] ** num_heads == nb_code, \
                        "the specified number of code is not compatible with the number of heads."
                else:
                    num_codes_per_head[vqvae] = nb_code

        if include_aug_tokens:  # include a [MASK] token
            return num_codes_per_head['pose_vqvae'] + 1, num_codes_per_head['root_vqvae'] + 1
        else:
            return num_codes_per_head['pose_vqvae'], num_codes_per_head['root_vqvae']

    def get_num_heads(self):
        """ brief: num heads for the tokenizers """
        return self._args['pose_vqvae']['num_heads'], self._args['root_vqvae']['num_heads']

    def get_num_frames_per_token(self):
        return (2 ** self._args['down_t'])

    # some handy properties
    @property
    def NUM_WITH_AUG_POSE_TOKENS(self):
        return self.get_num_codes(include_aug_tokens=True)[0]

    @property
    def NUM_WITH_AUG_ROOT_TOKENS(self):
        return self.get_num_codes(include_aug_tokens=True)[1]

    @property
    def POSE_MASK_ID(self):
        return self.get_num_codes(include_aug_tokens=True)[0] - 1

    @property
    def ROOT_MASK_ID(self):
        return self.get_num_codes(include_aug_tokens=True)[1] - 1

    @cached_property
    def ACCEPT_TEXT_EMB_INPUT(self):
        return True if self._args.get('text_embeddings', None) is not None else False
