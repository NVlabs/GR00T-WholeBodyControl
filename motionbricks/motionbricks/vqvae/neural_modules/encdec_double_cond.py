import torch.nn as nn
import torch
from motionbricks.vqvae.neural_modules.resnet import Resnet1D

class DoubleCondDecoder(nn.Module):
    def __init__(self,
                 input_emb_width: int = 3,
                 output_emb_width: int = 512,
                 down_t: int = 3,
                 width: int = 512,
                 depth: int = 3,
                 dilation_growth_rate: int = 3,
                 activation: str = 'relu',
                 norm: str = None,
                 target_cond_dim: int = -1,
                 external_cond_dim: int = -1,
                 cond_fusion_last_layer=False):
        """ @brief: consider both the internal target condition and external target condition.
            external condition is used to help generate the results, while target condition is directly the results
            we want to reconstructed.
            Target condition is a way to enforce strict constraints on the results.
        """
        super().__init__()

        # log configurations
        self._down_t, self._width, self._input_emb_width, self._output_emb_width = \
            down_t, width, input_emb_width, output_emb_width
        self._target_cond_dim, self._external_cond_dim = \
            target_cond_dim, external_cond_dim
        self._HAS_EXTERNAL_COND, self._HAS_TARGET_COND = external_cond_dim > 0, target_cond_dim > 0
        self._COND_FUSION_LAST_LAYER = cond_fusion_last_layer

        # step 1: the main model
        blocks = []
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))  # this does not change sequence length
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

        """ step 2: external cond embeddings.
        At each layer, the external embedding will be merged with the hidden state. The external condition is dense
        and expected to be always available in each frame.
        """
        if self._HAS_EXTERNAL_COND:
            external_cond_blocks = []
            for i in range(down_t + (1 if self._COND_FUSION_LAST_LAYER else 0)):
                cond_feat_dim = (2 ** (down_t - i)) * self._external_cond_dim
                external_cond_blocks.append(nn.Linear(cond_feat_dim + width, width))
                external_cond_blocks.append(nn.ReLU())
            self.external_cond_blocks = nn.ModuleList(external_cond_blocks)

        """ step 3: target cond embeddings.
        The target condition is sparse and expected to be available in certain frames.
        we replace the hidden state with the target condition embeddings for given frames.
        In earlier layers where each position corresponds to multiple frames, we reshape the hidden states to map to
        each frame position (see @forward method)
        """
        if self._HAS_TARGET_COND:
            target_cond_blocks = []
            assert width % (2 ** down_t) == 0, \
                "width % (2 ** down_t) needs to 0 so that hidden can be split for each frame in earlier layers."
            for i in range(down_t + (1 if self._COND_FUSION_LAST_LAYER else 0)):
                target_cond_blocks.append(nn.Linear(self._target_cond_dim, int(width / (2 ** (down_t - i)))))
                target_cond_blocks.append(nn.ReLU())
            self.target_cond_blocks = nn.ModuleList(target_cond_blocks)

    def forward(self, x: torch.Tensor, external_cond: torch.Tensor = None,
                target_cond: torch.Tensor = None, has_target_cond: torch.Tensor = None,
                token_mask: torch.Tensor = None):
        """ @brief: the decoder could take the external condition and target condition as input.
        @params x: shape -> [batch, feat_dim, timesteps // (2 ** down_t)]
        @params external_cond: shape -> [batch, timesteps, feat_dim]
        @params target_cond: shape -> [batch, timesteps, feat_dim]
        @params has_target_cond: shape -> [batch, timesteps] (dtype=bool)
        """
        batch_size = x.shape[0]

        # preprocess
        x = x * token_mask[:, None, :] if token_mask is not None else x  # zeroing out the padded tokens' embeddings
        h = self.model[0](x)  # conv1d
        h = self.model[1](h)  # relu; h.shape = ([batch, width, timesteps // (2 ** down_t)])

        for i in range(self._down_t + (1 if self._COND_FUSION_LAST_LAYER else 0)):
            numFrames_per_position = 2 ** (self._down_t - i)
            numPositions = h.shape[-1]  # numPositions = timesteps // numFrames_per_position
            timesteps = numPositions * numFrames_per_position

            # step 1: consider the target cond
            if (not self._HAS_TARGET_COND) or target_cond is None or has_target_cond is None:
                pass
            else:
                h_target_cond = self.target_cond_blocks[i * 2](target_cond)             # [batch, timesteps, feat]
                h_target_cond = self.target_cond_blocks[i * 2 + 1](h_target_cond)       # [batch_size, timesteps, feat]

                # h had shape [batch, feat=self._width, numPositions] at the beginning of each loop
                h = h.transpose(1, 2)                                                   # [batch, numPos, self._width]
                h = h.reshape([batch_size, numPositions * numFrames_per_position,
                               self._width // numFrames_per_position])                  # [batch, timesteps, feat]
                h = torch.where(has_target_cond[:, :, None], h_target_cond, h)          # [batch, timesteps, feat]
                h = h.reshape([batch_size, numPositions, self._width]).transpose(1, 2)  # [batch, feat, numPositions]

            # step 2: merge the emb from external cond and the original h
            if self._HAS_EXTERNAL_COND:
                assert external_cond is not None and external_cond.shape[1] == timesteps

                h_cond = external_cond.reshape([batch_size, numPositions, -1])          # [batch, numPosition, feat]
                h = torch.cat([h.transpose(1, 2), h_cond], dim=-1)                      # [batch, numPosition, feat]
                h = self.external_cond_blocks[i * 2](h).transpose(1, 2)                 # [batch, feat, numPosition]
                h = self.external_cond_blocks[i * 2 + 1](h)  # relu

            if i == self._down_t:  # if fusion the cond at the last layer, skip the last main model
                continue

            # step 3: the main model
            h = self.model[i + 2][0](h, token_mask)
            h = self.model[i + 2][1](h)  # upsampling
            token_mask = token_mask.repeat_interleave(2, 1) if token_mask is not None else None
            h = h * token_mask[:, None, :] if token_mask is not None else h  # zeroing out padded tokens' embeddings
            h = self.model[i + 2][2](h)  # conv1d                                       # [batch, feat_dim, numPosition]

        # post process
        h = h * token_mask[:, None, :] if token_mask is not None else h  # zeroing out the padded tokens' embeddings
        h = self.model[2 + self._down_t](h)         # conv1d
        h = self.model[2 + self._down_t + 1](h)     # relu
        h = h * token_mask[:, None, :] if token_mask is not None else h  # zeroing out the padded tokens' embeddings
        h = self.model[2 + self._down_t + 2](h)     # conv1d
        h = h * token_mask[:, None, :] if token_mask is not None else h  # zeroing out the padded tokens' embeddings
        return h
