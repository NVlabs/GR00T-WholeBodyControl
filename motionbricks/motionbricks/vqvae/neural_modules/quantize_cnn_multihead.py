# NOTE: taken from MotionGPT code base
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize

class QuantizeEMAResetMultiHead(nn.Module):
    def __init__(self, nb_code: int, code_dim: int, args):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = args.mu                       # the decay for ema reset
        self.num_heads = args.num_heads
        assert self.code_dim % self.num_heads == 0, "code dim cannot be divided by the number of heads."
        self.nb_code_per_head = int(round(2 ** (np.log2(self.nb_code) / self.num_heads)))
        assert self.nb_code_per_head ** self.num_heads == self.nb_code, \
            "the specified number of code is not compatible with the number of heads."
        self.vq = VectorQuantize(
            dim=self.code_dim,
            codebook_dim=self.code_dim // self.num_heads,       # smaller codebook dimension is acceptable
            heads=self.num_heads,             # number of heads to vector quantize, codebook shared across all heads
            separate_codebook_per_head=True,  # whether to have a separate codebook per head.
            codebook_size=self.nb_code_per_head,
            accept_image_fmap=False,
            threshold_ema_dead_code = 1,      # if the number of code usage < 1; reset it
            decay=self.mu,
            kmeans_init=bool(args.kmeans_init)
        )  # commitment loss coeff = 1.0 since we do that weighting outside  #
        self.init = True
        self._calculate_per_head_perplexity = getattr(args, "calculate_per_head_perplexity", False)

    @torch.no_grad()
    def compute_perplexity(self, code_idx, nb_code=None):
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code if nb_code is None else nb_code,
                                  code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def compute_perplexity_per_head(self, code_idx):
        # Calculate new centres
        perplexity = 0.0
        for i in range(self.num_heads):
            perplexity += self.compute_perplexity(code_idx[:, :, i].view(-1), nb_code=self.nb_code_per_head)
        return perplexity / self.num_heads

    def from_mh_indices_to_overall_indices(self, mh_indices):
        exponential_mul = \
            self.nb_code_per_head ** torch.arange(0, self.num_heads)[None, None, :].to(mh_indices.device)
        overall_indices = (exponential_mul * mh_indices).sum(dim=-1)
        return overall_indices

    def from_overall_indices_to_mh_indices(self, overall_indices):
        exponential_mul = \
            self.nb_code_per_head ** torch.arange(0, self.num_heads)[None, None, :].to(overall_indices.device)

        mh_indices = (overall_indices[:, :, None] % (exponential_mul * self.nb_code_per_head)) // exponential_mul
        return mh_indices

    def dequantize(self, code_idx, use_overall_indices = True):
        if use_overall_indices:
            mh_indices = self.from_overall_indices_to_mh_indices(code_idx)
        else:
            mh_indices = code_idx  # [batch, numTokens, numHeads]
        batch_size, numToken = code_idx.shape[0], code_idx.shape[1]
        x = self.vq.get_codes_from_indices(mh_indices).view([batch_size, numToken, -1])
        return x

    def forward(self, x):
        N, width, T = x.shape
        # expected the shape of the input to vq: (1, 1024, 256) --> batch, Timesteps, feat_dim
        # The input to x is batch, width(feat_dim), T
        x = x.permute(0, 2, 1).contiguous()

        # quantize and dequantize through bottleneck
        x_d, mh_indices, commit_loss = self.vq(x)

        # Update embeddings
        if self._calculate_per_head_perplexity:
            if self.num_heads == 1:
                mh_indices = mh_indices[:, :, None]
            perplexity = self.compute_perplexity_per_head(mh_indices)
        else:
            overall_indices = self.from_mh_indices_to_overall_indices(mh_indices)
            perplexity = self.compute_perplexity(overall_indices.view([-1]))

        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)

        return x_d, commit_loss, perplexity

    def forward_into_idx(self, x, fetch_overall_indices: bool = True):
        N, width, T = x.shape

        # expected the shape of the input: (1, 1024, 256) --> batch, T, width
        # The input to x is batch, width, T
        x = x.permute(0, 2, 1).contiguous()

        # quantize and dequantize through bottleneck
        _, mh_indices, _ = self.vq(x)
        overall_indices = self.from_mh_indices_to_overall_indices(mh_indices)

        if fetch_overall_indices:
            return overall_indices
        else:
            return mh_indices

class QuantizeEMAResetLFQ(nn.Module):
    pass
