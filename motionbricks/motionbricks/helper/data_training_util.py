import torch
import torch as t
import numpy as np
from motionbricks.motionlib.core.motion_reps import MotionRepBase


def sample_motion_segments_from_motion_clips(motions: t.Tensor, motion_lengths: t.Tensor,
                                             num_desired_frames: int, batchsize_mul_factor_for_segments: int,
                                             info: dict = {},
                                             motion_rep: MotionRepBase = None):
    """ @brief: the dataloader provides motion clips, which are variable length sequences of motion frames.
    In training of vqvae and the backbone model, we require fixed length motion segments.

    @param motions: normalized global motion rep, shape [batch_size, max_motion_length, motion_dim]

    @param motion_lengths: length of each motion clip. shape [batch_size],

    @param num_desired_frames: the number of frames in each motion segment.
        NOTE: the output motion segment will have (num_desired_frames + 1) frames so that user could decide whether to
        use just global motion rep or need to convert global to local later. The convertion loses 1 frame.

    @param batchsize_mul_factor_for_segments:
        the number of output motion segments = batch_size * batchsize_mul_factor_for_segments
    """
    batch_size = motions.shape[0]
    augmented_batch_size = int(batch_size * batchsize_mul_factor_for_segments)
    device = motions.device

    valid_samples_id = (motion_lengths >= num_desired_frames + 1)  # 1 additional frame for global-local convertion
    num_invalid_samples = batch_size - valid_samples_id.sum()
    assert num_invalid_samples < batch_size, "all samples are invalid."
    num_new_motions = (batchsize_mul_factor_for_segments - 1) * batch_size + num_invalid_samples
    p_sample = \
        (motion_lengths - num_desired_frames).clip(min=.0) * valid_samples_id.float()  # invalid samples has 0.0 prob
    p_sample = p_sample.cpu().numpy().astype("float64")  # this is needed otherwise sum!= 1.0 precision error
    new_motions_ids = np.random.choice(batch_size, replace=True,
                                       size=num_new_motions.item(), p=(p_sample / p_sample.sum()))

    motions = t.concat([motions[valid_samples_id], motions[new_motions_ids]], dim=0)
    info['chosen_ids'] = t.concat([t.where(valid_samples_id)[0], t.from_numpy(new_motions_ids).to(device)], dim=0)
    motion_lengths = t.concat([motion_lengths[valid_samples_id], motion_lengths[new_motions_ids]], dim=0)

    m_start_idx = t.randint(low=0, high=31415926, size=[motions.shape[0]]).to(device)  # just use a big high value
    m_start_idx = m_start_idx % (motion_lengths - (num_desired_frames + 1) + 1)  # at least (numFrame+1) frames left

    m_chunk_idx = torch.arange(num_desired_frames + 1).tile([augmented_batch_size, 1]).to(device) + m_start_idx[:, None]
    motions = motions.gather(1, m_chunk_idx[:, :, None].tile([1, 1, motions.shape[-1]]))

    return motions


def sample_keyframes(motions: t.Tensor, max_num_keyframes: int, prob_num_keyframes: list, bound_keyframes: bool = False):
    batch_size, num_frames = motions.shape[0], motions.shape[1]
    if bound_keyframes: # only use the first and last frame as the keyframe
        chosen_keyframes = t.zeros([batch_size, num_frames], dtype=t.bool).to(motions.device)
        chosen_keyframes[:, [0, -1]] = True
        masked_motions = motions * chosen_keyframes[:, :, None].float()
    else:
        assert max_num_keyframes == len(prob_num_keyframes) - 1, "for compatibility we still have max_num_keyframes"
        assert num_frames >= max_num_keyframes

        num_keyframes = np.random.choice(np.arange(max_num_keyframes + 1),
                                        p=prob_num_keyframes, size=batch_size, replace=True)
        num_keyframes = t.tensor(num_keyframes).to(motions.device).clip(max=num_frames)

        candidates = t.rand_like(motions[:, :, 0])
        chosen_keyframes = candidates.sort()[1] < num_keyframes[:, None]

        masked_motions = motions * chosen_keyframes[:, :, None].float()

    return chosen_keyframes, masked_motions

def convert_sparse_cond_to_dense_cond_if_needed(cond: t.Tensor, has_cond: t.Tensor, num_max_cond: int):
    """ @brief: we could either provide
    In the dense case:
        cond: [batch_size, numFrames (num_max_cond), feat_dim]
        has_cond: [batch_size, numFrames (num_max_cond)] (bool)
    In the sparse case (could be potentially more efficient for onnx trt models):
        cond: [batch_size, numCondionedFrames, feat_dim]
        has_cond: [batch_size, numConditionedFrames] (int)
    """
    assert cond.shape[1] == has_cond.shape[1]
    batch_size, feat_dim = cond.shape[0], cond.shape[2]

    if has_cond.dtype == t.bool:  # the dense case
        assert cond.shape[1] == num_max_cond
        return cond, has_cond
    else:  # the sparse case; this is very handy for onnx trt models for minimum data transfer
        assert cond.shape[1] <= num_max_cond

        # construct the dense results
        has_cond_map = t.nn.functional.one_hot(has_cond, num_classes=num_max_cond)  # [batch, numCondF, numF]
        dense_has_cond = has_cond_map.sum(dim=1).reshape([batch_size, num_max_cond, 1]).tile([1, 1, feat_dim]).bool()
        dense_cond = t.zeros([batch_size, num_max_cond, feat_dim]).to(cond.device, cond.dtype)
        dense_cond[dense_has_cond] = cond.view(-1)  # they are all in flatten view
        return dense_cond, dense_has_cond

def extract_feature_from_motion_rep(x: t.Tensor | None,
                                    motion_rep: MotionRepBase, feature: str, fetch_feat_idx: bool = False):
    """ @brief: extract the corresponding features from the original motion representation
    """

    if feature in ['root_with_mask', 'root_without_heading_with_mask',
                   'root_without_hip_height_with_mask', 'root_without_hip_height_without_heading_with_mask']:
        # in the end there's a mask bit attached to the feature vector returned
        # 1 indicates a reliable feature and 0 indicates a unrealiable feature
        if fetch_feat_idx:
            raise NotImplementedError("fetch_feat_idx is not supported for with_mask features")
        else:
            raw_feature = extract_feature_from_motion_rep(x, motion_rep, feature.split("_with_mask")[0])
            mask = t.zeros([x.shape[0], x.shape[1], 1]).to(x.device)
            return t.concat([raw_feature, mask], dim=-1)

    elif feature == 'root':
        if fetch_feat_idx:
            return motion_rep.indices['root']
        else:
            return x if x.shape[-1] == len(motion_rep.indices['root']) else x[:, :, motion_rep.indices['root']]

    elif feature == 'root_without_heading':
        if motion_rep.root_mode == 'global':
            feat_idx = np.concatenate([motion_rep.indices['global_root_pos_2d'],
                                       np.array([i for i in motion_rep.indices['global_root_pos']
                                                 if i not in motion_rep.indices['global_root_pos_2d']])])
        else:
            feat_idx = np.concatenate([motion_rep.indices['local_root_vel'],
                                       motion_rep.indices['global_root_y']])
        if fetch_feat_idx:
            return feat_idx
        else:
            return x if x.shape[-1] == len(feat_idx) else x[:, :, feat_idx]

    elif feature == 'root_without_hip_height':
        if motion_rep.root_mode == 'global':
            feat_idx = np.concatenate([motion_rep.indices['global_root_pos_2d'],
                                       motion_rep.indices['global_root_heading']])
        else:
            feat_idx = np.concatenate([motion_rep.indices['local_root_rot_vel'],
                                       motion_rep.indices['local_root_vel']])
        if fetch_feat_idx:
            return feat_idx
        else:
            return x if x.shape[-1] == len(feat_idx) else x[:, :, feat_idx]

    elif feature == 'root_without_hip_height_without_heading':
        if motion_rep.root_mode == 'global':
            feat_idx = motion_rep.indices['global_root_pos_2d']
        else:
            feat_idx = motion_rep.indices['local_root_vel']
        if fetch_feat_idx:
            return feat_idx
        else:
            return x if x.shape[-1] == len(feat_idx) else x[:, :, feat_idx]

    elif feature == 'pose':
        if fetch_feat_idx:
            return motion_rep.indices['all']
        else:
            assert x.shape[-1] == len(motion_rep.indices['all'])
            return x

    elif feature == 'joint_positions_and_rotations':
        feat_idx = np.concatenate([motion_rep.indices['ric_data'],
                                   motion_rep.indices['global_rot_data']])
        if fetch_feat_idx:
            return feat_idx
        else:
            return x if x.shape[-1] == len(feat_idx) else x[:, :, feat_idx]

    elif feature == 'joint_positions_and_rotations_and_hip_height':
        if motion_rep.root_mode == 'global':
            feat_idx = np.concatenate([np.array([i for i in motion_rep.indices['global_root_pos']
                                                 if i not in motion_rep.indices['global_root_pos_2d']]),
                                       motion_rep.indices['ric_data'],
                                       motion_rep.indices['global_rot_data']])
        else:
            feat_idx = np.concatenate([motion_rep.indices['global_root_y'],
                                       motion_rep.indices['ric_data'],
                                       motion_rep.indices['global_rot_data']])
        if fetch_feat_idx:
            return feat_idx
        else:
            return x if x.shape[-1] == len(feat_idx) else x[:, :, feat_idx]

    else:
        raise NotImplementedError
