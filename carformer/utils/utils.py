import torch
import numpy as np
from enum import IntEnum
from typing import List
from collections import defaultdict
from collections.abc import MutableMapping
from carla_birdeye_view import (
    BirdViewProducerObjectLevelRenderer,
    BirdViewCropType,
    PixelDimensions,
)
import os


class TokenTypeIDs(IntEnum):
    GOAL = 4
    STATE = 0
    BEV = 1
    ACTION = 2
    REWARD = 3
    EOS = -1


def interleave(
    tensors, axis=1, widths=None, token_type_ids_mapping=None, padding_masks=None
):
    """
    Interleave tensors along the specified axis.
    Args:
        tensors: A list of tensors to interleave, each of shape (..., num_steps*width, ...) at the specified axis
        axis: The axis to interleave along
        widths: The widths of each tensor. If None, the tensors are assumed to be the same width
        token_type_ids_mapping: A list of token type ids to assign to each tensor. If None, the token type ids are assigned sequentially
        padding_masks: A list of padding mask tensors of shape (batch, num_steps*width). 0 denotes not padded while 1 denotes a padding tensor.
                       If None, no padding is assumed. The padding masks are interleaved to match the interleaved tensors to ensure that
                       the padding is in the correct place and to get the attention mask correct.
    Returns:
        A tuple of (interleaved tensors, interleaved token type ids, interleaved attention masks)
    """
    if widths == None:
        widths = np.ones(len(tensors), dtype=int)

    assert len(tensors) == len(widths)

    sum_widths = sum(widths)

    tensor_widths = np.array([t.shape[axis] for t in tensors])
    # print(tensor_widths)
    tensor_width_cumsum = np.cumsum(tensor_widths)

    # Combined tensor
    combined = torch.cat(tensors, axis)

    device = combined.device

    if not token_type_ids_mapping:
        token_type_ids_mapping = [i for i in range(len(tensors))]
    token_type_ids = torch.cat(
        [
            torch.ones(tensors[i].shape[0], tensors[i].shape[axis], dtype=torch.int32)
            * token_type_ids_mapping[i]
            for i in range(len(tensors))
        ],
        axis=1,
    ).to(device)

    if not padding_masks:
        padding_masks = [
            torch.zeros(
                (tensors[i].shape[0], tensors[i].shape[axis]), dtype=torch.float
            )
            for i in range(len(tensors))
        ]

    padding_mask = torch.cat(padding_masks, 1).to(device)
    # Attention mask is 1 for non-padding and 0 for padding
    attention_mask = 1 - padding_mask

    # Create a list of indices to interleave
    indices = np.arange(combined.shape[axis])

    to_interleave = []
    max_timestamp = np.max(
        [
            np.ceil(tensor_widths[i] / widths[i]).astype(int)
            for i in range(len(tensors))
            if widths[i] > 0
        ]
    )

    for i in range(len(tensors)):
        index_slice_length = max_timestamp * widths[i]
        # print(index_slice_length)
        # np.ceil(tensor_widths[i] / widths[i]).astype(int) * widths[i]
        # )
        to_interleave.append(
            indices[:index_slice_length].reshape(max_timestamp, widths[i])
            + (tensor_width_cumsum[i - 1] if i > 0 else 0)
        )

    # Interleaved indices
    interleaved_indices = np.concatenate(to_interleave, axis=1).flatten()[
        : tensor_width_cumsum[-1]
    ]

    # To torch
    interleaved_indices = torch.from_numpy(interleaved_indices).to(device)

    # print(interleaved_indices)

    # Interleave
    interleaved = torch.index_select(combined, axis, interleaved_indices)
    # interleaved = torch.gather(combined, axis, interleaved_indices)

    # Token type ids
    interleaved_token_type_ids = token_type_ids[:, interleaved_indices]
    # Attention mask
    interleaved_attention_mask = attention_mask[:, interleaved_indices]

    # print("hello world")
    # print(interleaved)
    # print(interleaved_token_type_ids)
    # print(tensors)

    return interleaved, interleaved_token_type_ids, interleaved_attention_mask


def deinterleave(interleaved_tensors, interleaved_token_type_ids, axis=1):
    """
    Deinterleave tensors along the specified axis.
    Args:
        interleaved_tensors: A tensor of shape (..., step_width, ...) to deinterleave
        interleaved_token_type_ids: A tensor of shape (batch, step_width) containing the token type ids
        axis: The axis to deinterleave along
    Returns:
        A dictionary of deinterleaved tensors, keyed by token type id
    """
    results_by_tokentype = defaultdict(list)

    for tokenType in TokenTypeIDs:
        max_axis_length = 0
        for batch_idx in range(interleaved_token_type_ids.shape[0]):
            # Get the indices of the unique token type ids
            token_type_id_indices = torch.where(
                interleaved_token_type_ids[batch_idx] == tokenType
            )[0]

            # Get the tensor corresponding to the token type id
            results_by_tokentype[tokenType].append(
                torch.index_select(
                    interleaved_tensors[batch_idx].unsqueeze(0),
                    axis,
                    token_type_id_indices,
                )
            )
            max_axis_length = max(
                max_axis_length, results_by_tokentype[tokenType][-1].shape[axis]
            )

        # Pad the tensors to the max length
        output_shape = list(results_by_tokentype[tokenType][0].shape)
        for i in range(len(results_by_tokentype[tokenType])):
            output_shape[axis] = (
                max_axis_length - results_by_tokentype[tokenType][i].shape[axis]
            )
            if output_shape[axis] > 0:
                results_by_tokentype[tokenType][i] = torch.cat(
                    [
                        results_by_tokentype[tokenType][i],
                        torch.zeros(
                            output_shape, dtype=results_by_tokentype[tokenType][i].dtype
                        ).to(results_by_tokentype[tokenType][i].device),
                    ],
                    axis=axis,
                )

        # Concatenate the tensors
        results_by_tokentype[tokenType] = torch.cat(
            results_by_tokentype[tokenType], axis=0
        )

    return results_by_tokentype


def move_padding_to_end(
    input_ids,
    token_type_ids=None,
    attention_mask=None,
    inputs_embeds=None,
    other_tensors=None,
    padding_id=0,
    axis=-1,
    trim_padding=False,
):
    """
    Moves the padding tokens to the end of the tensor
    Args:
        input_ids: A tensor of shape (batch, num_steps*width)
        attention_mask: A tensor of shape (batch, num_steps*width)
        token_type_ids: A tensor of shape (batch, num_steps*width)
        padding_id: The id of the padding token
        axis: The axis to move the padding tokens along, defaults to -1
        trim_padding: Whether to trim the padding tokens from the end of the tensor
    Returns:
        A tuple of (input_ids, attention_mask, token_type_ids) with the padding tokens moved to the end
    """
    padding_idx = input_ids == padding_id

    # Argsort to get the indices of the sorted array
    indices = torch.argsort(padding_idx.int(), dim=axis, stable=True)
    # Use the indices to push the padding tokens to the end
    padding_idx = padding_idx.gather(dim=axis, index=indices)

    if trim_padding:
        # Get the index of the first axis where booleans are True
        sum_arr = torch.sum(padding_idx.int(), dim=0) == padding_idx.shape[0]
        if sum_arr.any():
            trimmed_length = torch.argmax(sum_arr.int())
        else:
            trimmed_length = padding_idx.shape[axis]

        # print(trimmed_length)
        # print(f"Trimming the length by {padding_idx.shape[axis] - trimmed_length} elements. Initial length: {padding_idx.shape[axis]}, final length: {trimmed_length}")

        indices = indices[:, :trimmed_length]

    input_ids = torch.gather(input_ids, dim=-1, index=indices)
    if token_type_ids is not None:
        token_type_ids = torch.gather(token_type_ids, dim=-1, index=indices)

    if attention_mask is not None:
        attention_mask = torch.gather(attention_mask, dim=-1, index=indices)
        # mark padding tokens as 0
        attention_mask[input_ids == padding_id] = 0

    if inputs_embeds is not None:
        # Last dimension is the embedding dimension, so we need to gather along the second to last dimension
        inputs_embeds = torch.gather(
            inputs_embeds,
            dim=-2,
            index=indices.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]),
        )

    # Other tensors are treated the same as inputs_embeds
    if other_tensors is not None:
        other_tensors = tuple(
            torch.gather(
                tensor,
                dim=-2,
                index=indices.unsqueeze(-1).expand(-1, -1, tensor.shape[-1]),
            )
            for tensor in other_tensors
        )

    if attention_mask is None and token_type_ids is None and inputs_embeds is None:
        return input_ids
    else:
        return tuple(
            x
            for x in [
                input_ids,
                token_type_ids,
                attention_mask,
                inputs_embeds,
                other_tensors,
            ]
            if x is not None
        )


def change_padding_to_ignore_index(input_ids, padding_idx=0, ignore_index=-100):
    """
    Changes the padding tokens to the ignore index (in place)
    Args:
        input_ids: A tensor of shape (batch, num_steps*width)
    Returns:
        A tensor of shape (batch, num_steps*width) with the padding tokens changed to the ignore index
    """
    input_ids[input_ids == padding_idx] = ignore_index
    return input_ids


# TODO: Implement for batched inputs
#       currently assumes entire batch is identical
# TODO: Optimize. This is slow.
def interpolate_next_token_type_id(
    token_type_ids, widths=None, token_type_ids_mapping=None, interpolation_length=1
):
    """
    Interpolate the next token type id for each token type id in the token type ids tensor.
    Args:
        token_type_ids: A tensor of shape (batch/1, seq_len) containing the token type ids
        widths: A list of widths of each token type id
        token_type_ids_mapping: A list of token type ids
    Returns:
        A single token type id
    """
    if widths == None:
        raise ValueError("Widths must be specified.")
    else:
        assert len(widths) == len(token_type_ids_mapping)

    if token_type_ids_mapping == None:
        # Take the unique token type ids
        token_type_ids_mapping = np.unique(token_type_ids)

    # Get the next token type ids
    # Filter out any token type ids that are not in the token type ids mapping
    filtered_type_ids = [
        token_type_id
        for token_type_id in token_type_ids[0]
        if int(token_type_id) in token_type_ids_mapping
    ]

    # flatten the mapping
    # widths = (2, 1, 3)
    # token_type_ids_mapping = (0, 1, 2)
    # flattened_mapping = (0, 0, 1, 2, 2, 2)
    flattened_mapping = []
    for i, token_type_id in enumerate(token_type_ids_mapping):
        flattened_mapping.extend([token_type_id] * widths[i])

    flattened_mapping = np.asarray(flattened_mapping)

    indices = np.arange(interpolation_length) + len(filtered_type_ids)
    indices = indices % len(flattened_mapping)

    # Interpolate the next token type ids
    next_token_type_id = flattened_mapping[indices]

    np_rep = next_token_type_id.repeat(token_type_ids.shape[0]).reshape(
        -1, interpolation_length
    )

    return torch.tensor(np_rep, dtype=torch.long, device=token_type_ids.device)


def calculate_model_stats(model):
    trainable_param_num = 0
    all_param_num = 0
    trainable_param_size = 0
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        if param.requires_grad:
            trainable_param_num += 1
            trainable_param_size += param.nelement() * param.element_size()
            trainable_param_num += param.nelement()
        else:
            all_param_num += param.nelement()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        all_param_num += buffer.nelement()

    all_param_num += trainable_param_num

    size_all_mb = (param_size + buffer_size) / 1024**2
    size_trainable_mb = trainable_param_size / 1024**2
    print("Total model size: {:.3f}MB".format(size_all_mb))
    print(
        "Trainable params: {} ({:.3f}MB)".format(trainable_param_num, size_trainable_mb)
    )
    return size_all_mb, size_trainable_mb, all_param_num, trainable_param_num


import numpy as np
import torch

mask_colors_np = (
    np.asarray(
        [
            [105, 105, 105],
            [255, 255, 255],
            [252, 175, 62],
            [138, 226, 52],
            [0, 255, 0],
            [255, 255, 0],
            [255, 0, 0],
            [173, 127, 168],
            [0, 0, 0],
        ]
    ).astype(np.float32)
    / 255
)

mask_colors_pt = torch.from_numpy(mask_colors_np)  # .cuda()


# @profile
def as_rgb(img):
    # Shape:
    # ((batch), 9, height, width), batch optional
    # Output shape:
    # ((batch), 3, height, width), batch optional
    # If np array, convert to torch tensor
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()
    else:
        img = img.float()

    if img.ndim == 3:
        remove_batch_dim = True
        img = img[None, ...]
    else:
        remove_batch_dim = False

    # global mask_colors_pt
    # if img.device != mask_colors_pt.device:
    # print(mask_colors_pt)
    mask_colors = mask_colors_pt.to(img.device)

    # mask_colors = mask_colors_pt

    output_shape = img.shape[:1] + (3,) + img.shape[2:]

    if img.shape[1] == 9:
        indices = 8 - torch.argmax(torch.flip(img, [1]), axis=1)
        output = mask_colors[indices].permute(0, 3, 1, 2)
    else:
        none_layer = (torch.sum(img, keepdims=True, axis=1) == 0).float()
        # print(img.shape)
        # Change to 9 channels
        img = torch.cat([img, none_layer], axis=1)

        indices = 8 - torch.argmax(torch.flip(img, [1]), axis=1)

        output = mask_colors[indices].permute(0, 3, 1, 2)

    if remove_batch_dim:
        output = output[0]

    return output


# Numpy version
def as_rgb_np(img):
    # Shape:
    # ((batch), 9, height, width), batch optional
    # Output shape:
    # ((batch), 3, height, width), batch optional
    if img.ndim == 3:
        remove_batch_dim = True
        img = img[None, ...]
    else:
        remove_batch_dim = False

    output_shape = img.shape[:1] + (3,) + img.shape[2:]

    if img.shape[1] == 9:
        indices = 8 - np.argmax(np.flip(img, 1), axis=1)
        output = mask_colors_np[indices].transpose(0, 3, 1, 2)
    else:
        output = np.zeros(output_shape, dtype=float)

        indices = 7 - np.argmax(np.flip(img, 1), axis=1)[None, ...]

        mask = np.sum(img, keepdims=True, axis=1) > 0

        output[0, :, mask[0, 0]] = mask_colors_np[indices[mask]]

    if remove_batch_dim:
        output = output[0]

    return output


# TODO: Cleanup
# To numpy
# Input:
# Either a list, numpy array or torch tensor
# Output:
# Numpy array
def to_numpy(x, dtype=np.float32):
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if np.issubdtype(dtype, np.floating):
            x = x.float()

        return x.numpy().astype(dtype)
    elif isinstance(x, list):
        return np.asarray(x).astype(dtype)
    else:
        raise ValueError("Input must be a list, numpy array or torch tensor")


def convert_numpy_to_type(x, type="numpy"):
    if type == "numpy":
        return x
    elif type == "torch":
        return torch.from_numpy(x)
    elif type == "list":
        return x.tolist()
    else:
        raise ValueError("Invalid type")


bev_size = 192
bev_crop = "front"

object_renderer = BirdViewProducerObjectLevelRenderer(
    PixelDimensions(bev_size, bev_size),
    pixels_per_meter=5,
    crop_type=BirdViewCropType.FRONT_AREA_ONLY,
)


def as_rgb_objectlevel(img, input_ids):
    return


# Normalize version compatible with torch tensors
def normalize_angle_torch(x):
    x = x % (2 * np.pi)  # force in range [0, 2 pi)
    x = torch.where(x > np.pi, x - 2 * np.pi, x)  # move to [-pi, pi)
    return x


import random

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def get_block_masks_from_token_type_ids(token_type_ids, to_mask):
    mask = torch.ones_like(token_type_ids, dtype=torch.bool)

    if to_mask:
        for to_mask_id in to_mask:
            start_idxes = (token_type_ids[:, 1:] == to_mask_id).logical_and(
                token_type_ids[:, :-1] != to_mask_id
            )

            mask[:, 1:][(token_type_ids == to_mask_id)[:, 1:]] = False
            mask[:, 1:][start_idxes] = True

    mask = mask.cumsum(1)
    return mask


# Flatten backbone config nested dicts into a single dict
# a: {b: c} -> {"a.b": c}
def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
