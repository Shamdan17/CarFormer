import torch
import numpy as np
from enum import IntEnum
from typing import List


class TokenTypeIDs(IntEnum):
    GOAL = 4
    STATE = 0
    BEV = 1
    ACTION = 2
    REWARD = 3


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
        widths = np.ones(len(tensors), dtype=np.int)

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
    for i in range(len(tensors)):
        index_slice_length = (
            np.ceil(tensor_widths[i] / widths[i]).astype(np.int) * widths[i]
        )
        to_interleave.append(
            indices[:index_slice_length].reshape(-1, widths[i])
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
    result = {}
    for tokenType in TokenTypeIDs:
        # Get the indices of the unique token type ids
        token_type_id_indices = torch.where(interleaved_token_type_ids[0] == tokenType)[
            0
        ]
        if len(token_type_id_indices) > 0:
            # Get the tensor corresponding to the token type id
            result[tokenType] = torch.index_select(
                interleaved_tensors, axis, token_type_id_indices
            )

    return result


def move_padding_to_end(
    input_ids,
    token_type_ids=None,
    attention_mask=None,
    padding_id=0,
    axis=-1,
):
    """
    Moves the padding tokens to the end of the tensor
    Args:
        input_ids: A tensor of shape (batch, num_steps*width)
        attention_mask: A tensor of shape (batch, num_steps*width)
        token_type_ids: A tensor of shape (batch, num_steps*width)
    Returns:
        A tuple of (input_ids, attention_mask, token_type_ids) with the padding tokens moved to the end
    """
    padding_idx = input_ids == padding_id

    # Argsort to get the indices of the sorted array
    indices = torch.argsort(padding_idx.int(), dim=axis, stable=True)

    input_ids = torch.gather(input_ids, dim=-1, index=indices)
    if token_type_ids is not None:
        token_type_ids = torch.gather(token_type_ids, dim=-1, index=indices)

    if attention_mask is not None:
        attention_mask = torch.gather(attention_mask, dim=-1, index=indices)
        # mark padding tokens as 0
        attention_mask[input_ids == padding_id] = 0

    if attention_mask is None and token_type_ids is None:
        return input_ids
    else:
        return tuple(
            x for x in [input_ids, token_type_ids, attention_mask] if x is not None
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


# BEV to RGB conversion

# DEFAULT_HEIGHT = 336  # its 84m when density is 4px/m
# DEFAULT_WIDTH = 150  # its 37.5m when density is 4px/m

# BirdView = np.ndarray  # [np.uint8] with shape (level, y, x)
# RgbCanvas = np.ndarray  # [np.uint8] with shape (y, x, 3)

# COLOR_ON = 1


# class RGB:
#     VIOLET = (173, 127, 168)
#     ORANGE = (252, 175, 62)
#     CHOCOLATE = (233, 185, 110)
#     CHAMELEON = (138, 226, 52)
#     SKY_BLUE = (114, 159, 207)
#     DIM_GRAY = (105, 105, 105)
#     DARK_GRAY = (50, 50, 50)
#     RED = (255, 0, 0)
#     GREEN = (0, 255, 0)
#     YELLOW = (255, 255, 0)
#     WHITE = (255, 255, 255)


# class BirdViewMasks(IntEnum):
#     PEDESTRIANS = 7
#     RED_LIGHTS = 6
#     YELLOW_LIGHTS = 5
#     GREEN_LIGHTS = 4
#     AGENT = 3
#     VEHICLES = 2
#     #    CENTERLINES = 2
#     LANES = 1
#     ROAD = 0

#     @staticmethod
#     def top_to_bottom() -> List[int]:
#         return list(BirdViewMasks)

#     @staticmethod
#     def bottom_to_top() -> List[int]:
#         return list(reversed(BirdViewMasks.top_to_bottom()))


# RGB_BY_MASK = {
#     BirdViewMasks.PEDESTRIANS: RGB.VIOLET,
#     BirdViewMasks.RED_LIGHTS: RGB.RED,
#     BirdViewMasks.YELLOW_LIGHTS: RGB.YELLOW,
#     BirdViewMasks.GREEN_LIGHTS: RGB.GREEN,
#     BirdViewMasks.AGENT: RGB.CHAMELEON,
#     BirdViewMasks.VEHICLES: RGB.ORANGE,
#     # BirdViewMasks.CENTERLINES: RGB.CHOCOLATE,
#     BirdViewMasks.LANES: RGB.WHITE,
#     BirdViewMasks.ROAD: RGB.DIM_GRAY,
# }

# BIRDVIEW_SHAPE_CHW = (len(RGB_BY_MASK), DEFAULT_HEIGHT, DEFAULT_WIDTH)
# BIRDVIEW_SHAPE_HWC = (DEFAULT_HEIGHT, DEFAULT_WIDTH, len(RGB_BY_MASK))


# class BirdViewProducer:
#     """Responsible for producing top-down view on the map, following agent's vehicle.

#     About BirdView:
#     - top-down view, fixed directly above the agent (including vehicle rotation), cropped to desired size
#     - consists of stacked layers (masks), each filled with ones and zeros (depends on MaskMaskGenerator implementation).
#         Example layers: road, vehicles, pedestrians. 0 indicates -> no presence in that pixel, 1 -> presence
#     - convertible to RGB image
#     - Rendering full road and lanes masks is computationally expensive, hence caching mechanism is used
#     """

#     def __init__(self) -> None:
#         pass

#     @staticmethod
#     def as_rgb(birdview: BirdView) -> RgbCanvas:
#         _, h, w = birdview.shape
#         rgb_canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
#         nonzero_indices = lambda arr: arr == COLOR_ON

#         for mask_type in BirdViewMasks.bottom_to_top():
#             rgb_color = RGB_BY_MASK[mask_type]
#             mask = birdview[mask_type]
#             # If mask above contains 0, don't overwrite content of canvas (0 indicates transparency)
#             rgb_canvas[nonzero_indices(mask)] = rgb_color
#         return rgb_canvas


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
    trainable_param_size = 0
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        if param.requires_grad:
            trainable_param_num += 1
            trainable_param_size += param.nelement() * param.element_size()
            trainable_param_num += param.nelement()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    size_trainable_mb = trainable_param_size / 1024**2
    print("Total model size: {:.3f}MB".format(size_all_mb))
    print(
        "Trainable params: {} ({:.3f}MB)".format(trainable_param_num, size_trainable_mb)
    )


# mask_colors = np.asarray(
#     [[105, 105, 105],
#     [255,255,255],
#     [252,175,62],
#     [138,226,52],
#     [  0,255, 0],
#     [255,255, 0],
#     [255,  0, 0],
#     [173,127,168],
#     [0, 0, 0]]
# ).astype(float)/255

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
