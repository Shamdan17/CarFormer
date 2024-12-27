from torch.utils.data import DataLoader
import argparse
from carformer.data import SequenceDataset
from tqdm import tqdm, trange
from carformer.wanderer import Wanderer
from carformer.backbone import gpt2
from carformer.utils import calculate_model_stats, as_rgb
from carformer.utils import TokenTypeIDs
from carformer.data import from_object_level_vector
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import wandb
from collections import defaultdict
import os
import numpy as np
import cv2
import json
from carformer.encoders.StoSAVi_metrics import (
    get_metrics,
    save_bins_to_image_nomiou,
    get_binary_mask_from_rgb,
)

# from matplotlib import pyplot as plt
# from matplotlib import rc
from .lazyplotlib import lazyplot as plt
from .lazyplotlib import lazyrc as rc

# activate latex text rendering
rc("text", usetex=True)


def save_bin_probs(
    deinterleaved_logits,
    ground_truth_labels,
    save_dir,
    offset_map,
    vocab_size_map,
    token_quantizers,
    token_ids=[TokenTypeIDs.ACTION, TokenTypeIDs.STATE, TokenTypeIDs.REWARD],
    save_idx=0,
    save_prefix="",
    save_suffix="",
    predicted_legend_label="Predicted",
    ground_truth_legend_labels=["Ground Truth"],
    save_separate=False,
):
    # deinterleaved_logits: id -> (seq_len, num_tokens) or list of (seq_len, num_tokens)
    # ground_truth_labels: id -> (seq_len, num_tokens) or list of id -> (seq_len, num_tokens)
    # save_dir: str
    # offset_map: dict(id -> start)
    # vocab_size_map: dict(id -> vocab_size)
    # token_quantizers: list of quantizers
    # token_ids: list of token ids
    # token_widths: list of token widths
    # save_idx: int, index of the sequence to save within the batch
    # save_prefix: str, prefix to add to the saved file name
    # save_suffix: str, suffix to add to the saved file name
    # predicted_legend_label: str, label for the predicted probabilities in the legend. If deiniterleaved_logits is a list, this will be a list of labels
    # ground_truth_legend_labels: str or list of str, labels for the ground truth probabilities in the legend
    # save_separate: bool, if True, save each input in the list as a separate file

    # If deinterleaved_logits is a list, then we have multiple predictions for the same sequence
    # If not, make it a list of length 1
    if not isinstance(deinterleaved_logits, list):
        deinterleaved_logits = [deinterleaved_logits]
        # Do the same for the predicted legend label
        if not isinstance(predicted_legend_label, list):
            predicted_legend_label = [predicted_legend_label]

    if isinstance(ground_truth_labels, dict):
        ground_truth_labels = [ground_truth_labels]
    if not isinstance(ground_truth_labels, list):
        ground_truth_labels = [ground_truth_labels]
    # Assert that the predicted legend label is a list of the same length as deinterleaved_logits
    assert len(predicted_legend_label) == len(
        deinterleaved_logits
    ), "The predicted legend label must be a list of the same length as deinterleaved_logits"

    if save_separate and len(deinterleaved_logits) > 1:
        # Call this function for each element in deinterleaved_logits
        for i, logits in enumerate(deinterleaved_logits):
            save_bin_probs(
                logits,
                ground_truth_labels,
                save_dir,
                offset_map,
                vocab_size_map,
                token_quantizers,
                token_ids,
                save_idx,
                save_prefix,
                save_suffix + f"_{i}",
                predicted_legend_label[i],
                ground_truth_legend_labels,
                save_separate,
            )

    for token_id, token_quantizer in zip(token_ids, token_quantizers):
        # if token_width != 1:
        #     continue
        token_width = len(token_quantizer)

        if token_id == TokenTypeIDs.ACTION:
            token_width = len(token_quantizer.attribute_names)
            # print("OVERRIDING TOKEN WIDTH FOR ACTION: ", token_width)

        for width_idx in range(token_width):
            width_dim = token_quantizer.get_dim_width(width_idx)
            start_idx_offset, end_idx_offset = token_quantizer.get_dim_boundaries(
                width_idx
            )

            # Resort the probs and labels according to the quantizer centroids
            centroids = token_quantizer.concatenated_centroids[
                start_idx_offset:end_idx_offset
            ]

            sorted_centroid_idx = np.argsort(centroids.flatten())

            if token_width > 1:
                name = token_quantizer.get_attribute_name(width_idx)
            else:
                name = token_id.name.lower()

            # First get the ground truth labels
            label_key = f"quantized_label_{token_id.name.lower()}"

            figsize = None
            fig = None
            ax = None

            for ground_truth_label, ground_truth_legend_label in zip(
                ground_truth_labels, ground_truth_legend_labels
            ):
                gt_labels = (
                    ground_truth_label[label_key][save_idx] - offset_map[token_id]
                )

                # print(gt_labels)

                # The gt_labels corresponding to the current width_idx are every width_idx-th element
                gt_labels = gt_labels[width_idx::token_width] - start_idx_offset

                # print(gt_labels)

                # convert gt_label to one-hot encoding with vocab_size_map[token_id] classes
                gt_labels = torch.nn.functional.one_hot(
                    gt_labels, num_classes=width_dim
                )

                gt_labels = gt_labels.cpu().numpy()

                gt_labels = gt_labels[:, sorted_centroid_idx]

                # Start plotting GT if not started
                if figsize is None:
                    figsize = (gt_labels.shape[0] * 4, 3)

                    n_classes = gt_labels.shape[1]

                    fig, ax = plt.subplots(
                        1, gt_labels.shape[0], figsize=figsize, squeeze=False
                    )

                # Plot the ground truth as a histogram
                for i, gt_prob in enumerate(gt_labels):
                    ax[0, i].hist(
                        range(n_classes),
                        bins=n_classes,
                        range=(-0.5, n_classes - 0.5),
                        weights=gt_prob,
                        alpha=0.5,
                        label=f"{ground_truth_legend_label} {name}",
                    )
                    ax[0, i].set_xticks(np.arange(n_classes))
                    ax[0, i].set_xticklabels(
                        [
                            f"{token_quantizer.concatenated_centroids.flatten()[j + start_idx_offset]:.1f}"
                            for j in sorted_centroid_idx
                        ],
                        ha="right",
                        rotation=45,
                    )

            # Iterate over the list of predictions
            for pred_idx, pred_probs in enumerate(deinterleaved_logits):
                pred_probs = pred_probs[token_id][save_idx][
                    width_idx::token_width,
                    offset_map[token_id]
                    + start_idx_offset : offset_map[token_id]
                    + end_idx_offset,
                ].softmax(dim=-1)

                pred_probs = pred_probs.cpu().numpy()

                pred_probs = pred_probs[:, sorted_centroid_idx]
                for i, pred_prob in enumerate(pred_probs):
                    ax[0, i].hist(
                        range(n_classes),
                        bins=n_classes,
                        range=(-0.5, n_classes - 0.5),
                        weights=pred_prob,
                        alpha=0.5,
                        label=f"{predicted_legend_label[pred_idx]} {name}",
                    )

            for i in range(gt_labels.shape[0]):
                ax[0, i].legend()

            # print(sorted_centroid_idx.shape, pred_probs.shape, gt_labels.shape)

            save_folder = os.path.join(
                save_dir,
                "bins",
                f"{save_prefix}_{name}",
            )

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            save_path = os.path.join(save_folder, f"{name}_probs_{save_suffix}.png")

            plt.tight_layout()
            plt.savefig(save_path)

            plt.close()
            # plt.show()


def visualize_trajectory_actions(
    trajectory,
    save_dir,
    action_names_mapping,
    save_prefix="",
    save_suffix="",
    save_idx=None,
):
    # Visualize the actions as a stem plot
    if save_idx is not None:
        actions = trajectory["action"][save_idx]
    else:
        actions = trajectory["action"]

    # Iterate over the action_names dictionary
    for action_idx, action_name in action_names_mapping.items():
        # Get the action values
        action_values = actions[:, action_idx]

        # Get the time steps
        time_steps = np.arange(len(action_values))

        # Plot the stem plot
        plt.figure(figsize=(10, 3))
        # y must range between -1 and 1, so we set the axis limits accordingly
        plt.ylim(-1.1, 1.1)
        plt.stem(time_steps, action_values, use_line_collection=True)
        plt.xlabel("Time Step")
        plt.ylabel(action_name)

        save_folder = os.path.join(
            save_dir,
            "actions",
            f"{save_prefix}_{action_name}",
        )

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(
            save_folder, f"{action_name}_actions_{save_suffix}.png"
        )

        plt.tight_layout()
        plt.savefig(save_path)

        plt.close()


def extract_target_point_from_trajectory_goals(goals, goal_type):
    trajectory_length = goals.shape[0]

    target_pts = np.zeros((trajectory_length, 2))

    if "target_point" in goal_type:
        goal_idx = goal_type.index("target_point")
    else:
        return None

    for i in range(trajectory_length):
        cur_target_point = goals[i][goal_idx : goal_idx + 2].numpy().copy()  # 2D

        cur_target_point[0] *= -1

        cur_target_point = np.flip(cur_target_point)

        target_pts[i] = cur_target_point

    return target_pts


def extract_waypoints_from_trajectory_actions(actions, action_type, waypoint_count=4):
    trajectory_length = actions.shape[0]

    waypoints = np.zeros((trajectory_length, waypoint_count, 2))

    if "waypoints" in action_type:
        waypoint_idx = action_type.index("waypoints")
    else:
        return None

    for i in range(trajectory_length):
        cur_waypoints = actions[i][
            waypoint_idx : waypoint_idx + waypoint_count * 2
        ]  # 2D

        cur_waypoints = cur_waypoints.reshape(-1, 2).numpy()

        cur_waypoints[:, 0] *= -1

        cur_waypoints = np.flip(cur_waypoints)

        waypoints[i] = cur_waypoints

    return waypoints


def draw_points_on_trajectory(
    canvas,
    points,
    color=(0, 255, 255),
    first_origin=None,
    pix_per_meter=5,
    skip_size=192,
    color_decay=0.85,
    radius=4,
):
    for i, point_list in enumerate(points):
        point_list = first_origin + point_list * pix_per_meter

        point_list[0] += i * skip_size

        for j, point in enumerate(point_list):
            cv2.circle(
                img=canvas,
                center=tuple(point.astype(np.int32)),
                radius=np.rint(radius).astype(int),
                color=tuple(c * (color_decay**j) for c in color),
                thickness=cv2.FILLED,
            )


def visualize_trajectory_waypoints(
    trajectory,
    save_dir,
    goal_type,
    action_type,
    waypoint_count=4,
    save_prefix="",
    save_suffix="",
    save_idx=None,
    pix_per_meter=5,
    croptype="front",
):
    if save_idx is not None:
        # print(trajectory.keys())
        bevs = trajectory["bev"][save_idx]
        goals = trajectory["goal"][save_idx]
        actions = trajectory["action"][save_idx]
    else:
        bevs = trajectory["bev"]
        goals = trajectory["goal"]
        actions = trajectory["action"]

    trajectory_length = bevs.shape[0]

    canvas = np.zeros((BEV_SIZE, BEV_SIZE * trajectory_length, 3), dtype=np.uint8)

    if croptype == "front":
        origin = (BEV_SIZE // 2, BEV_SIZE)
    else:
        origin = (BEV_SIZE // 2, BEV_SIZE // 2)

    waypoints = extract_waypoints_from_trajectory_actions(
        actions, action_type, waypoint_count=waypoint_count
    )

    target_points = extract_target_point_from_trajectory_goals(goals, goal_type)

    for i in range(trajectory_length):
        bev = as_rgb(bevs[i]).permute(1, 2, 0)

        # Convert to 0-255 uint8
        bev = (bev * 255).numpy().astype(np.uint8)

        # Add the image to the canvas
        canvas[:BEV_SIZE, i * BEV_SIZE : (i + 1) * BEV_SIZE, :] = bev

    if waypoints is not None:
        draw_points_on_trajectory(
            canvas=canvas,
            points=waypoints,
            first_origin=origin,
            pix_per_meter=pix_per_meter,
            skip_size=BEV_SIZE,
            color=(0, 255, 255),
            radius=BEV_SIZE / 100,
        )
    if target_points is not None:
        draw_points_on_trajectory(
            canvas=canvas,
            points=target_points.reshape(-1, 1, 2),
            first_origin=origin,
            pix_per_meter=pix_per_meter,
            skip_size=BEV_SIZE,
            color=(255, 0, 255),
            radius=BEV_SIZE / 50,
        )

    # save the canvas into save_dir
    save_folder = os.path.join(save_dir, "trajectory_waypoints")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = os.path.join(
        save_folder,
        "_".join([x for x in [save_prefix, "trajectory", save_suffix] if x]) + ".png",
    )

    cv2.imwrite(save_path, canvas)


asset_path_rel = "../../visualizations/assets"
assets_path = os.path.join(os.path.dirname(__file__), asset_path_rel)
images_dict = {
    "left": os.path.join(assets_path, "left.bmp"),
    "right": os.path.join(assets_path, "right.png"),
    "lanefollow": os.path.join(assets_path, "straight.bmp"),
    "leftlane": os.path.join(assets_path, "leftlane.bmp"),
    "rightlane": os.path.join(assets_path, "rightlane.bmp"),
    "straight": os.path.join(assets_path, "intersectionstraight.bmp"),
    "steer": os.path.join(assets_path, "steer.bmp"),
    "throttle": os.path.join(assets_path, "throttle.bmp"),
}

# check all images exist
for k, v in images_dict.items():
    if not os.path.isfile(v):
        raise FileNotFoundError(f"Could not find {v}")


# Load image function, if desired_size is None then we don't resize the image
def load_image(img, desired_size=None):
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    # If img is 4 dimensional, then use the alpha channel as all 3 channels
    if img.shape[-1] == 4:
        img[:, :, 0] = img[:, :, 3]
        img[:, :, 1] = img[:, :, 3]
        img[:, :, 2] = img[:, :, 3]
        img = img[:, :, :3]
    if desired_size is not None:
        img = cv2.resize(img, desired_size)

    return img


# From carla (We subtract 1)
# LEFT = 1
# RIGHT = 2
# STRAIGHT = 3
# LANEFOLLOW = 4
# CHANGELANELEFT = 5
# CHANGELANERIGHT = 6

BEV_SIZE = 192

highlevel_action_image_mapping = {
    0: load_image(images_dict["left"], (BEV_SIZE, BEV_SIZE)),
    1: load_image(images_dict["right"], (BEV_SIZE, BEV_SIZE)),
    2: load_image(images_dict["straight"], (BEV_SIZE, BEV_SIZE)),
    3: load_image(images_dict["lanefollow"], (BEV_SIZE, BEV_SIZE)),
    4: load_image(images_dict["leftlane"], (BEV_SIZE, BEV_SIZE)),
    5: load_image(images_dict["rightlane"], (BEV_SIZE, BEV_SIZE)),
}


def visualize_trajectory_highlevel_actions(
    trajectory,
    save_dir,
    save_prefix="",
    save_suffix="",
    save_idx=None,
):
    # Visualize BEVs and high-level actions

    if save_idx is not None:
        # print(trajectory.keys())
        bevs = trajectory["bev"][save_idx]
        highlevel_actions = trajectory["goal"][save_idx]
    else:
        bevs = trajectory["bev"]
        highlevel_actions = trajectory["goal"]

    trajectory_length = bevs.shape[0]

    canvas = np.zeros((BEV_SIZE * 2, BEV_SIZE * trajectory_length, 3), dtype=np.uint8)

    for i in range(trajectory_length):
        bev = as_rgb(bevs[i]).permute(1, 2, 0)

        # Convert to 0-255 uint8
        bev = (bev * 255).numpy().astype(np.uint8)

        highlevel_action = highlevel_actions[i]

        # Get the image
        image = highlevel_action_image_mapping[highlevel_action.item()]

        # Add the image to the canvas
        canvas[:BEV_SIZE, i * BEV_SIZE : (i + 1) * BEV_SIZE, :] = bev
        canvas[BEV_SIZE:, i * BEV_SIZE : (i + 1) * BEV_SIZE, :] = image

    # save the canvas into save_dir
    save_folder = os.path.join(save_dir, "trajectory")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = os.path.join(
        save_folder,
        "_".join([x for x in [save_prefix, "trajectory", save_suffix] if x]) + ".png",
    )

    cv2.imwrite(save_path, canvas)


# Convert to accel-steer
def convert_actions(action, action_types):
    action_types_sorted = sorted(action_types)
    steer = None
    if "steer" in action_types_sorted:
        steer = action[:, action_types.index("steer")]
    else:
        # Make 0s if no steer
        steer = np.zeros((action.shape[0], 1), dtype=action.dtype)

    accel = None
    if "acceleration" in action_types_sorted:
        accel = action[:, action_types.index("acceleration")]
    else:
        if "throttle" in action_types_sorted:
            accel = action[:, action_types.index("throttle")]

        if "brake" in action_types_sorted:
            if accel is None:
                accel = -action[:, action_types.index("brake")]
            else:
                accel = accel - action[:, action_types.index("brake")]

        if accel is None:
            # Make 0s if no accel
            accel = np.zeros((action.shape[0], 1), dtype=action.dtype)

    # [T], [T] -> [T, 2]
    return np.stack([accel, steer], axis=1)


def get_action_canvas(
    action,
    action_types,
    size=192,
    bev_canvas=None,
    copy_bev_canvas=False,
    bev_crop_type="front",
    pix_per_meter=5,
):
    if "waypoints" in action_types:
        # Draw the predicted waypoints
        return get_action_waypoints(
            action,
            action_types,
            size=size,
            bev_canvas=bev_canvas,
            copy_bev_canvas=copy_bev_canvas,
            bev_crop_type=bev_crop_type,
            pix_per_meter=pix_per_meter,
        )
    else:
        return get_action_arrows(action, action_types, size=size)


def get_action_waypoints(
    action,
    action_types,
    size=192,
    bev_canvas=None,
    copy_bev_canvas=False,
    bev_crop_type="front",
    pix_per_meter=5,
):
    if bev_canvas is None:
        bev_canvas = np.zeros((size, size * action.shape[0], 3), dtype=np.uint8)
    else:
        if copy_bev_canvas:
            bev_canvas = bev_canvas.copy()

    # For now, assert size of action is 8 and no other action types
    assert len(action_types) == 1 and action_types[0] == "waypoints"

    waypoints = extract_waypoints_from_trajectory_actions(
        action, action_types, waypoint_count=4
    )[
        -1:
    ]  # [4, 2]

    if bev_crop_type == "front":
        origin = (size // 2, size)
    else:
        origin = (size // 2, size // 2)

    # Draw the waypoints
    draw_points_on_trajectory(
        canvas=bev_canvas,
        points=waypoints,
        first_origin=origin,
        pix_per_meter=pix_per_meter,
        skip_size=size,
        radius=size / 100,
        color=(0, 0, 255),
    )

    if not copy_bev_canvas:
        return None
    return bev_canvas


def get_action_arrows(action, action_types, size=192):
    # Convert an array of 2d actions (acceleration, steering) to an array of arrows rasterized as images
    # Input: action: (T, 2) array of actions
    # Output: (size, T*size, 3) array of arrows
    # Each arrow starts at the center of the image and points in the direction of the action
    # Y axis is acceleration, X axis is steering
    # Positive acceleration is up, negative is down
    # Positive steering is right, negative is left
    # By arrows we mean quivers in matplotlib sense
    # Magnitude of the arrow is the magnitude of the action, which ranges from -1 to 1
    action = convert_actions(action, action_types)

    canvas = np.zeros((size, size * action.shape[0], 3), dtype=np.uint8)
    for i, a in enumerate(action):
        # Get the magnitude and direction of the action
        magnitude = np.linalg.norm(a)
        direction = np.arctan2(a[1], a[0])

        center = (size // 2 + i * size, size // 2)

        # Convert to pixel coordinates
        x = magnitude * np.sin(direction)
        y = magnitude * np.cos(direction)

        # Draw the arrow
        canvas = cv2.arrowedLine(
            canvas,
            center,
            (int(center[0] + x * size // 2), int(center[1] - y * size // 2)),
            (100, 100, 100),
            5,
        )

    return canvas


def get_goal_canvas(
    commands,
    command_types,
    size=192,
    color=(255, 255, 255),
    bev_canvas=None,
    copy_bev_canvas=False,
    bev_crop_type="front",
    pix_per_meter=5,
):
    if "command" in command_types:
        return get_command_canvas(commands, size=size, color=color)
    else:
        return get_targetpoint_canvas(
            commands,
            command_types,
            size=size,
            color=color,
            bev_canvas=bev_canvas,
            copy_bev_canvas=copy_bev_canvas,
            bev_crop_type=bev_crop_type,
            pix_per_meter=pix_per_meter,
        )


def get_targetpoint_canvas(
    commands,
    command_types,
    color=(255, 0, 255),
    size=192,
    bev_canvas=None,
    copy_bev_canvas=False,
    bev_crop_type="front",
    pix_per_meter=5,
):
    if bev_canvas is None:
        bev_canvas = np.zeros((size, size * commands.shape[0], 3), dtype=np.uint8)
    else:
        if copy_bev_canvas:
            bev_canvas = bev_canvas.copy()

    # For now, assert size of action is 8 and no other action types
    assert len(command_types) == 1 and command_types[0] == "target_point"

    target_points = extract_target_point_from_trajectory_goals(commands, command_types)

    if bev_crop_type == "front":
        origin = (size // 2, size)
    else:
        origin = (size // 2, size // 2)

    # Draw the target points
    draw_points_on_trajectory(
        canvas=bev_canvas,
        points=target_points.reshape(-1, 1, 2),
        first_origin=origin,
        pix_per_meter=pix_per_meter,
        skip_size=size,
        color=color,
        radius=size / 50,
    )

    if not copy_bev_canvas:
        return None
    return bev_canvas


def get_command_canvas(commands, size=192, color=(255, 255, 255)):
    canvas = np.zeros((size, size * commands.shape[0], 3))
    # Normalize color
    color = np.array(color) / 255.0
    for i, goal in enumerate(commands):
        img = highlevel_action_image_mapping[goal.item()]

        if size != img.shape[0]:
            img = cv2.resize(img, (size, size)) / 255.0

        canvas[:, i * size : (i + 1) * size, :] = img * color

    return canvas


def get_state_canvas(states, state_types, size=192):
    # Convert an array of states to an array of images
    # Input: action: (T, N) array of actions, where N is the number of handled state types
    # Output: (size*N, T*size, 3) array of arrows
    # If the state is lights, then the image is a circle with the color of the light
    # Since we only use whether there is a lights hazard or not, if the flag is 1
    # then the circle is red, otherwise it is green.
    # For the speed, the image is a number with the speed value. We make sure it fits in the image
    # First handle lights
    state_types = [x for x in sorted(state_types) if not "bev" in x]
    lights_canvas = None
    light_colors = None
    if "lights" in state_types:
        light_colors = []
        lights = states[:, state_types.index("lights")]
        for light in lights:
            if light == 1:
                light_colors.append((0, 0, 255))
            else:
                light_colors.append((0, 255, 0))

    # Then handle speed
    speed_canvas = None
    if "speed" in state_types:
        speed = states[:, state_types.index("speed")]
        speed_canvas = np.zeros((size, size * states.shape[0], 3), dtype=np.uint8)
        for i, s in enumerate(speed):
            # Draw the speed, truncate speed to 2 decimal places
            # First we get the size of the text
            text_size = cv2.getTextSize(f"{s:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

            center = (
                size // 2 + i * size - text_size[0] // 2,
                size // 2 - text_size[1] // 2,
            )

            speed_canvas = cv2.putText(
                speed_canvas,
                f"{s:.2f}",
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    to_contat = []
    if lights_canvas is not None:
        to_contat.append(lights_canvas)
    if speed_canvas is not None:
        to_contat.append(speed_canvas)

    if len(to_contat) == 0:
        return None

    return light_colors, np.concatenate(to_contat, axis=0)


def visualize_trajectory_action_predictions(
    batch,
    batch_outputs,
    save_dir=None,
    save_idx=0,
    action_source="transformer",
    *args,
    **kwargs,
):
    # Move batch to cpu
    # batch = {k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    # Copy batch shallowly
    batch = {k: v for k, v in batch.items()}
    if action_source == "transformer":
        if "output_dict" in batch_outputs:
            batch_outputs = batch_outputs["output_dict"]
        batch_outputs = {k: v for k, v in batch_outputs.items()}

        # Change the batch action to be the predicted action
        pred_actions = batch_outputs[TokenTypeIDs.ACTION].to(batch["action"].device)
    elif action_source == "gru":
        if "waypoints" in batch_outputs:
            pred_actions = batch_outputs["waypoints"].to(batch["action"].device)
        else:
            pred_actions = batch_outputs.to(batch["action"].device)
    else:
        raise ValueError(f"Unknown action source {action_source}")

    # pred_actions_len = pred_actions.shape[0]

    batch["action"] = torch.cat(
        (
            batch["action"],
            pred_actions.reshape(
                batch["action"].shape[0], -1, batch["action"].shape[-1]
            ),
        ),
        dim=1,
    )

    return visualize_trajectory(batch, save_dir, save_idx=save_idx, *args, **kwargs)


def visualize_trajectory(
    batch,
    save_dir=None,
    model=None,
    save_prefix="",
    save_suffix="",
    save_idx=None,
    action_type="steer-acceleration",
    state_type="speed-lights",
    goal_type="command",
    bev_crop_type="front",
    save_subfolder="trajectory-full",
    pix_per_meter=5,
    add_boundary=True,
    boundary_width=4,
    include_targets=True,
    batch_outputs=None,
):
    canvas, canvas_unit_size = get_bev_canvas(
        batch,
        save_idx,
        model,
        batch_outputs=batch_outputs,
        include_targets=include_targets,
    )

    canvas_to_reuse = canvas[:canvas_unit_size, :canvas_unit_size]

    actions = batch["action"][save_idx].cpu()

    action_canvas = get_action_canvas(
        actions,
        sorted(action_type.split("-")),
        size=canvas_unit_size,
        bev_canvas=canvas_to_reuse,
        copy_bev_canvas=True,
        bev_crop_type=bev_crop_type,
        pix_per_meter=pix_per_meter,
    )

    states = batch["state"][save_idx].cpu().numpy()

    light_colors, state_canvas = get_state_canvas(
        states, sorted(state_type.split("-")), size=canvas_unit_size
    )

    if "command" in goal_type:
        goal_color = {
            "command": (0, 155, 0),
            "highlevel_command": (155, 0, 155),
        }[goal_type]

        goal_canvas = get_command_canvas(
            batch["goal"][save_idx].cpu().numpy(),
            size=canvas_unit_size,
            color=goal_color,
        )
    elif "target_point" in goal_type:
        goal_canvas = get_targetpoint_canvas(
            batch["goal"][save_idx].cpu(),
            sorted(goal_type.split("-")),
            size=canvas_unit_size,
            color=(255, 0, 0),
            bev_canvas=action_canvas,
            copy_bev_canvas=False,
            bev_crop_type=bev_crop_type,
            pix_per_meter=pix_per_meter,
        )
    else:
        goal_canvas = None

    to_concat = [
        x for x in [canvas, action_canvas, goal_canvas, state_canvas] if x is not None
    ]

    canvas = np.concatenate(to_concat, axis=0)

    if add_boundary or light_colors is not None:
        if light_colors is None:
            light_colors = [(255, 0, 255)] * (canvas.shape[0] // canvas_unit_size)
        boundary_width_half = boundary_width // 2

        for x in range(0, canvas.shape[0], canvas_unit_size):
            for clr, y in zip(
                light_colors, range(0, canvas.shape[1], canvas_unit_size)
            ):
                canvas[
                    x : x + boundary_width_half,
                    y : y + canvas_unit_size,
                ] = clr

                canvas[
                    max(x - boundary_width_half + canvas_unit_size, 0) : x
                    + canvas_unit_size,
                    y : y + canvas_unit_size,
                ] = clr

                canvas[
                    x : x + canvas_unit_size,
                    y : y + boundary_width_half,
                ] = clr

                canvas[
                    x : x + canvas_unit_size,
                    max(y - boundary_width_half + canvas_unit_size, 0) : y
                    + canvas_unit_size,
                ] = clr

    if save_dir is not None:
        # save the canvas into save_dir
        save_folder = os.path.join(save_dir, save_subfolder)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(
            save_folder,
            "_".join([x for x in [save_prefix, "trajectory", save_suffix] if x])
            + ".png",
        )

        cv2.imwrite(save_path, canvas)

    return canvas


def visualize_slot_preds_with_actions(
    batch,
    batch_outputs,
    model,
    save_dir=None,
    save_idx=0,
    save_prefix="",
    save_suffix="",
    action_source="transformer",
    include_targets=False,
    action_type="steer-acceleration",
    state_type="speed-lights",
    goal_type="command",
    bev_crop_type="front",
    save_subfolder="trajectory-full",
    pix_per_meter=5,
    *args,
    **kwargs,
):
    gt = batch["bevslots"]
    gt_objs = batch["bevobject"]
    # gt_obj_ids = batch["bevobjecttype"]
    # gt_reproduced = None
    slotlatents = model.bev_encoder.encode_slots(gt)

    T = gt_objs.shape[1]

    src_slotlatents = slotlatents[:, :T]
    src_slots = model.bev_encoder.interpret_slots(src_slotlatents)

    gt_reproduced_bins = get_binary_mask_from_rgb(src_slots)
    gt_rgb = gt[save_idx].cpu()[
        model.bev_encoder.test_time_context : model.bev_encoder.test_time_context + 1
    ]
    gt_rgb_bins = get_binary_mask_from_rgb(gt_rgb)

    preds = batch_outputs["output_dict"][1]
    preds_rgb = model.bev_encoder.interpret_slots(preds)

    preds_bin = get_binary_mask_from_rgb(preds_rgb.cpu())

    if include_targets and not slotlatents.shape[1] == T:
        tgt_slotlatents = slotlatents[:, -T:]
        tgt_slots = model.bev_encoder.interpret_slots(tgt_slotlatents)
        tgt_slots_bin = get_binary_mask_from_rgb(tgt_slots)

    batch = {k: v for k, v in batch.items()}
    if action_source == "transformer":
        if "output_dict" in batch_outputs:
            batch_outputs = batch_outputs["output_dict"]
        batch_outputs = {k: v for k, v in batch_outputs.items()}

        # Change the batch action to be the predicted action
        pred_actions = batch_outputs[TokenTypeIDs.ACTION].to(batch["action"].device)
    elif action_source == "gru":
        if "waypoints" in batch_outputs:
            pred_actions = batch_outputs["waypoints"].to(batch["action"].device)
        else:
            pred_actions = batch_outputs.to(batch["action"].device)
    else:
        raise ValueError(f"Unknown action source {action_source}")

    # pred_actions_len = pred_actions.shape[0]

    batch["action"] = torch.cat(
        (
            batch["action"],
            pred_actions.reshape(
                batch["action"].shape[0], -1, batch["action"].shape[-1]
            ),
        ),
        dim=1,
    )

    bevcanvas = save_bins_to_image_nomiou(
        [
            gt_rgb_bins.unsqueeze(0),
            gt_reproduced_bins.unsqueeze(0),
            preds_bin.unsqueeze(0),
        ],
        save_dir,
        return_image=True,
    )

    canvas_to_reuse = bevcanvas[: bevcanvas.shape[0], -bevcanvas.shape[0] :, :]

    actions = batch["action"][save_idx].cpu()

    get_action_canvas(
        actions,
        sorted(action_type.split("-")),
        size=192 * 2,
        bev_canvas=canvas_to_reuse,
        copy_bev_canvas=False,
        bev_crop_type=bev_crop_type,
        pix_per_meter=pix_per_meter * 2,
    )

    # import ipdb

    # ipdb.set_trace()
    impath = os.path.join(
        save_dir,
        f"{save_prefix}.png",
    )

    # Save as epoch_{epoch}.png in the log directory
    cv2.imwrite(
        impath,
        bevcanvas,
    )


def get_bev_canvas(
    batch,
    batch_idx,
    model=None,
    batch_outputs=None,
    labels=None,
    include_targets=True,
):
    if model is not None:
        bev_mode = "bev" if not model.cfg.training["object_level"] else "bevobject"
        if model.cfg.training["object_level"] and model.cfg.training.get(
            "use_slots", False
        ):
            bev_mode = "bevslots"
    else:
        if "bev" in batch:
            bev_mode = "bev"
        elif "bevslots" in batch:
            bev_mode = "bevslots"
        elif "bevobject" in batch:
            bev_mode = "bevobject"
        else:
            raise ValueError("No bev in batch")
    # print("BEVVING IN BEV MODE: ", bev_mode)
    targets_rgb = None

    if bev_mode == "bev":
        gt = batch["bev"]
        gt_reproduced = None
        if model is not None:
            gt_reproduced = (
                model.bev_encoder.interpret(model.bev_encoder.encode(gt[batch_idx]))
                .flatten(0, 1)  # T dimension
                .permute(2, 0, 3, 1)
                .cpu()
            )

        gt = gt[batch_idx].cpu()

        gt_rgb = as_rgb(gt > 0).permute(2, 0, 3, 1)
        gt_rgb = gt_rgb.reshape(gt_rgb.shape[0], -1, gt_rgb.shape[-1])

        if batch_outputs is not None:
            preds = batch_outputs["bev"]
            preds = preds[batch_idx].cpu()
            preds_rgb = as_rgb(preds > 0).permute(2, 0, 3, 1)
            preds_rgb = preds_rgb.reshape(preds_rgb.shape[0], -1, preds_rgb.shape[-1])
        else:
            preds_rgb = None
    elif bev_mode == "bevslots":
        # import ipdb; ipdb.set_trace()  # fmt:skip
        gt = batch["bevslots"]
        gt_objs = batch["bevobject"]
        gt_obj_ids = batch["bevobjecttype"]
        gt_reproduced = None
        if model is not None:
            slotlatents = model.bev_encoder.encode_slots(gt[batch_idx].unsqueeze(0))

            T = gt_objs.shape[1]

            src_slotlatents = slotlatents[:, :T]
            src_slots = model.bev_encoder.interpret_slots(src_slotlatents)

            gt_reproduced = (
                src_slots.flatten(0, 1).permute(2, 0, 3, 1).cpu()  # T dimension
            )
            gt_rgb = gt[batch_idx].cpu()[
                model.bev_encoder.test_time_context : model.bev_encoder.test_time_context
                + T
            ]

            gt_rgb = gt_rgb.permute(2, 0, 3, 1)
            gt_reproduced = gt_reproduced.flatten(1, -2)
            gt_rgb = gt_rgb.flatten(1, -2)
            if include_targets and not slotlatents.shape[1] == T:
                tgt_slotlatents = slotlatents[:, -T:]
                tgt_slots = model.bev_encoder.interpret_slots(tgt_slotlatents)
                targets_rgb = (
                    tgt_slots.flatten(0, 1).permute(2, 0, 3, 1).cpu()
                )  # T dimension
                targets_rgb = targets_rgb.flatten(1, -2)
                targets_rgb = (targets_rgb * 0.5 + 0.5).clamp(0, 1)
            gt_rgb = (gt_rgb * 0.5 + 0.5).clamp(0, 1)
            gt_reproduced = (gt_reproduced * 0.5 + 0.5).clamp(0, 1)
        else:
            gt_rgb = np.zeros((192, 192 * len(gt[batch_idx]), 3))

        if batch_outputs is not None:  # TODO: Fix this
            preds_rgb = None
            # preds = batch_outputs["bevslots"]
            # preds = preds[batch_idx].cpu()
            # preds_rgb = as_rgb(preds > 0).permute(2, 0, 3, 1)
            # preds_rgb = preds_rgb.reshape(preds_rgb.shape[0], -1, preds_rgb.shape[-1])
        else:
            preds_rgb = None
    else:
        # TODO: Fix this
        gt = batch["bevobject"]
        gt_types = batch["bevobjecttype"]

        if "targetbevobject" in batch:
            gt_labels = batch["targetbevobject"]
            gt_label_types = batch["targetbevobjecttype"]
        else:
            gt_labels = None
            gt_label_types = None

        if model is not None:
            gt_rgb = (
                model.bev_encoder.interpret(gt[batch_idx], gt_types[batch_idx])
                .squeeze(1)  # T dimension
                .transpose(
                    (1, 0, 2, 3)
                )  # Because last dimension is channels (3) unlike bev
            ).astype(np.float32) / 255.0

            if gt_labels is not None:
                gt_label_rgb = (
                    model.bev_encoder.interpret(
                        gt_labels[batch_idx], gt_label_types[batch_idx]
                    )
                    .squeeze(1)  # T dimension
                    .transpose(
                        (1, 0, 2, 3)
                    )  # Because last dimension is channels (3) unlike bev
                ).astype(np.float32) / 255.0

                # Multiply all channels by 0.5
                gt_label_rgb *= 0.5

                # Wherever gt_rgb is 0, make it gt_label_rgb
                gt_rgb = gt_rgb * (gt_rgb > 0) + gt_label_rgb * (gt_rgb == 0)
        else:
            gt_rgb = np.zeros((192, 192 * len(gt[batch_idx]), 3))
        gt_rgb = gt_rgb.reshape(gt_rgb.shape[0], -1, gt_rgb.shape[-1])

        if model is not None:
            pass
        # gt_reproduced = np.zeros_like(gt_rgb)
        # We always have gt_reproduced as None regardless of whether we have model or not (for now)
        gt_reproduced = None
        if batch_outputs is not None:
            # TODO: Future vehicle predictions
            preds_rgb = np.zeros_like(gt_rgb)
        else:
            preds_rgb = None

    if gt_reproduced is not None:
        gt_reproduced = gt_reproduced.reshape(
            gt_reproduced.shape[0], -1, gt_reproduced.shape[-1]
        )

    # print("[gt_rgb, gt_reproduced, targets_rgb, preds_rgb]")
    # for x in [gt_rgb, gt_reproduced, targets_rgb, preds_rgb]:
    #     print(x.shape if x is not None else None)

    rows = [x for x in [gt_rgb, gt_reproduced, targets_rgb, preds_rgb] if x is not None]
    # for row in rows:
    #     print(row.min())
    #     print(row.max())
    #     print(row.shape)
    canvas_unit_size = gt_rgb.shape[0]

    canvas = np.zeros((canvas_unit_size * len(rows), gt_rgb.shape[1], 3))
    # canvas = np.zeros((canvas_unit_size * 3, gt_rgb.shape[1], 3))

    # canvas[: canvas_unit_size, :, :] = gt_rgb
    # canvas[canvas_unit_size : canvas_unit_size * 2, :, :] = gt_reproduced
    # canvas[canvas_unit_size * 2 :, :, :] = preds_rgb
    for i, row in enumerate(rows):
        canvas[i * canvas_unit_size : (i + 1) * canvas_unit_size, :, :] = row

    # Convert canvas of 0-1 floats to 0-255 uint8
    canvas = (canvas * 255).astype(np.uint8)
    # Rgb to bgr
    canvas = canvas[:, :, ::-1]

    return canvas, canvas_unit_size


def visualize_input_from_batch(
    batch,
    batch_idx,
    batch_outputs,
    labels,
    save_dir,
    save_affix,
    model,
    save_prefix,
):
    canvas, img_size = get_bev_canvas(
        batch=batch,
        batch_idx=batch_idx,
        model=model,
        batch_outputs=batch_outputs,
        labels=labels,
    )

    impath = os.path.join(
        save_dir,
        f"{save_prefix}_predictions",
        "epoch_{}.png".format(save_affix),
    )

    # Save as epoch_{epoch}.png in the log directory
    cv2.imwrite(
        impath,
        canvas,
    )

    save_bin_probs(
        batch_outputs["deinterleaved_outputs"],
        labels,
        save_dir,
        model.cfg.quantization_offset_map,
        model.cfg.quantization_vocab_size_map,
        [model.action_quantizer, model.state_quantizer, model.reward_quantizer],
        save_idx=batch_idx,
        save_prefix=save_prefix,
        save_suffix=f"{save_affix}",
    )

    return impath


def get_bev_kind(batch, model=None):
    if model is not None:
        bev_mode = "bev" if not model.cfg.training["object_level"] else "bevobject"
        if model.cfg.training["object_level"] and model.cfg.training["use_slots"]:
            bev_mode = "bevslots"
    else:
        if "bev" in batch:
            bev_mode = "bev"
        elif "bevslots" in batch:
            bev_mode = "bevslots"
        elif "bevobject" in batch:
            bev_mode = "bevobject"
        else:
            raise ValueError("No bev in batch")


def print_embedding_counters(
    embedding_counter,
    token_type_mapping,
    sorted_by_value=True,
):
    # embedding_counter: An EmbeddingCounter instance
    # token_type_mapping: dict(token_type -> {"quantizer":quantizer, "offset":offset}))
    # sorted_by_value: bool, if True, sort the tokens by their counts

    # Iterate over the token types
    print("=" * 25)
    print("Embedding counts:")

    for split in ["training", "validation"]:
        counts = (
            getattr(embedding_counter, f"get_{split}_embedding_counts")().cpu().numpy()
        )

        print(f"{split.capitalize()}:")
        for token_type in token_type_mapping.keys():
            params = token_type_mapping[token_type]
            quantizer = params["quantizer"]
            offset = params["offset"]
            width = len(quantizer)
            for dim_idx in range(width):
                dim_name = quantizer.get_attribute_name(dim_idx)
                print(dim_idx, dim_name)
                token_id_begin, token_id_end = quantizer.get_dim_boundaries(dim_idx)

                counts_slice = counts[offset + token_id_begin : offset + token_id_end]

                if sorted_by_value:
                    sorting_index = quantizer.get_centroid_sorting_indices(dim_idx)
                    print(sorting_index.shape, counts_slice.shape, sorting_index)
                    counts_slice = counts_slice[sorting_index]

                print(
                    f"{dim_name}:".capitalize(),
                    f"\nValues: \t{quantizer.get_centroids(dim_idx)[sorting_index]}",
                    f"\nCounts: \t{counts_slice}",
                )

    print("=" * 25)


def get_probs_and_gt(
    deinterleaved_logits,
    ground_truth_labels,
    offset_map,
    vocab_size_map,
    token_quantizers,
    token_ids=[TokenTypeIDs.ACTION],
):
    result = {}
    for token_id, token_quantizer in zip(token_ids, token_quantizers):
        assert token_id in offset_map
        assert token_id in vocab_size_map

        result[token_id] = {}

        token_width = len(token_quantizer)

        if token_id == TokenTypeIDs.ACTION:
            token_width = len(token_quantizer.attribute_names)

        for width_idx in range(token_width):
            width_dim = token_quantizer.get_dim_width(width_idx)

            start_idx_offset, end_idx_offset = token_quantizer.get_dim_boundaries(
                width_idx
            )

            # Resort the probs and labels according to the quantizer centroids
            centroids = token_quantizer.concatenated_centroids[
                start_idx_offset:end_idx_offset
            ]

            sorted_centroid_idx = np.argsort(centroids.flatten())

            if token_width > 1:
                name = token_quantizer.get_attribute_name(width_idx)
            else:
                name = token_id.name.lower()

            # First get the ground truth labels
            label_key = f"quantized_label_{token_id.name.lower()}"

            pred_scores = deinterleaved_logits[token_id][
                :,
                width_idx::token_width,
                offset_map[token_id]
                + start_idx_offset : offset_map[token_id]
                + end_idx_offset,
            ].softmax(dim=-1)

            gt_labels = (
                ground_truth_labels[label_key][:, width_idx::token_width]
                - offset_map[token_id]
                - start_idx_offset
            )

            pred_scores = pred_scores.flatten(end_dim=-2)
            gt_labels = gt_labels.reshape(-1)

            # Make one hot
            gt_labels = torch.nn.functional.one_hot(
                gt_labels.long(), num_classes=width_dim
            )

            gt_labels = gt_labels[:, sorted_centroid_idx]

            # Get gt_labels by argmaxing
            gt_labels = gt_labels.argmax(dim=-1)

            gt_labels = gt_labels.cpu().numpy()

            pred_scores = (
                pred_scores[:, sorted_centroid_idx].argmax(dim=-1).cpu().numpy()
            )

            pred_val_labels = [
                f"{token_quantizer.concatenated_centroids.flatten()[j + start_idx_offset]:.1f}"
                for j in sorted_centroid_idx
            ]

            result[token_id][name] = {
                "labels": gt_labels,
                "preds": pred_scores,
                "pred_val_labels": pred_val_labels,
            }

    return result


from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff


# Use plotly to create a confusion matrix heatmap
import plotly.express as px


def create_confusion_matrix(preds, labels, axis_labels, confusion_matrix_name):
    matrix = (
        confusion_matrix(
            labels, preds, labels=np.arange(len(axis_labels)), normalize="true"
        )
        * 100
    )

    x = axis_labels
    y = axis_labels

    # change each element of z to type string for annotations
    # z_text = [[str(y) for y in x] for x in matrix]

    fig = px.imshow(
        matrix,
        x=x,
        y=y,
        # color_continuous_scale="bluered",
        aspect="auto",
        labels={
            "x": f"Pred:",
            "y": "Real:",
            "color": "% of preds",
        },
        width=800,
        height=600,
    )
    fig.update_xaxes(side="top", tickangle=90)

    # fig.show()
    return fig
