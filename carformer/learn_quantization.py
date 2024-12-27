from quantizer import KMeansQuantizer
import numpy as np
import torch
import argparse
from data import SequenceDataset, PlantSequenceDataset
import os
from tqdm import tqdm
import json
from carformer.backbone import gpt2
import sys


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        # "--input_folder", type=str, default="/home/shadi/data/iteration_one"
        "--input_folder",
        type=str,
        default="/home/shadi/hdd/data/iteration_four_overfit_v2_nolights",
    )
    parser.add_argument(
        "--config", type=str, default="configs/gpt2q-tok-mini-config.json"
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=20,
        help="Maximum length of a sequence in terms of transitions",
    )
    parser.add_argument(
        "--state_type",
        type=str,
        default="speed-lights",
        help="Type of states to use, seperated by -. Possible values: bev, bevbin, speed, bevobject, tlhazard",
    )
    parser.add_argument(
        "--action_type",
        type=str,
        default="steer-acceleration",
        help="Type of actions to use, seperated by -. Possible values: steer, acceleration, throttle, break, waypoints",
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        default="reward",
        help="Type of reward to use",
        choices=[
            "all_reward",
            "speed_reward",
            "steering_reward",
            "lane_dist_reward",
            "route_angle_reward",
        ],
    )
    parser.add_argument(
        "--goal_type",
        type=str,
        default="highlevel_command",
        help="Type of actions to use, seperated by -. Possible values: highlevel_command, command, pooled_command, target_point",
    )
    parser.add_argument(
        "--integrate-rewards-to-go",
        action="store_true",
        help="Inject rewards to go into the reward as a heuristic for use during planning",
    )
    # Num epochs
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)

    # Overfit on a single batch
    parser.add_argument("--overfit", action="store_true")

    parser.add_argument("--moving_window_size", type=int, default=2)

    parser.add_argument("--subsample_ratio", type=float, default=1.0)

    parser.add_argument("--plant_data", action="store_true")

    # Quantization arguments
    parser.add_argument(
        "--action_quantization_type",
        type=str,
        default="dim",
        help="Type of action quantization to use",
        choices=["whole", "dim"],
    )
    parser.add_argument(
        "--action_quantization_num_classes",
        type=int,
        default=32,
        help="Number of classes to quantize actions to",
    )
    parser.add_argument(
        "--action_quantization_dim_widths",
        help="Widths of each dimension to quantize actions to. Should be a list of integers of length equal to the number of dimensions in the action space",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--state_quantization_type",
        type=str,
        default="dim",
        help="Type of state quantization to use",
        choices=["whole", "dim"],
    )
    parser.add_argument(
        "--state_quantization_num_classes",
        type=int,
        default=16,
        help="Number of classes to quantize states to",
    )
    parser.add_argument(
        "--state_quantization_dim_widths",
        help="Widths of each dimension to quantize actions to. Should be a list of integers of length equal to the number of dimensions in the state space",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--reward_quantization_type",
        type=str,
        default="dim",
        help="Type of reward quantization to use",
        choices=["whole", "dim"],
    )
    parser.add_argument(
        "--reward_quantization_num_classes",
        type=int,
        default=16,
        help="Number of classes to quantize rewards to",
    )
    parser.add_argument(
        "--reward_quantization_dim_widths",
        help="Widths of each dimension to quantize actions to. Should be a list of integers of length equal to the number of dimensions in the reward space",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--goal_quantization_type",
        type=str,
        default="dim",
        help="Type of goal quantization to use",
        choices=["whole", "dim"],
    )
    parser.add_argument(
        "--goal_quantization_num_classes",
        type=int,
        default=16,
        help="Number of classes to quantize goals to",
    )
    parser.add_argument(
        "--goal_quantization_dim_widths",
        help="Widths of each dimension to quantize actions to. Should be a list of integers of length equal to the number of dimensions in the goal space",
        nargs="+",
        type=int,
        default=None,
    )

    # Save paths
    parser.add_argument(
        "--quantizer_save_path",
        type=str,
        default="bin/testquantizers",
        help="Path to save quantizer to",
    )

    parser.add_argument(
        "--flatten_waypoints",
        action="store_true",
        help="Flatten waypoints to be lists of 2D points instead of lists of N 2D points",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    data_module = SequenceDataset if not args.plant_data else PlantSequenceDataset

    dataset = data_module(
        args.input_folder,
        args.state_type,
        args.action_type,
        args.reward_type,
        args.goal_type,
        args.integrate_rewards_to_go,
        args.max_len,
        split="train" if not args.overfit else "val",
        max_instances=args.batch_size if args.overfit else None,
        frame_stride=args.moving_window_size,
    )

    os.makedirs(args.quantizer_save_path, exist_ok=True)

    # Quantize actions
    # Dataloader to load in batches
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=6
    )

    config = {
        "data_config": {
            "state_type": args.state_type,
            "action_type": args.action_type,
            "reward_type": args.reward_type,
            "input_folder": args.input_folder,
            "max_len": args.max_len,
            "goal_type": args.goal_type,
            "integrate_rewards_to_go": args.integrate_rewards_to_go,
            "moving_window_size": args.moving_window_size,
            "subsample_ratio": args.subsample_ratio,
            "plant_data": args.plant_data,
        },
        "quantization_config": {
            "action_quantization_type": args.action_quantization_type,
            "action_quantization_num_classes": args.action_quantization_num_classes,
            "action_quantization_dim_widths": args.action_quantization_dim_widths,
            "state_quantization_type": args.state_quantization_type,
            "state_quantization_num_classes": args.state_quantization_num_classes,
            "state_quantization_dim_widths": args.state_quantization_dim_widths,
            "reward_quantization_type": args.reward_quantization_type,
            "reward_quantization_num_classes": args.reward_quantization_num_classes,
            "reward_quantization_dim_widths": args.reward_quantization_dim_widths,
            "goal_quantization_type": args.goal_quantization_type,
            "goal_quantization_num_classes": args.goal_quantization_num_classes,
            "goal_quantization_dim_widths": args.goal_quantization_dim_widths,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "flatten_waypoints": args.flatten_waypoints,
        },
        "run_command": " ".join(["python"] + sys.argv),
    }

    with open(os.path.join(args.quantizer_save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    statistics = {}

    # Get all as batches
    for key in tqdm(["action", "state", "reward", "goal"], desc="Quantizing"):
        items = []
        for batch in tqdm(
            dataset_loader, desc="Loading data", total=args.max_batches, leave=False
        ):
            if key == "action" and args.flatten_waypoints:
                assert (
                    args.action_type == "waypoints"
                ), "Flatten waypoints flag used but action type is not exclusively waypoints"
                batch[key] = batch[key].reshape(-1, 2)

            items.append(batch[key])
            if len(items) > args.max_batches:
                break

        quantizer = KMeansQuantizer(
            getattr(args, f"{key}_quantization_num_classes"),
            getattr(args, f"{key}_quantization_type"),
            getattr(args, f"{key}_quantization_dim_widths", None),
        )
        quantizer.train(items)

        avg_error = 0

        max_error = 0
        for item in items:
            quantized = quantizer.decode(quantizer.encode(item.numpy()))

            avg_error += torch.mean(torch.abs(item - quantized)).item() / len(items)
            max_error = max(torch.max(torch.abs(item - quantized)).item(), 0)

        print("Average error for", key, "is", avg_error)
        print("Max error for", key, "is", max_error)

        statistics[key] = {
            "avg_error": avg_error,
            "max_error": max_error,
            "num_classes": getattr(args, f"{key}_quantization_num_classes"),
            "type": getattr(args, f"{key}_quantization_type"),
            "dim_widths": getattr(args, f"{key}_quantization_dim_widths", None),
        }

        np.save(
            os.path.join(args.quantizer_save_path, f"{key}_quantizer.npy"), quantizer
        )

    json.dump(
        statistics,
        open(os.path.join(args.quantizer_save_path, "stats.json"), "w"),
        indent=4,
    )
