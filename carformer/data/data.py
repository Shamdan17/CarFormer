from time import time
import numpy as np
import torch
import os
from .data_parser import Parser
import hashlib
from tqdm import trange
from carla_birdeye_view import (
    BirdViewProducerObjectLevelRenderer,
    PixelDimensions,
    BirdViewCropType,
)
from .data_utils import (
    postprocess_bev_objects,
    get_object_level_filter,
    transform_waypoints,
)
from .plant_data import PlantSequenceDataset


class Timestep:
    # @profile
    def __init__(
        self,
        parser,
        index,
        preprocessing_functions=None,
        filtering_functions=None,
        include_noise=True,
    ):
        self.index = index
        self.parser = parser

        self.goal = self.parser.get_goal(self.index)
        self.state = self.parser.get_state(
            self.index,
            preprocessing_functions=preprocessing_functions,
            filtering_functions=filtering_functions,
        )

        if include_noise:
            self.action, self.noisy = self.parser.get_action(
                self.index, include_noise=True
            )
        else:
            self.action = self.parser.get_action(self.index)

        self.reward = self.parser.get_reward(self.index)

        # No done yet, use is_terminal?
        # self.done = self.parser.get_done(self.index)

    def __repr__(self):
        return f"Timestep(index={self.index}, goal={self.goal}, state={self.state}, action={self.action}, reward={self.reward})"


class TimeStepDataset(torch.utils.data.Dataset):
    # TODO: Replace kwargs with config
    def __init__(
        self,
        dataset_path,
        parser,
        config,
        preprocessing_functions=None,
        filtering_functions=None,
        throw_error_if_not_enough_timesteps=True,
    ):
        self.dataset_path = dataset_path
        self.parser = parser
        self.integrate_rewards_to_go = integrate_rewards_to_go
        self.sequence_length = sequence_length
        self.future_horizon = future_horizon
        self._round_func = np.floor if drop_last else np.ceil
        self.frame_stride = frame_stride
        self.trim_first_and_last = trim_first_and_last
        self.trim_count = trim_count if trim_first_and_last else 0
        if self.trim_first_and_last:
            if self.parser.get_size() > self.trim_count * 2:
                self.num_steps = (
                    self.parser.get_size() - 2 * self.trim_count
                ) // self.frame_stride
            else:
                self.num_steps = 0
        else:
            self.num_steps = self.parser.get_size() // self.frame_stride
        self.length = int(
            self._round_func(self.num_steps / (sequence_length + self.future_horizon))
        )
        self.preprocessing_functions = preprocessing_functions
        self.filtering_functions = filtering_functions
        self.throw_error_if_not_enough_timesteps = throw_error_if_not_enough_timesteps
        self.use_future_ego_waypoints = use_future_ego_waypoints

    def __len__(self):
        # return min(200, self.length)
        return self.length

    # @profile
    def __getitem__(self, idx):
        # Get timestep
        idx_start = idx * self.sequence_length + self.trim_count

        # idx_end = min(idx_start + self.sequence_length + self.future_horizon, self.num_steps)
        idx_end = idx_start + self.sequence_length + self.future_horizon

        if idx_end > self.num_steps:
            if not self.throw_error_if_not_enough_timesteps:
                raise IndexError(
                    f"Index {idx_end} out of bounds for dataset of size {self.num_steps}"
                )
            else:
                idx_end = self.num_steps

        timesteps = [
            Timestep(
                self.parser,
                idx * self.frame_stride,
                self.preprocessing_functions,
                self.filtering_functions,
            )
            for idx in range(idx_start, idx_end)
        ]

        if self.integrate_rewards_to_go:
            timesteps[-1].reward["rewards_to_go"] = 0
            for i in range(len(timesteps) - 2, -1, -1):
                timesteps[i].reward["rewards_to_go"] = (
                    timesteps[i + 1].reward["rewards_to_go"]
                    + timesteps[i + 1].reward["reward"]
                )

        if self.use_future_ego_waypoints:
            assert len(timesteps) == self.sequence_length + self.future_horizon

            # waypoints = np.zeros(
            #     (self.sequence_length, self.future_horizon * 2)
            # )  # Each waypoint is 2D
            for i in range(self.sequence_length):
                ego_states = [
                    t.action["ego_matrix"]
                    for t in timesteps[i : i + self.future_horizon + 1]
                ]
                # Skip first one because it is the current ego matrix
                transformed_ego_matrices = transform_waypoints(ego_states)[1:]

                waypoints = np.stack(transformed_ego_matrices, axis=0)[
                    :, :2, -1
                ].flatten()

                timesteps[i].action["waypoints"] = waypoints

            # Remove ego_matrix from state
            for i in range(self.sequence_length):
                timesteps[i].action.pop("ego_matrix", None)

        if self.future_horizon > 0:
            timesteps = timesteps[: -self.future_horizon]

        return timesteps

    def __getweight__(self, idx, reduce="mean"):
        idx_start = idx * self.sequence_length + self.trim_count

        idx_end = min(idx_start + self.sequence_length, self.num_steps)

        weights = [
            self.parser.get_weight(idx * self.frame_stride)
            for idx in range(idx_start, idx_end)
        ]

        if reduce == "mean":
            return np.mean(weights)
        elif reduce == "sum":
            return np.sum(weights)
        else:
            raise NotImplementedError(f"Reduce method {reduce} not implemented")

    # def init_noisy(self):
    #     size = self.parser.get_size()
    #     self.noisy_array = np.zeros(size, dtype=bool)
    #     for i in trange(size-1, 0, -1, desc="Skipping noisy sequences"):
    #         test = ""

    def __getnoisy__(self, idx, reduce="any"):
        idx_start = idx * self.sequence_length + self.trim_count

        idx_end = min(idx_start + self.sequence_length, self.num_steps)

        # if not hasattr(self, "noisy_array"):
        #     self.init_noisy()

        # To skip computing noisy for all timesteps, we can just return the last one if reduce is last
        if reduce == "last":
            return self.parser.get_noisy((idx_end - 1) * self.frame_stride)

        noisies = [
            self.parser.get_noisy(idx * self.frame_stride)
            for idx in range(idx_start, idx_end)
        ]

        if reduce == "any":
            return any(noisies)
        elif reduce == "all":
            return all(noisies)
        else:
            raise NotImplementedError(f"Reduce method {reduce} not implemented")

    def __repr__(self):
        return f"TimeStepDataset(dataset_path={self.dataset_path}, parser={self.parser}, integrate_rewards_to_go={self.integrate_rewards_to_go}, sequence_length={self.sequence_length}, drop_last={self._round_func == np.ceil}, frame_stride={self.frame_stride})"


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        split,
        config=None,
    ):
        split
        self.dataset_path = dataset_path
        self.object_level_max_route_length = config.object_level_max_route_length
        self.object_level_max_num_objects = config.object_level_max_num_objects
        self.include_traffic_lights_in_object_level = (
            config.include_traffic_lights_in_object_level
        )
        self.split_large_routes = config.split_long_routes
        self.split_threshold = config.split_threshold
        self.use_future_ego_waypoints = "waypoints" in config.action_type
        self.future_horizon = config.future_horizon
        self.trim_first_and_last = config.trim_first_and_last
        self.trim_count = config.trim_count

        if split == "train":
            self.routes = sorted(os.listdir(dataset_path))[:-6]
        elif split == "val":
            self.routes = sorted(os.listdir(dataset_path))[-6:-3]
        elif split == "test":
            self.routes = sorted(os.listdir(dataset_path))[-3:]

        x_min = int(-config.bev_crop_size / 2 + config.bev_size / 2)
        x_max = int(config.bev_crop_size / 2 + config.bev_size / 2)
        y_min = int(
            -config.bev_crop_size / (1 if config.bev_crop == "front" else 2)
            + bev_size / 2
        )
        y_max = int(
            config.bev_crop_size * (0 if config.bev_crop == "front" else 0.5)
            + bev_size / 2
        )

        bev_crop_func = (
            lambda x: x[:, y_min:y_max, x_min:x_max]
            if (config.bev_crop == "center")
            else x[:, y_min:y_max, x_min:x_max]
        )
        bev_preprocessing_function = (
            lambda x: np.delete(bev_crop_func(x), 2, axis=0)
            if x.shape[0] == 9
            else bev_crop_func(x)
        )

        obj_filter = get_object_level_filter(
            bev_size,
            bev_pixels_per_meter,
            bev_crop,
            split_long_routes,
            split_threshold,
            include_agent_in_object_level,
            include_traffic_lights_in_object_level,
            object_level_max_num_objects,
            object_level_max_route_length,
            sort_by_distance=True,
        )

        preprocessing_functions = {
            "bev": bev_preprocessing_function,
        }

        filtering_functions = {
            "bevobject": obj_filter,
        }

        self.route_data_parsers = [
            Parser(
                os.path.join(dataset_path, route),
                state_type,
                action_type,
                reward_type,
                goal_type,
            )
            for route in self.routes
        ]

        self.route_datasets = [
            TimeStepDataset(
                os.path.join(dataset_path, route),
                parser,
                config,
                preprocessing_functions=preprocessing_functions,
                filtering_functions=filtering_functions,
            )
            for route, parser in zip(self.routes, self.route_data_parsers)
        ]

        self.routes_num_sequences = [len(dataset) for dataset in self.route_datasets]

        self.total_num_sequences = np.sum(self.routes_num_sequences)
        if config.max_instances and config.max_instances > 0:
            self.total_num_sequences = min(
                self.total_num_sequences, config.max_instances
            )

        self.cumsum_num_sequences = np.cumsum(self.routes_num_sequences)

        self.frame_stride = config.frame_stride

        if config.skip_noisy:
            # Iterate over all routes and remove noisy sequences if last element is noisy
            self.valid_route_idxes = []
            noisy_list = self.getnoisy()
            for i, n in enumerate(noisy_list):
                if not n:
                    self.valid_route_idxes.append(i)
            print(
                "Skipped all noisy sequences. Num noisy sequences:",
                len(noisy_list) - len(self.valid_route_idxes),
                "out of",
                len(noisy_list),
                "sequences.",
            )
            self.total_num_sequences = (
                len(self.valid_route_idxes)
                if config.max_instances is None
                else min(len(self.valid_route_idxes), config.max_instances)
            )
        else:
            self.valid_route_idxes = None

        self.integrate_rewards_to_go = config.integrate_rewards_to_go

    def __len__(self):
        # return 16
        return self.total_num_sequences

    # @profile
    def __getitem__(self, idx):
        if self.valid_route_idxes is not None:
            idx = self.valid_route_idxes[idx]

        # Binary search to find the correct route
        route_idx = np.searchsorted(self.cumsum_num_sequences, idx, side="right")

        idx -= self.cumsum_num_sequences[route_idx - 1] if route_idx > 0 else 0

        # Get timestep sequence
        sequence = self.route_datasets[route_idx][idx]

        # Flatten and convert to tensors
        sequence = self.flatten_time_sequence(sequence)

        return sequence

    # Get the weight of the sequence based on a custom heuristic
    # Similar to getitem, but calls the internal getweight function
    def __getweight__(self, idx):
        if self.valid_route_idxes is not None:
            idx = self.valid_route_idxes[idx]
        # Binary search to find the correct route
        route_idx = np.searchsorted(self.cumsum_num_sequences, idx, side="right")

        idx -= self.cumsum_num_sequences[route_idx - 1] if route_idx > 0 else 0

        # Get timestep sequence
        weight = self.route_datasets[route_idx].__getweight__(idx)

        return weight

    # Get whether a sequence is noisy or not
    # Similar to getitem, but calls the internal getweight function
    def __getnoisy__(self, idx):
        # Binary search to find the correct route
        route_idx = np.searchsorted(self.cumsum_num_sequences, idx, side="right")

        idx -= self.cumsum_num_sequences[route_idx - 1] if route_idx > 0 else 0

        # Get timestep sequence
        noisy = self.route_datasets[route_idx].__getnoisy__(idx)

        return noisy

    def flatten_time_sequence(
        self, sequence, attributes=["goal", "state", "action", "reward"]
    ):
        assert len(sequence) != 0
        result = {}
        for attrib in attributes:
            if attrib == "state":
                continue
            result[attrib] = self.flatten_dicts(
                [getattr(timestep, attrib) for timestep in sequence]
            )

        if "state" in attributes:
            state_dict = self.flatten_state_dict(
                [getattr(timestep, "state") for timestep in sequence]
            )
            for k in state_dict:
                result[k] = state_dict[k]

        return result

    def flatten_dicts(self, dicts):
        flattened = {k: [] for k in dicts[0]}

        for dict in dicts:
            for k in flattened:
                flattened[k].append(dict[k])

        for k in flattened:
            if isinstance(flattened[k][0], np.ndarray):  # Array
                dtype = (
                    np.float32
                    if np.issubdtype(flattened[k][0].dtype, np.floating)
                    else np.int64
                )
                flattened[k] = np.stack(flattened[k]).astype(dtype)
            else:  # Scalar
                dtype = np.float32 if type(flattened[k][0]) is float else np.int64
                flattened[k] = np.asarray(flattened[k], dtype=dtype)
                flattened[k] = np.expand_dims(flattened[k], axis=-1)

        keys = [k for k in dicts[0]]

        keys = self.sort_keys(keys)

        flattened_tensor = torch.tensor(
            np.concatenate([flattened[k] for k in keys], axis=-1)
        )

        return flattened_tensor

    def sort_keys(self, keys):
        # Sort keys so that the order is always the same
        # This is important to make sure that the model always gets the same input
        return sorted(keys)

    # We would like to treat bevs differently
    def flatten_state_dict(self, dicts):
        # print(dicts)
        flattened = {k: [] for k in dicts[0]}

        for dict in dicts:
            for k in flattened:
                flattened[k].append(dict[k])

        sorted_keys = sorted(k for k in dicts[0])

        result = {}
        flattened_tensor = torch.tensor(
            np.stack([flattened[k] for k in sorted_keys if not "bev" in k]).astype(
                np.float32
            )
        ).T

        result["state"] = flattened_tensor

        # concatenate and add every bev state separately to the result
        for k in sorted_keys:
            if k == "bevobject":
                # Steps:
                # Input:
                # List of vectorized vehicles (6d), lights(6d), and waypoints(6d)
                # They can be of different lengths, so we need to pad them to self.object_level_max_num_objects
                # First we concatenate them
                # Empty tensor so concat works if there are no objects
                # flattened_tensor = torch.tensor([]).float().reshape(0, 6)
                objects, types = postprocess_bev_objects(
                    flattened[k],
                    self.object_level_max_num_objects,
                    self.object_level_max_route_length,
                )
                result["bevobject"] = objects
                # result["bevobjectmask"] = types != type_mapping["padding"]

                # Convert pad type ids to 0 so that the embedding layer works
                # types = torch.where(
                #     types == type_mapping["padding"],
                #     torch.zeros_like(types),
                #     types,
                # )
                result["bevobjecttype"] = types
            elif "bev" in k:
                flattened_tensor = torch.tensor(np.stack(flattened[k])).float()
                result[k] = flattened_tensor

        return result

    # Get the weights of all sequences. This function caches the weights to avoid recomputing them. The cache is stored on disk in the
    # .cache folder, and the name of the file is computed from the hash of the state variables as well as the hash of the
    # length of the dataset. This means that if the dataset is modified, the cache will be invalidated.
    def getweights(self):
        # Check if the cache exists
        cache_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".cache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        # Calculate the hash of the state variables
        state_hash = hashlib.md5(
            f"{self.dataset_path}-{self.routes_num_sequences}-{self.total_num_sequences}-{self.frame_stride}".encode(
                "utf-8"
            )
        ).hexdigest()

        # Calculate the hash of the length of the dataset
        length_hash = hashlib.md5(
            f"{self.total_num_sequences}".encode("utf-8")
        ).hexdigest()

        instance_sample_weights_hash = hashlib.md5()

        for i in range(0, self.total_num_sequences, 100):
            instance_sample_weights_hash.update(
                f"{self.__getweight__(i)}".encode("utf-8")
            )

        instance_sample_weights_hash = instance_sample_weights_hash.hexdigest()

        # Check if the cache exists
        cache_file = os.path.join(
            cache_path,
            f"{state_hash}-{length_hash}-{instance_sample_weights_hash}-weights.npy",
        )

        if os.path.exists(cache_file):
            print("Loading weights from cache")
            return np.load(cache_file)

        # If the cache does not exist, compute the weights and save them to the cache
        weights = []
        # actions = []
        from tqdm import trange

        for i in trange(self.total_num_sequences):
            weights.append(self.__getweight__(i))
            # actions.append(self.__getitem__(i)["action"][:, 0])

        np.save(cache_file, np.asarray(weights))

        return weights

    def getnoisy(self):
        cache_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".cache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        # Calculate the hash of the state variables
        state_hash = hashlib.md5(
            f"{self.dataset_path}-{self.routes_num_sequences}-{self.total_num_sequences}-{self.frame_stride}".encode(
                "utf-8"
            )
        ).hexdigest()

        # Calculate the hash of the length of the dataset
        length_hash = hashlib.md5(
            f"{self.total_num_sequences}".encode("utf-8")
        ).hexdigest()

        # Check if the cache exists
        cache_file = os.path.join(cache_path, f"{state_hash}-{length_hash}-noisy.npy")

        if os.path.exists(cache_file):
            print("Loading noise weights from cache")
            return np.load(cache_file)

        # If the cache does not exist, check whether every instance is noisy or not and and save them to the cache
        noisy = []
        from tqdm import trange

        for i in trange(self.total_num_sequences, desc="Skipping noisy sequences"):
            noisy.append(self.__getnoisy__(i))

        np.save(cache_file, np.asarray(noisy))

        return noisy
