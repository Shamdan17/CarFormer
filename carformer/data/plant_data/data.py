from time import time
import numpy as np
import torch
import os
from os.path import join
from .data_parser import Parser
import hashlib
from tqdm import trange
from carla_birdeye_view import (
    BirdViewProducerObjectLevelRenderer,
    PixelDimensions,
    BirdViewCropType,
)
from glob import glob
from .data_utils import (
    postprocess_bev_objects,
    get_object_level_filter_from_config,
    transform_waypoints,
    OBJECT_TYPE_MAPPING,
    postprocess_bev_targets,
    transform_vehicles_according_to_egos,
    extract_forecast_targets,
)


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
    # @profile
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
        self.integrate_rewards_to_go = config.integrate_rewards_to_go
        self.context_length = config.context_length
        self.future_horizon = config.future_horizon
        self.object_level = config.object_level
        self.use_slots = config.get("use_slots", False)
        # print("use slots has been set to: ", self.use_slots)
        self.forecast_steps = config.get("forecast_steps", 1)
        for attr in ["action", "state", "bev"]:
            attr_dict = config.get(attr, {})
            for s_attr in attr_dict:
                self.future_horizon = max(
                    self.future_horizon, attr_dict[s_attr].get("future_horizon", 0)
                )
        config.future_horizon = self.future_horizon
        self.past_horizon = config.past_horizon
        self._round_func = np.floor if config.drop_last else np.ceil
        self.frame_stride = config.frame_stride
        self.trim_first_and_last = config.trim_first_and_last
        self.trim_count = config.trim_count if config.trim_first_and_last else 0
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
            self._round_func(
                (self.num_steps - self.future_horizon - self.past_horizon)
                / (config.context_length)
            )
        )
        if self.length < 1:
            self.length = 0
        self.preprocessing_functions = preprocessing_functions
        self.filtering_functions = filtering_functions
        self.throw_error_if_not_enough_timesteps = throw_error_if_not_enough_timesteps
        self.use_future_ego_waypoints = config.use_future_ego_waypoints
        self.use_future_vehicle_forcast = config.use_future_vehicle_forcast
        self.use_past_horizon_states = config.get("use_past_horizon_states", False)

    def __len__(self):
        return self.length

    # @profile
    def __getitem__(self, idx):
        # Get timestep
        idx_start = idx * (self.context_length) + self.trim_count

        # idx_end = min(idx_start + self.context_length + self.future_horizon, self.num_steps)
        idx_end = (
            idx_start + self.context_length + self.future_horizon + self.past_horizon
        )

        if idx_end > self.num_steps + self.trim_count:
            if self.throw_error_if_not_enough_timesteps:
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

        # Extract useful state from past horizon, discard the rest
        if self.use_past_horizon_states:
            assert (
                len(timesteps)
                == self.context_length + self.future_horizon + self.past_horizon
            )
            timesteps[self.past_horizon].state["bevslots"] = np.concatenate(
                [t.state["bevslots"] for t in timesteps[: self.past_horizon + 1]],
                axis=0,
            )
            if "bevslotspercept" in timesteps[0].state:
                timesteps[self.past_horizon].state["bevslotspercept"] = np.concatenate(
                    [
                        t.state["bevslotspercept"]
                        for t in timesteps[: self.past_horizon + 1]
                    ],
                    axis=0,
                )
            # # If past horizon is 0, unsqueeze for consistency
            # if self.past_horizon == 0:
            #     timesteps = np.expand_dims(timesteps, axis=0)

        # Get rid of first past_horizon timesteps
        timesteps = timesteps[self.past_horizon :]

        if self.use_future_ego_waypoints:
            assert len(timesteps) == self.context_length + self.future_horizon, (
                "Expected length "
                + str(self.context_length + self.future_horizon)
                + " but got "
                + str(len(timesteps))
            )

            # waypoints = np.zeros(
            #     (self.context_length, self.future_horizon * 2)
            # )  # Each waypoint is 2D
            for i in range(self.context_length):
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

        if self.use_future_vehicle_forcast:
            extract_forecast_targets(
                timesteps,
                self.context_length,
                self.future_horizon,
                use_slots=self.use_slots,
                object_level=self.object_level,
                forecast_steps=self.forecast_steps,
            )

        # Remove bevobjids, ego_matrix from state
        for i in range(self.context_length):
            timesteps[i].action.pop("ego_matrix", None)
            timesteps[i].state.pop("bevobjids", None)

        if self.future_horizon > 0:
            timesteps = timesteps[: -self.future_horizon]

        return timesteps

    def __getweight__(self, idx, reduce="mean"):
        idx_start = idx * self.context_length + self.trim_count

        idx_end = min(idx_start + self.context_length, self.num_steps)

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

    def __getnoisy__(self, idx, reduce="last"):
        idx_start = idx * self.context_length + self.trim_count

        idx_end = min(idx_start + self.context_length, self.num_steps)

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
        return f"TimeStepDataset(dataset_path={self.dataset_path}, parser={self.parser}, integrate_rewards_to_go={self.integrate_rewards_to_go}, context_length={self.context_length}, drop_last={self._round_func == np.ceil}, frame_stride={self.frame_stride})"


class SequenceDataset(torch.utils.data.Dataset):
    # @profile
    def __init__(
        self,
        dataset_path,
        split,
        config,
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.config = config
        self.object_level_max_num_objects = config.get(
            "object_level_max_num_objects", None
        )
        self.object_level_max_route_length = config.get(
            "object_level_max_route_length", None
        )

        routes = sorted(glob(join(dataset_path, *(["*"] * 4))))
        # routes = sorted(glob(join(dataset_path, "PlanT_data_1", *(["*"] * 3))))

        # Shuffle with fixed seed 42 to make sure that the split is always the same
        rng = np.random.RandomState(42)
        rng.shuffle(routes)

        if split == "all":
            self.routes = routes
        else:
            # Split first 96% of routes into train, next 2% into val, and last 2% into test
            self.routes = (
                routes[: int(len(routes) * 0.94)]
                if split == "train"
                else (
                    routes[int(len(routes) * 0.94) : int(len(routes) * 0.97)]
                    if split == "val"
                    else routes[int(len(routes) * 0.97) :]
                )
            )

        x_min = int(-config.bev_crop_size / 2 + config.bev_size / 2)
        x_max = int(config.bev_crop_size / 2 + config.bev_size / 2)
        y_min = int(
            -config.bev_crop_size / (1 if config.bev_crop == "front" else 2)
            + config.bev_size / 2
        )
        y_max = int(
            config.bev_crop_size * (0 if config.bev_crop == "front" else 0.5)
            + config.bev_size / 2
        )

        bev_crop_func = lambda x: (
            x[:, y_min:y_max, x_min:x_max]
            if (config.bev_crop == "center")
            else x[:, y_min:y_max, x_min:x_max]
        )
        bev_preprocessing_function = lambda x: (
            np.delete(bev_crop_func(x), 2, axis=0)
            if x.shape[0] == 9
            else bev_crop_func(x)
        )

        filtering_functions = {}
        preprocessing_functions = {
            "bev": bev_preprocessing_function,
        }

        if config.object_level:
            obj_filter = get_object_level_filter_from_config(config)
            filtering_functions["bevobject"] = obj_filter

        if "bevslotspercept" in config.state_type:

            def image_percept_prep(image):
                # Input: 160 * 960 * 3
                image = np.asarray(image).reshape(160, 3, -1, 3)

                image = np.transpose(image, (1, 3, 0, 2))

                image = image[[1, 0, 2]]

                image = image / 255.0

                return image[np.newaxis, ...]

            preprocessing_functions["bevslotspercept"] = image_percept_prep

        if "bevslots" in config.state_type:
            from .data_utils import get_slots_preprocessing_function_from_config

            preprocessing_functions["bevslots"] = (
                get_slots_preprocessing_function_from_config(config)
            )

            if config.get("enlarge_vehicles", False):
                old_lambda = preprocessing_functions["bevslots"]

                def enlarge_small_objs_hook(objs):
                    min_dims = [1.51, 4.9, 2.12]
                    for obj in objs:
                        if not "extent" in obj:
                            continue  # Other objects like traffic lights
                        for i in range(len(min_dims)):
                            if obj["extent"][i] < min_dims[i]:
                                obj["extent"][i] = min_dims[i]
                    return objs

                preprocessing_functions["bevslots"] = lambda x: old_lambda(
                    enlarge_small_objs_hook(x)
                )

        # To avoid recomputing
        folder_to_ext = Parser.get_folder_to_ext(join(dataset_path, self.routes[0]))

        state_type = config.state_type
        action_type = config.action_type
        reward_type = config.reward_type
        goal_type = config.goal_type

        print("State type:", state_type)
        print("Action type:", action_type)
        print("Reward type:", reward_type)
        print("Goal type:", goal_type)

        self.route_data_parsers = [
            Parser(
                join(dataset_path, route),
                state_type,
                action_type,
                reward_type,
                goal_type,
                folder_to_ext=folder_to_ext,
            )
            for route in self.routes
        ]

        self.route_datasets = [
            TimeStepDataset(
                join(dataset_path, route),
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
                if config.max_instances < 0
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
        # print(flattened["bevslotspercept"][0].shape)

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
            elif k == "targetbevobject":
                # Pad to self.object_level_max_num_objects + self.object_level_max_route_length
                objects, types = postprocess_bev_targets(
                    flattened[k],
                    object_level_max_num_objects=self.object_level_max_num_objects,
                    object_level_max_route_length=self.object_level_max_route_length,
                )
                result["targetbevobject"] = objects
                result["targetbevobjecttype"] = types
                # Offset instead:
                # flattened[k][0]["bevobject"][flattened[k][0]["bevobjecttype"]==1] -  flattened["bevobject"][0]["vehicles"][flattened[k][0]["bevobjecttype"]==1]
            elif "bevslots" in k:
                # First frame and past context are in flattened[k][0]
                # Future context is in flattened[k][1:]
                # Flatten and concatenate
                flattened_tensor = torch.from_numpy(
                    np.concatenate(flattened[k], axis=0)
                ).float()
                result[k] = flattened_tensor
            elif "targetbev" == k:
                flattened_tensor = torch.tensor(np.stack(flattened[k])).float()
                result[k] = flattened_tensor
            elif "bev" in k:
                flattened_tensor = torch.tensor(np.stack(flattened[k])).float()
                result[k] = flattened_tensor

        return result

    # Get the weights of all sequences. This function caches the weights to avoid recomputing them. The cache is stored on disk in the
    # .cache folder, and the name of the file is computed from the hash of the state variables as well as the hash of the
    # length of the dataset. This means that if the dataset is modified, the cache will be invalidated.
    def getweights(self):
        # Check if the cache exists
        cache_path = join(os.path.dirname(os.path.realpath(__file__)), ".cache")
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
        cache_file = join(
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
        cache_path = join(os.path.dirname(os.path.realpath(__file__)), ".cache")
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
        cache_file = join(cache_path, f"{state_hash}-{length_hash}-noisy.npy")

        if os.path.exists(cache_file):
            print("Loading noise weights from cache")
            return np.load(cache_file)

        # If the cache does not exist, check whether every instance is noisy or not and and save them to the cache
        noisy = []
        from tqdm import trange

        # for i in trange(self.total_num_sequences, desc="Skipping noisy sequences"):
        # noisy.append(self.__getnoisy__(i))
        noisy = [False] * self.total_num_sequences

        np.save(cache_file, np.asarray(noisy))

        return noisy

    def get_parametrized_dirname(self):
        dataset_path_name = os.path.basename(self.dataset_path)

        attributes = [
            "object_level_max_route_length",
            "object_level_max_num_objects",
            "include_traffic_lights_in_object_level",
            "split_large_routes",
            "split_threshold",
            "use_future_ego_waypoints",
            "future_horizon",
            "past_horizon",
            "trim_first_and_last",
            "trim_count",
            "forecast_steps",
        ]

        appended_attributes = [
            "enlarge_vehicles",
            "legacy",
            "num_slots",
            "perceive_slots",
        ]  # Only added if not None

        attr_string = ""
        for attr in attributes:
            attr_string += f"{attr}={getattr(self.config, attr, None)}"

        for attr in appended_attributes:
            attr_val = getattr(self.config, attr, None)
            if attr_val is not None:
                attr_string += f"{attr}={attr_val}"

        state_hash = hashlib.md5(attr_string.encode("utf-8")).hexdigest()

        return "{}_{}_{}_{}".format(
            dataset_path_name, state_hash[:10], self.split, self.total_num_sequences
        )
