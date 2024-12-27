import json
from logging import root
import os
from PIL import Image
import numpy as np
import scipy
from .data_utils import to_object_level_vector, transform_route


class Parser:
    def __init__(
        self,
        root_path,
        state_type,
        action_type,
        reward_type,
        goal_type,
    ):
        self.root_path = root_path
        self.state_type = state_type
        self.action_type = action_type
        self.reward_type = reward_type
        self.goal_type = goal_type

        self.states = state_type.split("-")
        # If bev or bevbin is in the states, assert the other is not in the states
        assert "bev" not in self.states or "bevbin" not in self.states
        self.actions = action_type.split("-")
        self.rewards = reward_type.split("-")
        self.goals = goal_type.split("-")

        self._folder_to_ext = {}

        # Check the extension of the first file in every folder in path
        for folder in os.listdir(self.root_path):
            folder_path = os.path.join(self.root_path, folder)
            if os.path.isdir(folder_path):
                fl = os.listdir(folder_path)
                # Get extension
                if len(fl) > 0:
                    ext = os.path.splitext(fl[0])[1]
                    self._folder_to_ext[folder] = ext

    # @profile
    def get_state(self, idx, preprocessing_functions=None, filtering_functions=None):
        ts_prefix = str(idx).zfill(4)
        state_dict = json.load(
            open(os.path.join(self.root_path, "full_state", f"{ts_prefix}.json"), "r")
        )
        measurement_dict = json.load(
            open(os.path.join(self.root_path, "measurements", f"{ts_prefix}.json"), "r")
        )

        state = {}

        for s in self.states:
            if s == "bev":
                state["bev"] = Image.open(
                    os.path.join(
                        self.root_path,
                        "bev",
                        f"{ts_prefix}{self._folder_to_ext['bev']}",
                    )
                )
            elif s == "bevbin":
                state["bev"] = np.load(
                    os.path.join(
                        self.root_path,
                        "bev_binary_npz",
                        f"{ts_prefix}{self._folder_to_ext['bev_binary_npz']}",
                    )
                )["bev"]
                s = "bev"
            elif s == "bevobject":
                objectlevel_state_path = os.path.join(
                    self.root_path, "bev_objectlevel", f"{ts_prefix}.json"
                )

                waypoint_path = os.path.join(
                    self.root_path, "measurements", f"{ts_prefix}.json"
                )

                state["bevobject"] = {}

                objects = []

                # If the objectlevel state exists, load it
                if os.path.exists(objectlevel_state_path):
                    objects_state = json.load(open(objectlevel_state_path, "r"))["bev"]
                    objects.extend(objects_state)
                else:
                    raise ValueError(
                        f"Objectlevel state requested. However, object level state for {self.root_path} does not exist."
                    )

                # If the waypoint state exists, load it
                if os.path.exists(waypoint_path):
                    waypoints_lst = json.load(open(waypoint_path, "r"))[
                        "plant_waypoints"
                    ]
                    objects.extend(waypoints_lst)
                else:
                    raise ValueError(
                        f"Objectlevel state requested. However, waypoints for {self.root_path} does not exist."
                    )

                # We apply the preprocessing function to the list of objects
                # Unlike the states
                # print("BEFORE", [obj["class"] for obj in objects])
                if (
                    filtering_functions is not None
                    and "bevobject" in filtering_functions
                ):
                    objects = filtering_functions["bevobject"](objects)
                # print("AFTER", [obj["class"] for obj in objects])

                vehicles = []
                trafficlights = []
                waypoints = []

                for obj in objects:
                    vector_rep = to_object_level_vector(obj)
                    if obj["class"] == "Vehicle":
                        vehicles.append(vector_rep)
                    elif obj["class"] == "TrafficLight":
                        trafficlights.append(vector_rep)
                    elif obj["class"] == "Route":
                        waypoints.append(vector_rep)

                # Convert the objectlevel state to a numpy array
                state["bevobject"]["vehicles"] = np.array(vehicles, dtype=np.float32)
                state["bevobject"]["tlights"] = np.array(
                    trafficlights, dtype=np.float32
                )
                state["bevobject"]["waypoints"] = np.array(waypoints, dtype=np.float32)

            elif s == "speed":
                state["speed"] = state_dict["state"]["speed"]["speed"]
            elif s == "lights":
                state["lights"] = 1 if measurement_dict["light_hazard"] else 0
            else:
                raise ValueError(f"State type {s} not recognized")

            if preprocessing_functions is not None and s in preprocessing_functions:
                state[s] = preprocessing_functions[s](state[s])

        return state

    # @profile
    def get_action(self, idx, include_noise=False):
        ts_prefix = str(idx).zfill(4)
        state_dict = json.load(
            open(os.path.join(self.root_path, "full_state", f"{ts_prefix}.json"), "r")
        )

        action = {}

        for a in self.actions:
            if a == "steer":
                action["steer"] = state_dict["action"]["steer"]
            elif a == "throttle":
                action["throttle"] = state_dict["action"]["throttle"]
            elif a == "brake":
                action["brake"] = state_dict["action"]["brake"]
            elif a == "acceleration":
                brake = state_dict["action"]["brake"]
                throttle = state_dict["action"]["throttle"]
                if brake > 0:
                    action["acceleration"] = -brake
                else:
                    action["acceleration"] = throttle
            elif a == "waypoints":
                # Get the current ego matrix from the measurement dict
                measurement_dict = json.load(
                    open(
                        os.path.join(
                            self.root_path, "measurements", f"{ts_prefix}.json"
                        ),
                        "r",
                    )
                )
                action["ego_matrix"] = measurement_dict["ego_matrix"]
            else:
                raise ValueError(f"Action type {a} not recognized")

        if include_noise:
            is_noisy = False
            if "noisy_action" in state_dict:
                # is_noisy = state_dict["noisy_action"]["noise_applied"]
                # Check if the noisy_action steering is more than 0.1 different from the original action
                if (
                    abs(
                        state_dict["noisy_action"]["steer"]
                        - state_dict["action"]["steer"]
                    )
                    > 0.1
                ):
                    is_noisy = True

            return action, is_noisy
        else:
            return action

    def get_noisy(self, idx):
        ts_prefix = str(idx).zfill(4)
        state_dict = json.load(
            open(os.path.join(self.root_path, "full_state", f"{ts_prefix}.json"), "r")
        )

        if "noisy_action" in state_dict:
            # is_noisy = state_dict["noisy_action"]["noise_applied"]
            # Check if the noisy_action steering is more than 0.1 different from the original action
            if (
                abs(state_dict["noisy_action"]["steer"] - state_dict["action"]["steer"])
                > 0.1
            ):
                return True

        return False

    # @profile
    def get_reward(self, idx):
        # ["reward", "speed_reward", "steering_reward", "lane_dist_reward", "route_angle_reward"]
        ts_prefix = str(idx).zfill(4)
        state_dict = json.load(
            open(os.path.join(self.root_path, "full_state", f"{ts_prefix}.json"), "r")
        )

        rewards = self.reward_type.split("-")

        reward = {}

        for r in rewards:
            if r == "reward":
                reward["reward"] = state_dict["reward_dict"]["reward"]
            elif r == "speed_reward":
                reward["speed_reward"] = state_dict["reward_dict"]["speed_reward"]
            elif r == "steering_reward":
                reward["steering_reward"] = state_dict["reward_dict"]["steering_reward"]
            elif r == "lane_dist_reward":
                reward["lane_dist_reward"] = state_dict["reward_dict"][
                    "lane_dist_reward"
                ]
            elif r == "route_angle_reward":
                reward["route_angle_reward"] = state_dict["reward_dict"][
                    "route_angle_reward"
                ]
            else:
                raise ValueError(f"Reward type {r} not recognized")

        return reward

    # @profile
    def get_goal(self, idx):
        ts_prefix = str(idx).zfill(4)
        state_dict = json.load(
            open(os.path.join(self.root_path, "full_state", f"{ts_prefix}.json"), "r")
        )

        goal = {}

        for g in self.goals:
            if g == "highlevel_command":
                # Shift the highlevel command by 1 to make it 0-indexed
                goal["command"] = state_dict["state"]["highlevel_command"] - 1
            elif g == "command":
                goal["command"] = state_dict["state"]["command"]
            elif g == "pooled_command":
                # TODO: Maybe add a function to pool the commands from the chunk of states that belong this timestep
                raise NotImplementedError
            elif g == "target_point":
                # Get the target point from the state
                measurement_dict = json.load(
                    open(
                        os.path.join(
                            self.root_path, "measurements", f"{ts_prefix}.json"
                        ),
                        "r",
                    )
                )

                route = measurement_dict["waypoint_route"]
                ego_matrix = measurement_dict["ego_matrix"]

                route = transform_route(route, ego_matrix)
                local_command_point = np.array(
                    route[1] if len(route) > 1 else route[0], dtype=np.float32
                )
                goal["target_point"] = local_command_point
            else:
                raise ValueError(f"Goal type {g} not recognized")

        return goal

    def get_size(self):
        return len(os.listdir(os.path.join(self.root_path, "bev")))

    def get_weight(self, idx):
        ts_prefix = str(idx).zfill(4)
        state_dict = json.load(
            open(os.path.join(self.root_path, "full_state", f"{ts_prefix}.json"), "r")
        )

        # yaw = state_dict["state"]["imu"][-1]
        # # Rad to deg
        # yaw = yaw * 180 / np.pi
        # # Normalize to 0-360
        # yaw = yaw % 90 - 45
        # yaw = np.abs(yaw) / (45 * 0.75)

        # yaw_score = scipy.stats.norm(0, 1).pdf(yaw) ** 1

        # junction = 0.5 if state_dict["state"]["is_junction"] else 0.1

        # speed_score = (state_dict["state"]["speed"]["speed"] - 3.5) / 3.5

        # speed_score = scipy.stats.norm(0, 1).pdf(speed_score)

        # return yaw_score * speed_score * junction
        if state_dict["action"]["brake"] > 0.9:
            break_weight = 0.1
        else:
            break_weight = 1.0

        return break_weight
