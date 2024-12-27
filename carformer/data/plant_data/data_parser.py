# import orjson as json
import json
from logging import root
import os
from PIL import Image
import numpy as np
import scipy
from .data_utils import (
    to_object_level_vector,
    plant_to_carformer_object,
)


class Parser:
    def __init__(
        self,
        root_path,
        state_type,
        action_type,
        reward_type,
        goal_type,
        folder_to_ext=None,
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

        if folder_to_ext is None:
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
        else:
            self._folder_to_ext = folder_to_ext

    # @profile
    def get_state(self, idx, preprocessing_functions=None, filtering_functions=None):
        ts_prefix = str(idx).zfill(4)
        state_dict = json.load(
            open(os.path.join(self.root_path, "measurements", f"{ts_prefix}.json"), "r")
        )
        object_dict = json.load(
            open(os.path.join(self.root_path, "boxes", f"{ts_prefix}.json"), "r")
        )

        state = {}

        for s in self.states:
            if s == "bevbin":
                state["bev"] = np.load(
                    os.path.join(
                        self.root_path,
                        "bev_binary_npz",
                        f"{ts_prefix}.npz",
                    )
                )["bev"]
                s = "bev"
            elif s == "bevobject":
                state["bevobject"] = {}

                objects = []

                for object in object_dict:
                    object = plant_to_carformer_object(object)
                    if object["class"].strip():
                        objects.append(object)

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
                vehicle_ids = []
                trafficlights = []
                waypoints = []

                for obj in objects:
                    vector_rep = to_object_level_vector(obj)
                    if obj["class"] == "Vehicle":
                        vehicles.append(vector_rep)
                        vehicle_ids.append(obj["id"])
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
                state["bevobjids"] = np.array(vehicle_ids, dtype=np.int32)
            elif s == "bevslots":
                objects = json.load(
                    open(
                        os.path.join(
                            self.root_path, "bev_objectlevel", f"{ts_prefix}.json"
                        ),
                        "r",
                    )
                )["bev"]
                state["bevslots"] = objects
            elif s == "bevslotspercept":
                rgbbev = Image.open(
                    os.path.join(self.root_path, "rgb_bev", f"{ts_prefix}.png")
                )
                state["bevslotspercept"] = rgbbev
            elif s == "speed":
                state["speed"] = state_dict["speed"]
            elif s == "lights":
                state["lights"] = int(state_dict["light_hazard"])
            elif s == "hazards":
                state["hazards"] = state_dict["vehicle_hazard"]
            else:
                raise ValueError(f"State type {s} not recognized")

            if preprocessing_functions is not None and s in preprocessing_functions:
                state[s] = preprocessing_functions[s](state[s])

        return state

    # @profile
    def get_action(self, idx, include_noise=False):
        ts_prefix = str(idx).zfill(4)
        state_dict = json.load(
            open(os.path.join(self.root_path, "measurements", f"{ts_prefix}.json"), "r")
        )

        action = {}

        for a in self.actions:
            if a == "steer":
                action["steer"] = state_dict["steer"]
            elif a == "throttle":
                action["throttle"] = state_dict["throttle"]
            elif a == "brake":
                action["brake"] = int(state_dict["brake"])
            elif a == "acceleration":
                brake = int(state_dict["brake"])
                throttle = state_dict["throttle"]
                if brake > 0:
                    action["acceleration"] = -brake
                else:
                    action["acceleration"] = throttle
            elif a == "waypoints":
                # Get the current ego matrix from the measurement dict
                action["ego_matrix"] = state_dict["ego_matrix"]
            else:
                raise ValueError(f"Action type {a} not recognized")

        if include_noise:
            return action, False
        else:
            return action

    def get_noisy(self, idx):
        # return False
        ts_prefix = str(idx).zfill(4)
        state_dict = json.load(
            open(os.path.join(self.root_path, "measurements", f"{ts_prefix}.json"), "r")
        )
        return any(state_dict["walker_hazard"])

    # @profile
    def get_reward(self, idx):
        rewards = {}

        for r in self.rewards:
            rewards[r] = 0.0

        return rewards

    # @profile
    def get_goal(self, idx):
        ts_prefix = str(idx).zfill(4)
        state_dict = json.load(
            open(os.path.join(self.root_path, "measurements", f"{ts_prefix}.json"), "r")
        )

        goal = {}

        for g in self.goals:
            if g == "command":
                goal["command"] = state_dict["command"]
            elif g == "pooled_command":
                # TODO: Maybe add a function to pool the commands from the chunk of states that belong this timestep
                raise NotImplementedError
            elif g == "target_point":
                # Get the target point from the state
                local_command_point = np.array(state_dict["target_point"])
                goal["target_point"] = local_command_point
            else:
                raise ValueError(f"Goal type {g} not recognized")

        return goal

    def get_size(self):
        return len(os.listdir(os.path.join(self.root_path, "measurements")))

    def get_weight(self, idx):
        # Weight is 50 if the speed is < 0.1 and throttle is more than 0.5, else 1
        ts_prefix = str(idx).zfill(4)
        state_dict = json.load(
            open(os.path.join(self.root_path, "measurements", f"{ts_prefix}.json"), "r")
        )
        # if (-0.05 <= state_dict["speed"] < 0.1) and state_dict["throttle"] > 0.5:
        #     return 50.0

        # return 1.0

        if (-0.05 <= state_dict["speed"] < 0.1) and state_dict["throttle"] > 0.5:
            return 2.5
        else:
            return 1.0

    @staticmethod
    def get_folder_to_ext(dir):
        folder_to_ext = {}
        for folder in os.listdir(dir):
            folder_path = os.path.join(dir, folder)
            if os.path.isdir(folder_path):
                fl = os.listdir(folder_path)
                # Get extension
                if len(fl) > 0:
                    ext = os.path.splitext(fl[0])[1]
                    folder_to_ext[folder] = ext
                else:
                    raise ValueError(f"Folder {folder} is empty in directory {dir}")

        return folder_to_ext
