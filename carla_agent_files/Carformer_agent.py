import os
import json
import time
from pathlib import Path
from tkinter import N
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageOps
from matplotlib import pyplot as plt

import cv2
import torch
import numpy as np
import carla

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from carla_agent_files.agent_utils.filter_functions import *
from carla_agent_files.agent_utils.coordinate_utils import (
    preprocess_compass,
    inverse_conversion_2d,
)
from carla_agent_files.agent_utils.explainability_utils import *

from carla_agent_files.data_agent_boxes import DataAgent

from nav_planner import extrapolate_waypoint_route
from nav_planner import RoutePlanner_new as RoutePlanner
from scenario_logger import ScenarioLogger

from carformer.wanderer import Wanderer
from carformer.backbone import gpt2
from carformer.visualization.visutils import (
    save_bin_probs,
    visualize_input_from_batch,
    inference_visualization_from_batch,
    visualize_trajectory_action_predictions,
)
from carformer.encoders.StoSAVi_metrics import get_binary_mask_from_rgb
from carformer.utils import TokenTypeIDs
from carformer.data.plant_data import (
    get_object_level_filter_from_config,
    get_slots_preprocessing_function_from_config,
    plant_to_carformer_object,
    to_object_level_vector,
)
from carla_birdeye_view import (
    BirdViewProducerObjectLevelRenderer,
    BirdViewCropType,
    PixelDimensions,
)
from carformer.backbone.generation_utils import (
    AllGasNoBrakesProcessor,
    ProperActionsProcessor,
)
from carformer.data.plant_data import Parser, postprocess_bev_objects
from carla_birdeye_view import (
    BirdViewProducerObjectLevelRenderer,
    PixelDimensions,
    BirdViewCropType,
)
import numpy as np
from torch.utils.data.dataloader import default_collate
from visualization.visutils import print_embedding_counters
from srunner.scenariomanager.timer import GameTime
import copy


def get_entry_point():
    return "CarformerAgent"


SAVE_GIF = os.getenv("SAVE_GIF", "False").lower() in ("true", "1", "t")
bev_crop_size = 192
bev_size = 400
bev_crop = "front"

x_min = int(-bev_crop_size / 2 + bev_size / 2)
x_max = int(bev_crop_size / 2 + bev_size / 2)
y_min = int(-bev_crop_size / (1 if bev_crop == "front" else 2) + bev_size / 2)
y_max = int(bev_crop_size * (0 if bev_crop == "front" else 0.5) + bev_size / 2)

bev_crop_func = lambda x: (
    x[..., :, y_min:y_max, x_min:x_max]
    if (bev_crop == "center")
    else x[:, y_min:y_max, x_min:x_max]
)
bev_preprocessing_function = lambda x: (
    np.delete(bev_crop_func(x), 2, axis=-3) if x.shape[-3] == 9 else bev_crop_func(x)
)


# Adapted from https://github.com/autonomousvision/plant/blob/main/carla_agent_files/PlanT_agent.py
class CarformerAgent(DataAgent):
    def setup(self, path_to_conf_file, route_index=None, cfg=None, exec_or_inter=None):
        self.cfg = cfg
        self.exec_or_inter = exec_or_inter

        # first args than super setup is important!
        # args_file = open(os.path.join(path_to_conf_file, "args.txt"), "r")
        # self.args = json.load(args_file)
        # args_file.close()
        # self.cfg_agent = OmegaConf.create(self.args)

        super().setup(path_to_conf_file, route_index, cfg, exec_or_inter)

        print(f"Saving gif: {SAVE_GIF}")

        # Filtering
        self.points = MerweScaledSigmaPoints(
            n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x
        )
        self.ukf = UKF(
            dim_x=4,
            dim_z=4,
            fx=bicycle_model_forward,
            hx=measurement_function_hx,
            dt=1 / self.frame_rate,
            points=self.points,
            x_mean_fn=state_mean,
            z_mean_fn=measurement_mean,
            residual_x=residual_state_x,
            residual_z=residual_measurement_h,
        )

        # State noise, same as measurement because we
        # initialize with the first measurement later
        self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
        # Measurement noise
        self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
        self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
        # Used to set the filter state equal the first measurement
        self.filter_initialized = False
        # Stores the last filtered positions of the ego vehicle.
        # Used to realign.
        self.state_log = deque(maxlen=2)

        model_name = cfg.model_path
        EPOCH_NUM = cfg.epoch_num

        root_path = cfg.root_path
        model_dir = os.path.join(root_path, model_name)

        self.config = gpt2.GPT2Config.from_pretrained(
            model_dir,
        )

        # Update quantizer paths using the path of carformer
        import inspect, carformer

        carformer_path = os.path.dirname(inspect.getfile(carformer))
        # Go one layer up
        carformer_path = os.path.dirname(carformer_path)

        self.config.training["action_quantizer_path"] = os.path.join(
            carformer_path, self.config.training["action_quantizer_path_rel"]
        )
        self.config.training["goal_quantizer_path"] = os.path.join(
            carformer_path, self.config.training["goal_quantizer_path_rel"]
        )
        self.config.training["state_quantizer_path"] = os.path.join(
            carformer_path, self.config.training["state_quantizer_path_rel"]
        )
        self.config.training["reward_quantizer_path"] = os.path.join(
            carformer_path, self.config.training["reward_quantizer_path_rel"]
        )

        self.model = Wanderer(self.config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        from argparse import Namespace

        if self.config.training["object_level"]:
            self.bev_object_filter = get_object_level_filter_from_config(
                Namespace(**self.config.training)
            )
        else:
            self.bev_object_filter = lambda x: x

        if "bevslots" in self.config.training["state_type"]:
            self.use_slots = True
            original_preprocess_function = get_slots_preprocessing_function_from_config(
                Namespace(**self.config.training)
            )

            if self.cfg.enlarge_small_vehicles:
                if not self.cfg.unsupervised_enlargement:

                    def enlarge_small_objs_hook(objs):
                        min_dims = [1.51, 4.9, 2.12]
                        for obj in objs:
                            if not "extent" in obj:
                                continue  # Other objects like traffic lights
                            for i in range(len(min_dims)):
                                if obj["extent"][i] < min_dims[i]:
                                    obj["extent"][i] = min_dims[i]
                        return objs

                    preprocess_slot_function = lambda x: original_preprocess_function(
                        enlarge_small_objs_hook(x)
                    )
                    self.slots_postprocess = lambda x, prev: x
                else:
                    preprocess_slot_function = original_preprocess_function

                    from scipy.ndimage import label, center_of_mass
                    from carformer.data.plant_data.data_utils import VEHICLE_RGB_BY_MASK

                    min_area = 200  # 5 pixels per meter

                    op = "dilate"

                    if self.cfg.unsupervised_coloring_only:
                        op = "erode"

                    max_iters = 5
                    kernel = np.ones((3, 3), np.uint8)

                    def enlarge_small_objs_from_bev(bev, previous_bev=None):
                        # import ipdb; ipdb.set_trace()  # fmt:skip

                        final_bev = torch.zeros_like(bev)

                        # binarize bev
                        bev = (bev.numpy()[0].sum(axis=0) > 0).astype(np.uint8)

                        labeled_objs, count_objs = label(bev)
                        center_of_masses = center_of_mass(
                            bev, labeled_objs, range(1, count_objs + 1)
                        )

                        for i in range(1, count_objs + 1):
                            obj_mask = (labeled_objs == i).astype(np.uint8)

                            com = center_of_masses[i - 1]

                            edge_size = bev.shape[-1]

                            # Calculate distance from edges
                            min_edge_dist = (
                                min(com[0], edge_size - com[0]),
                                min(com[1], edge_size - com[1]),
                            )
                            print(min_edge_dist)

                            threshold_com = 10

                            scale_factor = 1.0

                            for edge_dist in min_edge_dist:
                                if edge_dist < threshold_com:
                                    scale_factor *= edge_dist / threshold_com
                                # print("Decreasing scale factor: ", scale_factor)

                            obj_area = obj_mask.sum()
                            if op == "dilate":
                                if obj_area < min_area * scale_factor:
                                    for _ in range(max_iters):
                                        obj_mask = cv2.dilate(obj_mask, kernel)
                                        obj_area = obj_mask.sum()
                                        if obj_area > min_area:
                                            break

                            else:
                                for _ in range(2):
                                    obj_mask = cv2.erode(obj_mask, kernel)

                            # Now object is enlarged, assign color
                            # Find the mode of the previous bev at given location, if any
                            best_color = None
                            if previous_bev is not None:
                                prev_area = previous_bev[0, :, obj_mask != 0]  # 3xN

                                prev_area_nonzero = prev_area[
                                    :, prev_area.sum(axis=0) != 0
                                ].numpy()

                                # Find most common color, if any, by looking at the mode along the color axis
                                if prev_area_nonzero.size > 0:

                                    colors, cnts = np.unique(
                                        prev_area_nonzero, axis=1, return_counts=True
                                    )

                                    best_color_cnt = cnts[np.argmax(cnts)]

                                    if best_color_cnt / obj_area > 0.5:
                                        best_color = colors[:, np.argmax(cnts)]

                            if best_color is None:
                                rnd_color = np.random.randint(
                                    0, len(VEHICLE_RGB_BY_MASK)
                                )

                                best_color = (
                                    np.asarray(VEHICLE_RGB_BY_MASK[rnd_color]) / 255.0
                                )

                            final_bev[0, :, obj_mask > 0] = (
                                torch.from_numpy(best_color).reshape(3, 1).float()
                            )

                        return final_bev

                    self.slots_postprocess = enlarge_small_objs_from_bev

            else:
                preprocess_slot_function = original_preprocess_function
                self.slots_postprocess = lambda x, prev: x

            self.slots_preprocess = preprocess_slot_function
        else:
            self.use_slots = False
            self.slots_preprocess = lambda x: x
            self.slots_postprocess = lambda x, prev: x

        if EPOCH_NUM == "best":
            model_path = os.path.join(model_dir, "best_model.pt")
        elif EPOCH_NUM == "last":
            model_path = os.path.join(model_dir, "last_model.pt")
        else:
            model_path = os.path.join(model_dir, "epochs", f"epoch_{EPOCH_NUM}.pt")

        model_checkpoint = torch.load(model_path)["model"]

        keys = list(model_checkpoint.keys())

        for k in keys:
            if "module." in k:
                model_checkpoint[k.replace("module.", "")] = model_checkpoint.pop(k)

        print(self.config)

        self.model.load_state_dict(model_checkpoint, strict=True)
        self.model.eval()

        self.context = []
        self.data_fps = self.config.dataset["fps"]
        self.frame_stride = self.config.training["frame_stride"]
        self.sim_fps = 20
        self.context_stride = int(self.sim_fps / self.data_fps) * self.frame_stride
        self.context_length_training = self.config.training["context_length"]
        self.past_horizon_training = self.config.training["past_horizon"]

        self.total_sequence_length = (
            self.context_length_training + self.past_horizon_training
        )

        self.context_array_max_length = (
            1 + (self.total_sequence_length - 1) * self.context_stride
        )

        print_embedding_counters(
            self.model.embedding_counter,
            {
                "action": {
                    "quantizer": self.model.action_quantizer,
                    "offset": self.model.quantization_offset_map[TokenTypeIDs.ACTION],
                },
            },
        )
        self.scenario_logger = False

        if self.log_path is not None:
            self.log_path = Path(self.log_path) / route_index
            Path(self.log_path).mkdir(parents=True, exist_ok=True)

            self.scenario_logger = ScenarioLogger(
                save_path=self.log_path,
                route_index=self.route_index,
                logging_freq=self.save_freq,
                log_only=False,
                route_only=False,  # with vehicles and lights
                roi=self.detection_radius + 10,
            )

        self.turn_controller = PIDController(K_P=0.9, K_I=0.75, K_D=0.3, n=20)
        self.speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=20)

        self.stuck_detector = 0
        self.forced_move = False
        self.forced_move_counter = 0
        self.infraction_count = 0

        # Create folder for visualization at trajectories_viz/timestamp in ms
        # Time in format: 2020_07_01_15_23_45_293 # Only 3 digits for ms
        import datetime

        time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
        self.trajectory_viz_dir = os.path.join("./trajectories_viz/", time)

        with open(os.path.join("evalconfig.yml"), "w") as f:
            OmegaConf.save(vars(self.cfg), f)

        with open(os.path.join("evalconfig.json"), "w") as f:
            metadata = {
                "route_index": self.route_index,
                "log_path": str(self.log_path),
                # USER in unix
                "user": os.getenv("USER"),
            }
            json.dump(metadata, f)

    def _init(self, hd_map):
        super()._init(hd_map)
        self._route_planner = RoutePlanner(7.5, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.save_mask = []
        self.save_topdowns = []
        self.timings_run_step = []
        self.timings_forward_model = []

        self.keep_ids = None

        self.control = carla.VehicleControl()
        self.control.steer = 0.0
        self.control.throttle = 0.0
        self.control.brake = 1.0

        self.initialized = True

        if self.scenario_logger:
            from srunner.scenariomanager.carla_data_provider import (
                CarlaDataProvider,
            )  # privileged

            self._vehicle = CarlaDataProvider.get_hero_actor()
            self.scenario_logger.ego_vehicle = self._vehicle
            self.scenario_logger.world = self._vehicle.get_world()

            vehicle = CarlaDataProvider.get_hero_actor()
            self.scenario_logger.ego_vehicle = vehicle
            self.scenario_logger.world = vehicle.get_world()

        if SAVE_GIF == True and (self.exec_or_inter == "inter"):
            self.save_path_mask = f"viz_img/{self.route_index}/masked"
            self.save_path_org = f"viz_img/{self.route_index}/org"
            Path(self.save_path_mask).mkdir(parents=True, exist_ok=True)
            Path(self.save_path_org).mkdir(parents=True, exist_ok=True)

    def sensors(self):
        result = super().sensors()
        return result

    def tick(self, input_data):
        result = super().tick(input_data)

        pos = self._route_planner.convert_gps_to_carla(input_data["gps"][1][:2])
        speed = input_data["speed"][1]["speed"]
        compass = preprocess_compass(input_data["imu"][1][-1])

        if not self.filter_initialized:
            self.ukf.x = np.array([pos[0], pos[1], compass, speed])
            self.filter_initialized = True

        self.ukf.predict(
            steer=self.control.steer,
            throttle=self.control.throttle,
            brake=self.control.brake,
        )
        self.ukf.update(np.array([pos[0], pos[1], compass, speed]))
        filtered_state = self.ukf.x
        self.state_log.append(filtered_state)
        result["gps"] = filtered_state[0:2]

        waypoint_route = self._route_planner.run_step(filtered_state[0:2])

        if len(waypoint_route) > 2:
            target_point, _ = waypoint_route[1]
            next_target_point, _ = waypoint_route[2]
        elif len(waypoint_route) > 1:
            target_point, _ = waypoint_route[1]
            next_target_point, _ = waypoint_route[1]
        else:
            target_point, _ = waypoint_route[0]
            next_target_point, _ = waypoint_route[0]

        ego_target_point = inverse_conversion_2d(target_point, result["gps"], compass)
        result["target_point"] = tuple(ego_target_point)

        if SAVE_GIF == True and (self.exec_or_inter == "inter"):
            result["rgb_back"] = input_data["rgb_back"]
            result["sem_back"] = input_data["sem_back"]

        if self.scenario_logger:
            waypoint_route = self._waypoint_planner.run_step(filtered_state[0:2])
            waypoint_route = extrapolate_waypoint_route(waypoint_route, 10)
            route = np.array([[node[0][1], -node[0][0]] for node in waypoint_route])[
                :10
            ]
            # Logging
            self.scenario_logger.log_step(route)

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None, keep_ids=None):
        self.keep_ids = keep_ids

        self.step += 1
        if not self.initialized:
            if "hd_map" in input_data.keys():
                self._init(input_data["hd_map"])
            else:
                self.control = carla.VehicleControl()
                self.control.steer = 0.0
                self.control.throttle = 0.0
                self.control.brake = 1.0
                if self.exec_or_inter == "inter":
                    return [], None
                return self.control

        # needed for traffic_light_hazard
        _ = super()._get_brake(stop_sign_hazard=0, vehicle_hazard=0, walker_hazard=0)
        tick_data = self.tick(input_data)
        label_raw = super().get_bev_boxes(input_data=input_data, pos=tick_data["gps"])

        if self.use_slots:
            cur_bevobj = tick_data["bev_objectlevel"]
            if len(self.context) > 0:
                prev_bevslots = self.context[-1][-1]["bevslots"]
            else:
                prev_bevslots = None

            preprocessed_slots = self.slots_preprocess(cur_bevobj)

            tick_data["bevslots"] = self.slots_postprocess(
                preprocessed_slots, prev_bevslots
            )

        self.context.append((label_raw, tick_data))
        if len(self.context) > self.context_array_max_length:
            self.context.pop(0)
        if len(self.context) < self.context_array_max_length:
            return self.control

        self.control = self._get_control(label_raw, tick_data)

        # inital_frames_delay = 40
        # if self.step < inital_frames_delay:
        #     self.control = carla.VehicleControl(0.0, 0.0, 1.0)

        return self.control

    def _get_control(self, label_raw, input_data):
        inp_dict = self.get_input_batch(label_raw, input_data)

        inp_dict = {k: v.to(self.device) for k, v in inp_dict.items()}

        if self.cfg.use_gru_output:
            ot = self.model.generate_gru_only_optimized(
                inp_dict,
                total_sequence_length=1,
                output_scores=True,
                do_sample=False,
            )
        else:
            ot = self.model.generate(
                inp_dict,
                total_sequence_length=1,
                output_scores=True,
                do_sample=False,
            )

        if self.cfg.use_gru_output:
            waypoints = ot["waypoints"][0, -1].reshape(4, 2).cpu()
        else:
            waypoints = ot["output_dict"][2][0].reshape(-1, 8)[-1, :].reshape(4, 2)

        print(GameTime.get_time(), waypoints[0, :].flatten())

        gt_velocity = torch.FloatTensor([input_data["speed"]]).unsqueeze(0)

        # waypoints
        is_stuck = False

        # unblock
        # divide by 2 because we process every second frame (This is not the case actually)
        # 1100 = 55 seconds * 20 Frames per second, we move for 0.5 second = 30 frames to unblock
        if (self.stuck_detector > self.cfg.creep_delay) and self.cfg.use_creep:
            creep_method = getattr(self.cfg, "creep_method", "default")
            if creep_method == "default":
                self.forced_move = True
                self.stuck_detector = 0
            elif creep_method == "advanced":
                creep_settings = getattr(self.cfg, "creep_settings", {})
                if creep_settings.get("check_bev", False):
                    bevhazard = self.check_vehicle_hazard_in_bev(
                        self.model, inp_dict, ot
                    )

                # if creep_settings.get("check_lights", False):
                # if self.traffic_light_hazard:
                #     self.stuck_detector -= 10

                if not (bevhazard or self.traffic_light_hazard):
                    self.forced_move = True
                    self.stuck_detector = 0
                else:
                    self.stuck_detector -= 50

        if self.forced_move and self.forced_move_counter < self.cfg.creep_duration:
            creep_method = getattr(self.cfg, "creep_method", "default")
            halt = False
            if creep_method == "advanced":
                creep_settings = getattr(self.cfg, "creep_settings", {})
                if creep_settings.get("check_bev", False):
                    bevhazard = self.check_vehicle_hazard_in_bev(
                        self.model, inp_dict, ot
                    )

                # if creep_settings.get("check_lights", False):
                if self.traffic_light_hazard or bevhazard:
                    halt = True

            if halt:
                print("Detected hazard. Halting creep.")
                self.forced_move = False
                self.forced_move_counter = 0
            else:
                print(
                    "Detected agent being stuck. Move for frame: ",
                    self.forced_move_counter,
                )
                is_stuck = True
                self.forced_move_counter += 1
        elif self.forced_move and self.forced_move_counter >= self.cfg.creep_duration:
            self.forced_move = False
            self.forced_move_counter = 0

        steer, throttle, brake = self.control_pid(waypoints, gt_velocity, is_stuck)

        if brake < 0.05:
            brake = 0.0
        if throttle > brake:
            brake = 0.0

        if brake:
            steer *= self.steer_damping

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        viz_trigger = (self.step % self.cfg.viz_interval == 0) and self.cfg.viz

        if viz_trigger and self.step > 2:
            bev_crop_type = (
                "center" if (self.config.training["object_level"]) else "front"
            )
            visualize_trajectory_action_predictions(
                inp_dict,
                ot,
                save_dir=self.trajectory_viz_dir,
                model=self.model,
                save_prefix="plant",
                save_suffix=f"plant_{self.step}",
                save_idx=0,
                action_type=self.config.training["action_type"],
                state_type=self.config.training["state_type"],
                goal_type=self.config.training["goal_type"],
                bev_crop_type=bev_crop_type,
                pix_per_meter=5,
                boundary_width=10 if is_stuck else 4,
                include_targets=False,  # Since we don't have targets at test time
                action_source="gru" if self.cfg.use_gru_output else "transformer",
            )

        return control

    def get_bevobject_representation(self, bev_objectlevel):
        # Get the BEV object representation from bev_objectlevel
        bev_objects, bev_types = postprocess_bev_objects(
            [bev_objectlevel],
            self.config.training["object_level_max_num_objects"],
            self.config.training["object_level_max_route_length"],
        )
        return bev_objects, bev_types

    def get_plant_bevobject_representation(self, labels_raw):
        state = {}

        objects = []

        for object in labels_raw:
            object = plant_to_carformer_object(object)
            if object["class"].strip():
                objects.append(object)

        # We apply the preprocessing function to the list of objects
        # Unlike the states
        # print("BEFORE", [obj["class"] for obj in objects])
        objects = self.bev_object_filter(objects)
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
        state["vehicles"] = np.array(vehicles, dtype=np.float32)
        state["tlights"] = np.array(trafficlights, dtype=np.float32)
        state["waypoints"] = np.array(waypoints, dtype=np.float32)

        objects, types = postprocess_bev_objects(
            [state],
            self.config.training["object_level_max_num_objects"],
            self.config.training["object_level_max_route_length"],
        )
        result = {}
        result["bevobject"] = objects
        result["bevobjecttype"] = types

        return result

    def get_input_batch(self, label_raw, input_data):
        inp_dict = {}

        if input_data["speed"] < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        speed = torch.FloatTensor([input_data["speed"]]).unsqueeze(0)
        lights = torch.FloatTensor([self.traffic_light_hazard]).unsqueeze(0)
        if "lights" in self.config.training["state_type"]:
            # speed = torch.cat((speed, lights), dim=-1)
            speed = torch.cat((lights, speed), dim=-1)
            # print("Speed: ", speed.shape, speed[0, :])

        inp_dict["state"] = speed.unsqueeze(0)

        bev = torch.tensor(bev_preprocessing_function(input_data["bev"])).unsqueeze(0)
        # print(bev.dtype)
        bev = bev.float()

        inp_dict["bev"] = bev.unsqueeze(0)

        # object_reps = [self.get_bevobject_representation(input_data["bev_objectlevel"])]
        if self.config.training["object_level"]:
            object_reps = self.get_plant_bevobject_representation(label_raw)

            inp_dict["bevobject"] = object_reps["bevobject"].unsqueeze(0)
            inp_dict["bevobjecttype"] = object_reps["bevobjecttype"].unsqueeze(0)

        if self.use_slots:
            slot_input_indices = self.context[0 :: self.context_stride]
            actual_indices = np.arange(0, len(self.context), self.context_stride)

            all_slots = []
            for idx, timestep in zip(actual_indices, slot_input_indices):
                cur_bevobj = timestep[-1]["bev_objectlevel"]

                all_slots.append(timestep[-1]["bevslots"])

            all_slots = torch.cat(all_slots, dim=0).unsqueeze(0)

            inp_dict["bevslots"] = all_slots

        if self.config.training["goal_type"] == "highlevel_command":
            goal_key = "highlevel_command"
            offset = 1
        elif self.config.training["goal_type"] == "command":
            goal_key = "command"
            offset = 0
        elif self.config.training["goal_type"] == "target_point":
            goal_key = "target_point"
            offset = -1
        else:
            raise ValueError(
                "Invalid goal type {} specified".format(
                    self.config.training["goal_type"]
                )
            )

        if "command" in self.config.training["goal_type"]:
            raise NotImplementedError
        elif "target_point" in self.config.training["goal_type"]:
            local_command_point = torch.FloatTensor(
                [input_data["target_point"][0], input_data["target_point"][1]]
            ).reshape(1, -1)

            # Clip between -18 and 18
            local_command_point = torch.clamp(local_command_point, min=-18, max=18)

            goals = local_command_point

        inp_dict["goal"] = goals.unsqueeze(0)

        # Handle if context size is 1 where we don't have a previous action/reward
        # if len(self.context) == 1:
        action = torch.zeros((0, 8), dtype=torch.int)
        reward = torch.zeros((0, 1), dtype=torch.int)

        inp_dict["reward"] = reward.unsqueeze(0)
        inp_dict["action"] = action.unsqueeze(0)

        return inp_dict

    def check_vehicle_hazard_in_bev(self, model, input_batch, output_dict):
        if self.use_slots:
            bevslotslatent = output_dict["backbone_inputs"]["to_cache"][
                "bevslotslatent"
            ]
            bevslots = model.bev_encoder.interpret_slots(bevslotslatent)

            bevslots = bevslots.squeeze(0).cpu()

            vehiclehazard = self.check_vehicle_hazard_in_bevslots(model, bevslots)
            return vehiclehazard
        else:
            if self.config.training["object_level"]:
                bevobject = input_batch["bevobject"]
                bevobjecttype = input_batch["bevobjecttype"]

                return self.check_vehicle_hazard_in_bevobject(
                    model, bevobject, bevobjecttype
                )

        raise ValueError("No vehicle hazard check method found")

    def check_vehicle_hazard_in_bevslots(self, model, bevslots):
        # Bevslots: per slot RGB, num_slots x bev_height x bev_width x 3
        try:
            creep_config = self.cfg.creep_settings
        except:
            raise ValueError("Creep settings not found in config")

        bev_crop = self.config.training["bev_crop"]
        bev_size = self.config.training["bev_crop_size"]
        px_per_meter = self.config.training["pixels_per_meter"]

        safety_distance_x = creep_config.get("safety_distance_x", 2.0) * px_per_meter
        safety_distance_y = creep_config.get("safety_distance_y", 8.0) * px_per_meter

        x_min = int(-safety_distance_x / 2 + bev_size / 2)
        x_max = int(safety_distance_x / 2 + bev_size / 2)

        if bev_crop == "center":
            y_max = int(bev_size / 2)
        else:
            y_max = int(bev_size)

        y_min = int(y_max - safety_distance_y)

        # plot and save bevslots to ./bevslots_step.png
        orig = bevslots.numpy()

        bevslots = bevslots[:, :, y_min:y_max, x_min:x_max]
        bevslots = bevslots[0]

        bevslots = get_binary_mask_from_rgb(bevslots)

        # Get the number of pixels that are occupied
        num_occupied_pixels = bevslots.sum()

        # if num_occupied_pixels > 20 or self.step % 50 == 0:
        #     plt.imshow(np.transpose(orig, (0, 2, 3, 1))[0, y_min:y_max, x_min:x_max, :])
        #     plt.savefig(f"./bevslots_{self.step}.png")
        #     plt.close()
        #     plt.imshow(np.transpose(orig, (0, 2, 3, 1))[0])
        #     plt.savefig(f"./bevslots_{self.step}_1.png")
        #     plt.close()

        if num_occupied_pixels > 20:
            # print("Vehicle hazard detected")
            return True
        else:
            return False

    def check_vehicle_hazard_in_bevobject(self, model, bevobject, bevobjecttype):
        try:
            creep_config = self.cfg.creep_settings
        except:
            raise ValueError("Creep settings not found in config")

        safety_distance_x = int(creep_config.get("safety_distance_x", 2.0) * 5)
        safety_distance_y = int(creep_config.get("safety_distance_y", 8.0) * 5)

        bevobjecttype = bevobjecttype.clone()

        # Make non-vehicle classes 0 so we only retain vehicles
        bevobjecttype[bevobjecttype != 1] = 0

        renderer = BirdViewProducerObjectLevelRenderer(
            PixelDimensions(192, 192),
            pixels_per_meter=5,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        )

        bev = model.bev_encoder.interpret(bevobject, bevobjecttype, renderer=renderer)

        bev_size = 192

        x_min = int(-safety_distance_x / 2 + bev_size / 2)
        x_max = int(safety_distance_x / 2 + bev_size / 2)

        y_max = int(bev_size / 2)

        y_min = int(y_max - safety_distance_y)

        bev = bev[0, 0, y_min:y_max, x_min:x_max, :]

        bev = bev.sum(axis=-1) > 0

        # Get the number of pixels that are occupied
        num_occupied_pixels = bev.sum()
        # if num_occupied_pixels > 20 or self.step % 50 == 0:
        #     plt.imshow(bev.astype(np.float32))
        #     plt.savefig(f"./bevslots_{self.step}.png")
        #     plt.close()
        #     renderer2 = BirdViewProducerObjectLevelRenderer(
        #         PixelDimensions(192, 192),
        #         pixels_per_meter=5,
        #         crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        #     )
        #     bev2 = model.bev_encoder.interpret(
        #         bevobject, bevobjecttype, renderer=renderer2
        #     )
        #     plt.imshow(bev2[0, 0])
        #     plt.savefig(f"./bevslots_{self.step}_1.png")
        #     plt.close()

        if num_occupied_pixels > 20:
            return True
        else:
            return False

    def destroy(self):
        super().destroy()
        if self.scenario_logger:
            self.scenario_logger.dump_to_json()
            del self.scenario_logger

        if SAVE_GIF == True and (self.exec_or_inter == "inter"):
            self.save_path_mask_vid = f"viz_vid/masked"
            self.save_path_org_vid = f"viz_vid/org"
            Path(self.save_path_mask_vid).mkdir(parents=True, exist_ok=True)
            Path(self.save_path_org_vid).mkdir(parents=True, exist_ok=True)
            out_name_mask = f"{self.save_path_mask_vid}/{self.route_index}.mp4"
            out_name_org = f"{self.save_path_org_vid}/{self.route_index}.mp4"
            cmd_mask = f"ffmpeg -r 25 -i {self.save_path_mask}/%d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {out_name_mask}"
            cmd_org = f"ffmpeg -r 25 -i {self.save_path_org}/%d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {out_name_org}"
            print(cmd_mask)
            os.system(cmd_mask)
            print(cmd_org)
            os.system(cmd_org)

            # delete the images
            os.system(f"rm -rf {Path(self.save_path_mask).parent}")

        # del self.net
        del self.model

    def control_pid(self, waypoints, velocity, is_stuck=False):
        """Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            velocity (tensor): speedometer input
        """
        waypoints = waypoints.data.cpu().numpy()
        # when training we transform the waypoints to lidar coordinate, so we need to change is back when control
        waypoints[:, 0] += 1.3

        speed = velocity[0].data.cpu().numpy()

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        if is_stuck:
            desired_speed = np.array(4.0)  # default speed of 14.4 km/h

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0
        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90
        if brake:
            angle = 0.0
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        return steer, throttle, brake


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = []
        self.n = n
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        if len(self._window) > self.n:
            self._window.pop(0)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


def create_BEV(labels_org, gt_traffic_light_hazard, target_point, pred_wp, pix_per_m=5):
    pred_wp = np.array(pred_wp.squeeze())
    s = 0
    max_d = 30
    size = int(max_d * pix_per_m * 2)
    origin = (size // 2, size // 2)
    PIXELS_PER_METER = pix_per_m

    color = [(255), (255)]

    # create black image
    image_0 = Image.new("L", (size, size))
    image_1 = Image.new("L", (size, size))
    image_2 = Image.new("L", (size, size))
    vel_array = np.zeros((size, size))
    draw0 = ImageDraw.Draw(image_0)
    draw1 = ImageDraw.Draw(image_1)
    draw2 = ImageDraw.Draw(image_2)

    draws = [draw0, draw1, draw2]
    imgs = [image_0, image_1, image_2]

    for ix, sequence in enumerate([labels_org]):
        # features = rearrange(features, '(vehicle features) -> vehicle features', features=4)
        for ixx, vehicle in enumerate(sequence):
            x = -vehicle["position"][1] * PIXELS_PER_METER + origin[1]
            y = -vehicle["position"][0] * PIXELS_PER_METER + origin[0]
            yaw = vehicle["yaw"] * 180 / 3.14159265359
            extent_x = vehicle["extent"][2] * PIXELS_PER_METER / 2
            extent_y = vehicle["extent"][1] * PIXELS_PER_METER / 2
            origin_v = (x, y)

            if vehicle["class"] == "Car":
                p1, p2, p3, p4 = get_coords_BB(x, y, yaw - 90, extent_x, extent_y)
                if ixx == 0:
                    for ix in range(3):
                        draws[ix].polygon(
                            (p1, p2, p3, p4), outline=color[0]
                        )  # , fill=color[ix])
                    ix = 0
                else:
                    draws[ix].polygon(
                        (p1, p2, p3, p4), outline=color[ix]
                    )  # , fill=color[ix])

                if "speed" in vehicle:
                    vel = vehicle["speed"] * 3  # /3.6 # in m/s # just for visu
                    endx1, endy1, endx2, endy2 = get_coords(x, y, yaw - 90, vel)
                    draws[ix].line(
                        (endx1, endy1, endx2, endy2), fill=color[ix], width=2
                    )

            elif vehicle["class"] == "Route":
                ix = 1
                image = np.array(imgs[ix])
                point = (int(x), int(y))
                cv2.circle(image, point, radius=3, color=color[0], thickness=3)
                imgs[ix] = Image.fromarray(image)

    for wp in pred_wp:
        x = wp[1] * PIXELS_PER_METER + origin[1]
        y = -wp[0] * PIXELS_PER_METER + origin[0]
        image = np.array(imgs[2])
        point = (int(x), int(y))
        cv2.circle(image, point, radius=2, color=255, thickness=2)
        imgs[2] = Image.fromarray(image)

    image = np.array(imgs[0])
    image1 = np.array(imgs[1])
    image2 = np.array(imgs[2])
    x = target_point[0][1] * PIXELS_PER_METER + origin[1]
    y = -(target_point[0][0]) * PIXELS_PER_METER + origin[0]
    point = (int(x), int(y))
    cv2.circle(image, point, radius=2, color=color[0], thickness=2)
    cv2.circle(image1, point, radius=2, color=color[0], thickness=2)
    cv2.circle(image2, point, radius=2, color=color[0], thickness=2)
    imgs[0] = Image.fromarray(image)
    imgs[1] = Image.fromarray(image1)
    imgs[2] = Image.fromarray(image2)

    images = [np.asarray(img) for img in imgs]
    image = np.stack([images[0], images[2], images[1]], axis=-1)
    BEV = image

    img_final = Image.fromarray(image.astype(np.uint8))
    if gt_traffic_light_hazard:
        color = "red"
    else:
        color = "green"
    img_final = ImageOps.expand(img_final, border=5, fill=color)

    ## add rgb image and lidar
    # all_images = np.concatenate((images_lidar, np.array(img_final)), axis=1)
    # all_images = np.concatenate((rgb_image, all_images), axis=0)
    all_images = img_final

    Path(f"bev_viz").mkdir(parents=True, exist_ok=True)
    # Saving to folder:
    # Print the folders absolute path
    all_images.save(f"bev_viz/{time.time()}_{s}.png")

    # return BEV


def get_coords(x, y, angle, vel):
    length = vel
    endx2 = x + length * math.cos(math.radians(angle))
    endy2 = y + length * math.sin(math.radians(angle))

    return x, y, endx2, endy2


def get_coords_BB(x, y, angle, extent_x, extent_y):
    endx1 = (
        x
        - extent_x * math.sin(math.radians(angle))
        - extent_y * math.cos(math.radians(angle))
    )
    endy1 = (
        y
        + extent_x * math.cos(math.radians(angle))
        - extent_y * math.sin(math.radians(angle))
    )

    endx2 = (
        x
        + extent_x * math.sin(math.radians(angle))
        - extent_y * math.cos(math.radians(angle))
    )
    endy2 = (
        y
        - extent_x * math.cos(math.radians(angle))
        - extent_y * math.sin(math.radians(angle))
    )

    endx3 = (
        x
        + extent_x * math.sin(math.radians(angle))
        + extent_y * math.cos(math.radians(angle))
    )
    endy3 = (
        y
        - extent_x * math.cos(math.radians(angle))
        + extent_y * math.sin(math.radians(angle))
    )

    endx4 = (
        x
        - extent_x * math.sin(math.radians(angle))
        + extent_y * math.cos(math.radians(angle))
    )
    endy4 = (
        y
        + extent_x * math.cos(math.radians(angle))
        + extent_y * math.sin(math.radians(angle))
    )

    return (endx1, endy1), (endx2, endy2), (endx3, endy3), (endx4, endy4)
