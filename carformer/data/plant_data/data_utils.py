import torch
import numpy as np
from copy import deepcopy

PLANT_TO_CARFORMER_OBJECT_TYPE_MAPPING = {
    "Car": "Vehicle",
    "Route": "Route",
    "Lane": "",
    "Light": "TrafficLight",
}

OBJECT_TYPE_MAPPING = {
    "vehicles": 1,
    "tlights": 2,
    "waypoints": 3,
    "padding": 0,
}

TARGET_OBJECT_TYPE_MAPPING = {
    "vehicle": 1,
    "invalid_vehicle": 0,
    "padding": -100,
}
OBJECT_TYPE_MAPPING_INV = {
    1: "Vehicle",
    2: "TrafficLight",
    3: "Route",
    0: "padding",
}


def postprocess_bev_objects(
    flattened_list,
    object_level_max_num_objects=10,
    object_level_max_route_length=4,
):
    # print(flattened_list)
    assert all(
        [
            len(flattened_list[i]["vehicles"]) + len(flattened_list[i]["tlights"])
            <= object_level_max_num_objects
            for i in range(len(flattened_list))
        ]
    ), "Number of objects exceeds max number of objects"
    assert all(
        [
            len(flattened_list[i]["waypoints"]) <= object_level_max_route_length
            for i in range(len(flattened_list))
        ]
    ), "Route length exceeds max route length"

    objects = torch.cat(
        [
            torch.cat(
                [
                    torch.from_numpy(flattened_list[i]["vehicles"]),
                    torch.from_numpy(flattened_list[i]["tlights"]),
                    torch.zeros(
                        (
                            object_level_max_num_objects
                            - len(flattened_list[i]["vehicles"])
                            - len(flattened_list[i]["tlights"]),
                            6,
                        ),
                        dtype=torch.float32,
                    ),
                    torch.from_numpy(flattened_list[i]["waypoints"]),
                    torch.zeros(
                        (
                            object_level_max_route_length
                            - len(flattened_list[i]["waypoints"]),
                            6,
                        ),
                        dtype=torch.float32,
                    ),
                ],
                dim=0,
            ).unsqueeze(0)
            for i in range(len(flattened_list))
        ]
    )

    types = torch.cat(
        [
            torch.cat(
                [
                    torch.full(
                        (len(flattened_list[i]["vehicles"]),),
                        OBJECT_TYPE_MAPPING["vehicles"],
                    ),
                    torch.full(
                        (len(flattened_list[i]["tlights"]),),
                        OBJECT_TYPE_MAPPING["tlights"],
                    ),
                    torch.full(
                        (
                            object_level_max_num_objects
                            - len(flattened_list[i]["vehicles"])
                            - len(flattened_list[i]["tlights"]),
                        ),
                        OBJECT_TYPE_MAPPING["padding"],
                    ),
                    torch.full(
                        (len(flattened_list[i]["waypoints"]),),
                        OBJECT_TYPE_MAPPING["waypoints"],
                    ),
                    torch.full(
                        (
                            object_level_max_route_length
                            - len(flattened_list[i]["waypoints"]),
                        ),
                        OBJECT_TYPE_MAPPING["padding"],
                    ),
                ],
                dim=0,
            ).unsqueeze(0)
            for i in range(len(flattened_list))
        ]
    ).long()

    return objects, types


def postprocess_bev_targets(
    flattened_list,
    object_level_max_num_objects=10,
    object_level_max_route_length=4,
):
    assert all(
        [
            len(flattened_list[i]["bevobject"]) <= object_level_max_num_objects
            for i in range(len(flattened_list))
        ]
    ), "Number of objects exceeds max number of objects"

    objects = torch.cat(
        [
            torch.cat(
                [
                    torch.from_numpy(flattened_list[i]["bevobject"]),
                    torch.zeros(
                        (
                            object_level_max_num_objects
                            + object_level_max_route_length
                            - len(flattened_list[i]["bevobject"]),
                            6,
                        ),
                        dtype=torch.float32,
                    ),
                ],
                dim=0,
            ).unsqueeze(0)
            for i in range(len(flattened_list))
        ]
    )

    types = torch.cat(
        [
            torch.cat(
                [
                    torch.from_numpy(flattened_list[i]["bevobjecttype"]),
                    torch.full(
                        (
                            object_level_max_num_objects
                            + object_level_max_route_length
                            - len(flattened_list[i]["bevobjecttype"]),
                        ),
                        TARGET_OBJECT_TYPE_MAPPING["padding"],
                        dtype=torch.float32,
                    ),
                ],
                dim=0,
            ).unsqueeze(0)
            for i in range(len(flattened_list))
        ]
    ).long()

    return objects, types


DEFAULT_TRAFFIC_LIGHT_MAPPING = {"red": 0, "yellow": 1, "green": 2}
DEFAULT_TRAFFIC_LIGHT_MAPPING_INV = {0: "red", 1: "yellow", 2: "green"}


def to_object_level_vector(dct, traffic_light_mapping=DEFAULT_TRAFFIC_LIGHT_MAPPING):
    """
    Convert a dictionary representation of an object to a vector representation.
    """

    if dct["class"] == "Vehicle":
        vector_rep = [
            dct["position"][0],
            dct["position"][1],
            dct["extent"][1],
            dct["extent"][2],
            normalize_angle(dct["yaw"]),
            dct["speed"],
        ]
    elif dct["class"] == "TrafficLight":
        state_idx = traffic_light_mapping[dct["state"]]

        vector_rep = [
            dct["position"][0],
            dct["position"][1],
            state_idx == 0,
            state_idx == 1,
            state_idx == 2,
            1.0,
        ]
    elif dct["class"] == "Route":
        vector_rep = [
            dct["position"][0],
            dct["position"][1],
            dct["extent"][1],
            dct["extent"][2],
            normalize_angle(dct["yaw"]),
            dct["id"],
        ]
    elif dct["class"] == "Pedestrian":
        # vector_rep = [
        #     dct["position"][0],
        #     dct["position"][1],
        #     dct["extent"][1],
        #     dct["extent"][2],
        #     dct["yaw"],
        #     dct["speed"],
        # ]
        return None
    else:
        raise ValueError("Invalid object class: {}".format(dct["class"]))

    return vector_rep


def from_object_level_vector(
    vector, object_type_id, traffic_light_mapping=DEFAULT_TRAFFIC_LIGHT_MAPPING_INV
):
    """
    Convert a vector representation of an object to a dictionary representation.
    """
    if OBJECT_TYPE_MAPPING_INV[object_type_id] == "Vehicle":
        dct = {
            "class": "Vehicle",
            "position": [vector[0], vector[1], 0.0],
            "extent": [0.0, vector[2], vector[3]],
            "yaw": vector[4],
            "speed": vector[5],
            "ego_vehicle": False,  # Always false for now, TODO: Think if this is alright
        }
    elif OBJECT_TYPE_MAPPING_INV[object_type_id] == "TrafficLight":
        idx = vector[2:5].argmax()
        dct = {
            "class": "TrafficLight",
            "position": [vector[0], vector[1], 0.0],
            "state": traffic_light_mapping[idx],
        }
    elif OBJECT_TYPE_MAPPING_INV[object_type_id] == "Route":
        dct = {
            "class": "Route",
            "position": [vector[0], vector[1], 0.0],
            "extent": [0.0, vector[2], vector[3]],
            "yaw": vector[4],
            "id": vector[5],
        }
    elif OBJECT_TYPE_MAPPING_INV[object_type_id] == "padding":
        dct = None
    else:
        raise ValueError("Invalid object type: {}".format(object_type_id))

    return dct


from carla_birdeye_view import (
    BirdViewProducerObjectLevelRenderer,
    BirdViewCropType,
    PixelDimensions,
)


def get_slots_preprocessing_function_from_config(config):
    num_slots = config.num_slots
    bev_crop_size = config.bev_crop_size
    bev_crop_type = config.bev_crop
    try:
        legacy = config.get("legacy", True)
    except:
        if hasattr(config, "legacy"):
            legacy = getattr(config, "legacy")
        else:
            legacy = True

    return get_slots_preprocessing_function(
        num_slots,
        bev_crop_size,
        legacy,
        bev_crop_type,
    )


def get_slots_preprocessing_function(
    num_slots, bev_crop_size, legacy=False, bev_crop_type="center"
):
    print("LEGACY SLOT RENDERER: ", legacy)
    vehicle_renderer = VehicleSlotRenderer(
        num_slots, bev_crop_size, legacy=legacy, bev_crop_type=bev_crop_type
    )

    slots_preprocessing_function = (
        lambda x: vehicle_renderer.get_ultimate_vehicle_colors(x).unsqueeze(0)
    )

    return slots_preprocessing_function


def get_object_level_filter_from_config(config):
    return get_object_level_filter(
        config.bev_crop_size,
        config.pixels_per_meter,
        config.bev_crop,
        config.split_long_routes,
        config.split_threshold,
        config.include_agent_in_object_level,
        config.include_traffic_lights_in_object_level,
        config.object_level_max_num_objects,
        config.object_level_max_route_length,
        config.sort_by_distance,
    )


def get_object_level_filter(
    img_size,
    pixels_per_meter,
    bev_crop,
    split_long_routes,
    split_threshold,
    include_agent_in_object_level,
    include_traffic_lights_in_object_level,
    object_level_max_num_objects,
    object_level_max_route_length,
    sort_by_distance,
):
    renderer = BirdViewProducerObjectLevelRenderer(
        PixelDimensions(img_size, img_size),
        pixels_per_meter=pixels_per_meter,
        crop_type=(
            BirdViewCropType.FRONT_AREA_ONLY
            if bev_crop == "front"
            else BirdViewCropType.FRONT_AND_REAR_AREA
        ),
    )

    def obj_filter(objs):
        # PlanT only: First object is always ego, so skip if flag on
        if not include_agent_in_object_level:
            objs = objs[1:]

        if split_long_routes:
            objs = renderer.split_long_routes(objs, split_threshold)

        return renderer.filter_objects_in_scene(
            objs,
            exclude_ego_vehicle=not include_agent_in_object_level,
            exclude_lights=not include_traffic_lights_in_object_level,
            object_limit=object_level_max_num_objects,
            route_limit=object_level_max_route_length,
            sort_by_distance=sort_by_distance,
        )

    return obj_filter


def plant_to_carformer_object(plant_obj, pos_offset=[1.3, 0, 2.5]):
    carformer_obj = {}
    carformer_obj["class"] = PLANT_TO_CARFORMER_OBJECT_TYPE_MAPPING[plant_obj["class"]]
    if not carformer_obj["class"].strip():
        return carformer_obj
    carformer_obj["position"] = plant_obj["position"]
    for i in range(3):
        carformer_obj["position"][i] += pos_offset[i]
    carformer_obj["extent"] = plant_obj["extent"]
    carformer_obj["yaw"] = plant_obj["yaw"]
    if "speed" in plant_obj:
        carformer_obj["speed"] = plant_obj["speed"]

    if (
        carformer_obj["position"][0] == 0.0
        and carformer_obj["position"][1] == 0.0
        and carformer_obj["position"][2] == 0.0
    ):
        carformer_obj["ego_vehicle"] = True
    else:
        carformer_obj["ego_vehicle"] = False

    carformer_obj["id"] = plant_obj["id"]

    return carformer_obj


def get_virtual_lidar_to_vehicle_transform():
    # This is a fake lidar coordinate
    T = np.eye(4)
    T[0, 3] = 1.3
    T[1, 3] = 0.0
    T[2, 3] = 2.5
    return T


def get_vehicle_to_virtual_lidar_transform():
    return np.linalg.inv(get_virtual_lidar_to_vehicle_transform())


def transform_waypoints(waypoints):
    """transform waypoints to be origin at ego_matrix"""

    # TODO should transform to virtual lidar coordicate?
    T = get_vehicle_to_virtual_lidar_transform()

    vehicle_matrix = np.array(waypoints[0])
    vehicle_matrix_inv = np.linalg.inv(vehicle_matrix)
    for i in range(1, len(waypoints)):
        matrix = np.array(waypoints[i])
        waypoints[i] = T @ vehicle_matrix_inv @ matrix

    return waypoints


def transform_objects(objects, ego_matrix):
    """transform waypoints to be origin at ego_matrix"""

    # TODO should transform to virtual lidar coordicate?
    T = get_vehicle_to_virtual_lidar_transform()

    vehicle_matrix = np.array(ego_matrix)
    vehicle_matrix_inv = np.linalg.inv(vehicle_matrix)
    transformed_objects = deepcopy(objects)
    for i in range(len(objects)):
        # Create obj_matrix from position and yaw, assume all non-yaw rotation is 0
        obj_matrix = np.eye(4)
        obj_matrix[:3, 3] = objects[i]["position"]
        obj_matrix[:3, :3] = np.array(
            [
                [np.cos(objects[i]["yaw"]), -np.sin(objects[i]["yaw"]), 0],
                [np.sin(objects[i]["yaw"]), np.cos(objects[i]["yaw"]), 0],
                [0, 0, 1],
            ]
        )

        matrix = np.array(obj_matrix)

        transformed_obj_matrix = T @ vehicle_matrix_inv @ matrix

        transformed_pos = transformed_obj_matrix[:3, 3]

        transformed_yaw = np.arctan2(
            transformed_obj_matrix[1, 0], transformed_obj_matrix[0, 0]
        )

        transformed_objects[i]["position"] = transformed_pos
        transformed_objects[i]["yaw"] = transformed_yaw

    return transformed_objects


def normalize_angle(x):
    x = x % (2 * np.pi)  # force in range [0, 2 pi)
    if x > np.pi:  # move to [-pi, pi)
        x -= 2 * np.pi
    return x


def transform_vehicles_according_to_egos(vehicles, future_ego_matrix, past_ego_matrix):
    vehicles = torch.from_numpy(vehicles)
    future_ego_matrix = torch.from_numpy(future_ego_matrix)
    past_ego_matrix = torch.from_numpy(past_ego_matrix)

    # Transform future vehicle 6d vectors to be relative to ego in the past
    # 6d vectors where 0, 1 are x, y and 4 is yaw. Assume z, roll, pitch are 0
    vehicle_pose_6dof_vectors = torch.zeros((len(vehicles), 6), dtype=torch.float32)
    vehicle_pose_6dof_vectors[:, 0] = vehicles[:, 0]
    vehicle_pose_6dof_vectors[:, 1] = -vehicles[:, 1]
    vehicle_pose_6dof_vectors[:, -1] = vehicles[:, 4]

    pose_matrices = pose_vec2mat(vehicle_pose_6dof_vectors)

    T_offset = get_virtual_lidar_to_vehicle_transform()
    T_rev_offset = get_vehicle_to_virtual_lidar_transform()

    # Transform future vehicle pose matrices to be relative to ego in the past
    # transformed_pose_matrices = np.matmul(
    #     np.linalg.inv(past_ego_matrix), np.matmul(future_ego_matrix, pose_matrices)
    # )
    transformed_pose_matrices = torch.from_numpy(
        np.linalg.inv(past_ego_matrix)
        @ future_ego_matrix.numpy()
        @ pose_matrices.numpy()
    )

    # Transform future vehicle pose matrices to be 6d vectors
    transformed_pose_6dof_vectors = mat2pose_vec(transformed_pose_matrices)

    transformed_vehicles = vehicles.clone()
    transformed_vehicles[:, 0] = transformed_pose_6dof_vectors[:, 0]
    transformed_vehicles[:, 1] = -transformed_pose_6dof_vectors[:, 1]
    transformed_vehicles[:, 4] = transformed_pose_6dof_vectors[:, -1]

    # Return as numpy array
    return transformed_vehicles.numpy()


def pose_vec2mat(vec: torch.Tensor):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz [B,6]
    Returns:
        A transformation matrix [B,4,4]
    """
    translation = vec[..., :3].unsqueeze(-1)  # [...x3x1]
    rot = vec[..., 3:].contiguous()  # [...x3]
    rot_mat = euler2mat(rot)  # [...,3,3]
    transform_mat = torch.cat([rot_mat, translation], dim=-1)  # [...,3,4]
    transform_mat = torch.nn.functional.pad(
        transform_mat, [0, 0, 0, 1], value=0
    )  # [...,4,4]
    transform_mat[..., 3, 3] = 1.0
    return transform_mat


def euler2mat(angle: torch.Tensor):
    """Convert euler angles to rotation matrix.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) [Bx3]
    Returns:
        Rotation matrix corresponding to the euler angles [Bx3x3]
    """
    shape = angle.shape
    angle = angle.view(-1, 3)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)
    zmat = torch.stack(
        [cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], dim=1
    ).view(-1, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack(
        [cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=1
    ).view(-1, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack(
        [ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], dim=1
    ).view(-1, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    rot_mat = rot_mat.view(*shape[:-1], 3, 3)
    return rot_mat


def mat2pose_vec(matrix: torch.Tensor):
    """
    Converts a 4x4 pose matrix into a 6-dof pose vector
    Args:
        matrix (ndarray): 4x4 pose matrix
    Returns:
        vector (ndarray): 6-dof pose vector comprising translation components (tx, ty, tz) and
        rotation components (rx, ry, rz)
    """

    # M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    rotx = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])

    # M[0, 2] = +siny, M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    cosy = torch.sqrt(matrix[..., 1, 2] ** 2 + matrix[..., 2, 2] ** 2)
    roty = torch.atan2(matrix[..., 0, 2], cosy)

    # M[0, 0] = +cosy*cosz, M[0, 1] = -cosy*sinz
    rotz = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])

    rotation = torch.stack((rotx, roty, rotz), dim=-1)

    # Extract translation params
    translation = matrix[..., :3, 3]
    return torch.cat((translation, rotation), dim=-1)


# For slots
from enum import IntEnum


class RGB:
    RICH_BLACK = ((0, 72, 55),)
    GIRLY_PINK = ((255, 70, 131),)
    OLD_MAUVE = ((101, 38, 65),)
    MIDDLE_PURPLE = (203, 142, 176)
    VIOLET = (173, 127, 168)
    ORANGE = (252, 175, 62)
    CHOCOLATE = (233, 185, 110)
    CHAMELEON = (138, 226, 52)
    SKY_BLUE = (114, 159, 207)
    DIM_GRAY = (105, 105, 105)
    DARK_GRAY = (50, 50, 50)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    WHITE = (255, 255, 255)

    PALE_CYAN = (149, 217, 247)
    EUCALYPTUS = (59, 232, 176)
    FLORAL_LAVENDER = (161, 129, 217)
    BROWN = (164, 92, 39)
    NAVY_BLUE = (23, 126, 207)
    FLAVESCENT = (255, 227, 148)
    BABY_PINK = (2, 151, 67)


class VehicleMasks(IntEnum):
    SLOT13 = 13
    SLOT12 = 12
    SLOT11 = 11
    SLOT10 = 10
    SLOT9 = 9
    SLOT8 = 8
    SLOT7 = 7
    SLOT6 = 6
    SLOT5 = 5
    SLOT4 = 4
    SLOT3 = 3
    SLOT2 = 2
    SLOT1 = 1
    SLOT0 = 0

    @staticmethod
    def top_to_bottom():
        return list(VehicleMasks)

    @staticmethod
    def bottom_to_top():
        return list(reversed(VehicleMasks.top_to_bottom()))


LEGACY_VEHICLE_RGB_BY_MASK = {
    VehicleMasks.SLOT13: RGB.FLAVESCENT,
    VehicleMasks.SLOT12: RGB.RICH_BLACK,
    VehicleMasks.SLOT11: RGB.GIRLY_PINK,
    VehicleMasks.SLOT10: RGB.FLORAL_LAVENDER,
    VehicleMasks.SLOT9: RGB.NAVY_BLUE,
    VehicleMasks.SLOT8: RGB.BABY_PINK,
    VehicleMasks.SLOT7: RGB.RED,
    VehicleMasks.SLOT6: RGB.YELLOW,
    VehicleMasks.SLOT5: RGB.PALE_CYAN,  # GREEN,
    VehicleMasks.SLOT4: RGB.CHAMELEON,
    VehicleMasks.SLOT3: RGB.FLAVESCENT,
    VehicleMasks.SLOT2: RGB.BROWN,
    VehicleMasks.SLOT1: RGB.EUCALYPTUS,
    VehicleMasks.SLOT0: RGB.DIM_GRAY,
}
COLOR_ON = 1

VEHICLE_RGB_BY_MASK = [
    RGB.RICH_BLACK,
    RGB.GIRLY_PINK,
    RGB.FLORAL_LAVENDER,
    RGB.NAVY_BLUE,
    RGB.BABY_PINK,
    RGB.RED,
    RGB.YELLOW,
    RGB.PALE_CYAN,
    RGB.CHAMELEON,
    RGB.FLAVESCENT,
    RGB.BROWN,
    RGB.EUCALYPTUS,
    RGB.DIM_GRAY,
]

from .compatibility import BirdViewProducerObjectLevelRenderer as slotsrenderer


class VehicleColorAssigner:
    def __init__(self):
        pass

    @staticmethod
    def as_rgb(birdview, num_slot, legacy=False):
        if legacy:
            return VehicleColorAssigner.as_rgb_legacy(birdview, num_slot)
        _, h, w = birdview.shape
        rgb_canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        nonzero_indices = lambda arr: arr == COLOR_ON

        # for mask_type in VehicleMasks.bottom_to_top()[:num_slot]:
        for slot_idx in range(num_slot):
            mask_idx = slot_idx % len(VEHICLE_RGB_BY_MASK)
            rgb_color = VEHICLE_RGB_BY_MASK[mask_idx]
            mask = birdview[slot_idx]
            # If mask above contains 0, don't overwrite content of canvas (0 indicates transparency)
            rgb_canvas[nonzero_indices(mask)] = rgb_color
        return rgb_canvas

    @staticmethod
    def as_rgb_legacy(birdview, num_slot):
        # TODO: fix it for batch
        _, h, w = birdview.shape
        rgb_canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        nonzero_indices = lambda arr: arr == COLOR_ON

        for mask_type in VehicleMasks.bottom_to_top()[:num_slot]:
            rgb_color = LEGACY_VEHICLE_RGB_BY_MASK[mask_type]
            mask = birdview[mask_type]
            # If mask above contains 0, don't overwrite content of canvas (0 indicates transparency)
            rgb_canvas[nonzero_indices(mask)] = rgb_color
        return rgb_canvas


from .compatibility import BirdViewCropType as compat_BirdViewCropType


class VehicleSlotRenderer:
    def __init__(
        self, num_slots, res=400, object="Vehicle", legacy=True, bev_crop_type="center"
    ):
        if bev_crop_type == "center":
            crop_type = compat_BirdViewCropType.FRONT_AND_REAR_AREA
        elif bev_crop_type == "front":
            crop_type = compat_BirdViewCropType.FRONT_AREA_ONLY
        else:
            raise ValueError(f"Invalid crop area: {bev_crop_type}")

        # Assert object is a string
        assert isinstance(object, str)

        # Assert legacy is a boolean
        assert isinstance(legacy, bool)

        self.renderer = slotsrenderer(
            PixelDimensions(res, res),
            5,
            num_slots=num_slots,
            crop_type=crop_type,
        )

        self.filter_fnc = lambda x: object == x["class"]
        self.legacy = legacy

        self.res = res
        self.num_slots = num_slots

    def get_ultimate_vehicle_colors(self, bev):
        # bev = json.load(open(path))['bev']
        bev_slots = self.renderer.as_slots(
            bev, randomize=False, filter_lambda=self.filter_fnc
        )

        assert bev_slots.shape == (self.res, self.res, self.num_slots)

        bev_slots = torch.from_numpy(bev_slots).permute(2, 0, 1)  # (num_slots,H,W)

        colorful_slots = assign_colors(bev_slots, self.num_slots)

        norm_colorful_slots = colorful_slots / 255

        # norm_colorful_slots[norm_colorful_slots==0]=1  # make background white

        assert norm_colorful_slots.shape == (3, self.res, self.res)

        return norm_colorful_slots


def assign_colors(slots, num_slots, legacy=False):
    result = torch.tensor(
        np.array(
            np.transpose(
                VehicleColorAssigner.as_rgb(
                    slots.sigmoid().detach().cpu().numpy() > 0.5, num_slots, legacy
                ),
                (2, 0, 1),
            )
        )
    ).float()
    return result


########################################################################

############## WARPING

###################################################################3####


def compute_relative_transform(origin, current, pix_per_meter=5):
    result = torch.bmm(torch.linalg.inv(origin), (current))

    return result


def get_affine_grid_transform(origin, current, inp_size=400, pix_per_meter=5):
    relative_transform = compute_relative_transform(origin, current, pix_per_meter)

    translation = relative_transform[:, :2, 3:] / ((inp_size / 2) / pix_per_meter)
    translation[:, [0, 1]] = translation[:, [1, 0]]

    affine_grid_transform = torch.cat(
        (torch.transpose(relative_transform[:, :2, :2], 2, 1), translation), axis=2
    )

    # rot x, y. dont take height.
    # affine_grid_transform = torch.from_numpy(affine_grid_transform).float()

    return affine_grid_transform


def warp_sequence(x, ego_matrices, mode="nearest", spatial_extent=None):
    """
    Batch-compatible warping function.

    Warps a sequence based on the first frame using ego vehicle transformation matrices.
    """
    sequence_length = x.shape[1]
    if sequence_length == 1:
        return x

    out = [x[:, 0]]

    # print('X. SHAPE ', x.shape)
    base_frame = ego_matrices[:, 0]  # torch.from_numpy()

    for t in range(1, sequence_length):
        curr_frame = ego_matrices[:, t]  # torch.from_numpy()
        aff_grid = get_affine_grid_transform(
            base_frame, curr_frame, inp_size=x.shape[-1], pix_per_meter=5
        )  # .unsqueeze(0)

        grid = torch.nn.functional.affine_grid(
            aff_grid, size=x[:, 0].shape, align_corners=False
        )

        warped_bev = torch.nn.functional.grid_sample(
            (x[:, t]),
            grid.float(),
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )

        out.append(warped_bev)

    return torch.stack(out, 1)


### Forecast utils
def extract_forecast_targets(
    timesteps,
    context_length,
    future_horizon,
    forecast_steps=1,
    use_slots=True,
    object_level=True,
):
    assert len(timesteps) == context_length + future_horizon
    # print("In extract forecast targets: ", use_slots)
    if use_slots:
        assert "bevslots" in timesteps[0].state
        timesteps[context_length - 1].state["bevslots"] = np.concatenate(
            [
                t.state["bevslots"]
                for t in timesteps[context_length - 1 : context_length + forecast_steps]
            ],
            axis=0,
        )
        if "bevslotspercept" in timesteps[0].state:
            timesteps[context_length - 1].state["bevslotspercept"] = np.concatenate(
                [
                    t.state["bevslotspercept"]
                    for t in timesteps[
                        context_length - 1 : context_length + forecast_steps
                    ]
                ],
                axis=0,
            )
    elif object_level:
        assert "bevobjids" in timesteps[0].state

        for i in range(context_length):
            # future_vehicle_timestep = timesteps[i + future_horizon]
            future_vehicle_timestep = timesteps[i + forecast_steps]  # TODO: parametrize
            # future_vehicle_timestep = timesteps[i + 0]
            current_ids = timesteps[i].state["bevobjids"]

            future_vehicle_map = {
                id: vehicle
                for id, vehicle in zip(
                    future_vehicle_timestep.state["bevobjids"],
                    future_vehicle_timestep.state["bevobject"]["vehicles"],
                )
            }

            future_vehicles = []
            future_vehicles_mask = []
            for id in current_ids:
                if id in future_vehicle_map:
                    future_vehicles.append(future_vehicle_map[id])
                    future_vehicles_mask.append(OBJECT_TYPE_MAPPING["vehicles"])
                else:
                    future_vehicles.append(np.zeros(6, dtype=np.float32))
                    future_vehicles_mask.append(OBJECT_TYPE_MAPPING["padding"])

            if len(future_vehicles) > 0:
                future_vehicles = np.stack(future_vehicles, axis=0)
                future_vehicles_types = np.stack(future_vehicles_mask, axis=0)
            else:
                future_vehicles = np.zeros((0, 6), dtype=np.float32)
                future_vehicles_types = np.zeros((0,), dtype=np.int32)

            cur_ego = np.asarray(timesteps[i].action["ego_matrix"])
            future_ego = np.asarray(future_vehicle_timestep.action["ego_matrix"])

            future_vehicles = transform_vehicles_according_to_egos(
                future_vehicles, future_ego, cur_ego
            )

            timesteps[i].state["targetbevobject"] = {}
            timesteps[i].state["targetbevobject"]["bevobject"] = future_vehicles
            timesteps[i].state["targetbevobject"][
                "bevobjecttype"
            ] = future_vehicles_types
    else:
        for i in range(context_length):
            timesteps[i].state["targetbev"] = timesteps[i + forecast_steps].state["bev"]
