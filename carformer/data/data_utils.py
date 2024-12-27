import torch


OBJECT_TYPE_MAPPING = {
    "vehicles": 1,
    "tlights": 2,
    "waypoints": 3,
    "padding": 0,
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
                        dtype=torch.float32,
                    ),
                    torch.full(
                        (len(flattened_list[i]["tlights"]),),
                        OBJECT_TYPE_MAPPING["tlights"],
                        dtype=torch.float32,
                    ),
                    torch.full(
                        (
                            object_level_max_num_objects
                            - len(flattened_list[i]["vehicles"])
                            - len(flattened_list[i]["tlights"]),
                        ),
                        OBJECT_TYPE_MAPPING["padding"],
                        dtype=torch.float32,
                    ),
                    torch.full(
                        (len(flattened_list[i]["waypoints"]),),
                        OBJECT_TYPE_MAPPING["waypoints"],
                        dtype=torch.float32,
                    ),
                    torch.full(
                        (
                            object_level_max_route_length
                            - len(flattened_list[i]["waypoints"]),
                        ),
                        OBJECT_TYPE_MAPPING["padding"],
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
            dct["yaw"],
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
            dct["yaw"],
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
        crop_type=BirdViewCropType.FRONT_AREA_ONLY
        if bev_crop == "front"
        else BirdViewCropType.FRONT_AND_REAR_AREA,
    )

    def obj_filter(objs):
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


import numpy as np


def transform_waypoints(waypoints):
    """transform waypoints to be origin at ego_matrix"""

    # TODO should transform to virtual lidar coordicate?
    # T = get_vehicle_to_virtual_lidar_transform()

    vehicle_matrix = np.array(waypoints[0])
    vehicle_matrix_inv = np.linalg.inv(vehicle_matrix)
    for i in range(
        0, len(waypoints)
    ):  # TODO: Start from 1 because 0 is ego vehicle initial position
        matrix = np.array(waypoints[i])
        waypoints[i] = vehicle_matrix_inv @ matrix

    return waypoints


def transform_route(targets, ego_matrix):
    """transform route to be origin at ego_matrix"""
    z_ego = ego_matrix[2][3]
    targets = np.array(targets)

    # Targets are only in the xy plane, add z coordinate as z_ego
    targets = np.hstack(
        (
            targets[:, 1:],
            -targets[:, :1],
            np.ones((targets.shape[0], 1)) * z_ego,
            np.ones((targets.shape[0], 1)),
        )
    )  # Shape: (N, 4)

    # Transform to vehicle coordinate
    targets = np.linalg.inv(ego_matrix) @ targets.T  # Shape: (4, N)

    return targets.T[:, :2]
