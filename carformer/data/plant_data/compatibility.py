# TODO: Merge with carla_birdeye_view

import logging
import numpy as np

from enum import IntEnum, auto, Enum

from typing import NamedTuple
from pathlib import Path
from typing import List
from filelock import FileLock
import cv2 as cv2

# Np print options set to 2 decimal places and suppress scientific notation
np.set_printoptions(precision=2, suppress=True)

import random


LOGGER = logging.getLogger(__name__)

DEFAULT_HEIGHT = 336  # its 84m when density is 4px/m
DEFAULT_WIDTH = 150  # its 37.5m when density is 4px/m

BirdView = np.ndarray  # [np.uint8] with shape (level, y, x)
RgbCanvas = np.ndarray  # [np.uint8] with shape (y, x, 3)


class RGB:
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


class BirdViewCropType(Enum):
    FRONT_AND_REAR_AREA = auto()  # Freeway mode
    FRONT_AREA_ONLY = auto()  # Like in "Learning by Cheating"


class BirdViewMasks(IntEnum):
    #    PEDESTRIANS = 8
    #    RED_LIGHTS = 7
    #    YELLOW_LIGHTS = 6
    #    GREEN_LIGHTS = 5
    #    AGENT = 4
    #    VEHICLES = 3
    #    CENTERLINES = 2
    PEDESTRIANS = 7
    RED_LIGHTS = 6
    YELLOW_LIGHTS = 5
    GREEN_LIGHTS = 4
    AGENT = 3
    VEHICLES = 2
    LANES = 1
    ROAD = 0

    @staticmethod
    def top_to_bottom() -> List[int]:
        return list(BirdViewMasks)

    @staticmethod
    def bottom_to_top() -> List[int]:
        return list(reversed(BirdViewMasks.top_to_bottom()))


RGB_BY_MASK = {
    BirdViewMasks.PEDESTRIANS: RGB.VIOLET,
    BirdViewMasks.RED_LIGHTS: RGB.RED,
    BirdViewMasks.YELLOW_LIGHTS: RGB.YELLOW,
    BirdViewMasks.GREEN_LIGHTS: RGB.GREEN,
    BirdViewMasks.AGENT: RGB.CHAMELEON,
    BirdViewMasks.VEHICLES: RGB.ORANGE,
    #    BirdViewMasks.CENTERLINES: RGB.CHOCOLATE,
    BirdViewMasks.LANES: RGB.WHITE,
    BirdViewMasks.ROAD: RGB.DIM_GRAY,
}

BIRDVIEW_SHAPE_CHW = (len(RGB_BY_MASK), DEFAULT_HEIGHT, DEFAULT_WIDTH)
BIRDVIEW_SHAPE_HWC = (DEFAULT_HEIGHT, DEFAULT_WIDTH, len(RGB_BY_MASK))


class Coord(NamedTuple):
    x: int
    y: int


class FltCoord(NamedTuple):
    x: float
    y: float


class Dimensions(NamedTuple):
    width: int
    height: int


PixelDimensions = Dimensions
Pixels = int
Meters = float
Canvas2D = np.ndarray  # of shape (y, x)


def rotate(image, angle, center=None, scale=1.0):
    assert image.dtype == np.uint8

    """Copy paste of imutils method but with INTER_NEAREST and BORDER_CONSTANT flags"""
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # return the rotated image
    return rotated


def circle_circumscribed_around_rectangle(rect_size: Dimensions) -> float:
    """Returns radius of that circle."""
    a = rect_size.width / 2
    b = rect_size.height / 2
    return float(np.sqrt(np.power(a, 2) + np.power(b, 2)))


def square_fitting_rect_at_any_rotation(rect_size: Dimensions) -> float:
    """Preview: https://pasteboard.co/J1XK62H.png"""
    radius = circle_circumscribed_around_rectangle(rect_size)
    side_length_of_square_circumscribed_around_circle = radius * 2
    return side_length_of_square_circumscribed_around_circle


class BirdViewProducerObjectLevelRenderer:
    def __init__(
        self,
        target_size: PixelDimensions,
        pixels_per_meter: int = 4,
        crop_type: BirdViewCropType = BirdViewCropType.FRONT_AND_REAR_AREA,
        num_slots: int = 10,
    ):
        self.target_size = target_size
        self._pixels_per_meter = pixels_per_meter
        self._crop_type = crop_type
        self._num_slots = num_slots

        # Divide target size by pixels_per_meter to get size in meters
        # self.target_size =
        # Calculate the radius of the circle circumscribed around the target_size rectangle
        self.radius = circle_circumscribed_around_rectangle(
            PixelDimensions(
                width=target_size.width / pixels_per_meter,
                height=target_size.height / pixels_per_meter,
            )
        )

        # print("Radius is " + str(self.radius))
        self._crop_type = crop_type

        # print("Crop type is " + str(crop_type))

        if crop_type is BirdViewCropType.FRONT_AND_REAR_AREA:
            # rendering_square_size = round(
            #     square_fitting_rect_at_any_rotation(self.target_size)
            # )

            self.origin = Coord(
                x=self.target_size.width // 2, y=self.target_size.height // 2
            )
        elif crop_type is BirdViewCropType.FRONT_AREA_ONLY:
            # We must keep rendering size from FRONT_AND_REAR_AREA (in order to avoid rotation issues)
            # enlarged_size = PixelDimensions(
            #     width=target_size.width, height=target_size.height * 2
            # )
            # rendering_square_size = round(
            #     square_fitting_rect_at_any_rotation(enlarged_size)
            # )
            self.origin = Coord(
                x=self.target_size.width // 2, y=self.target_size.height
            )
        else:
            raise NotImplementedError
        # self.rendering_area = PixelDimensions(
        #     width=rendering_square_size, height=rendering_square_size
        # )

        # Rotate by 90 degrees then translate by origin
        self.global_rotation_translation = np.array(
            [
                [0, 1, self.origin.x],
                [-1, 0, self.origin.y],
                [0, 0, 1],
            ]
        )

    def as_rgb(self, objects):
        """Converts a list of objects to RGB image."""
        canvas = np.zeros(
            shape=(self.target_size.width, self.target_size.height, 3), dtype=np.uint8
        )

        for object in objects:
            self.render(canvas, object)

        return canvas

    def render(self, canvas: RgbCanvas, object) -> None:
        """Renders object on canvas."""
        # print("Object: ", object)
        if object["class"] == "TrafficLight":
            self.render_light(canvas, object)
            return

        if object["class"] == "Vehicle":
            if object["ego_vehicle"]:
                color = RGB_BY_MASK[BirdViewMasks.AGENT]
            else:
                color = RGB_BY_MASK[BirdViewMasks.VEHICLES]
        elif object["class"] == "Pedestrian":
            color = RGB_BY_MASK[BirdViewMasks.PEDESTRIANS]
        else:
            raise ValueError(f"Unknown object class: {object['class']}")

        self.render_polygon(
            canvas,
            object["extent"][1],
            object["extent"][2],
            object["position"][0],
            -object["position"][1],
            object["yaw"],
            color,
        )

    def render_light(self, canvas: RgbCanvas, object, forced_color=None) -> None:
        """Renders traffic light on canvas."""
        corners = np.asarray(
            [
                [
                    object["position"][0] * self._pixels_per_meter,
                    -object["position"][1] * self._pixels_per_meter,
                    1,
                ]
            ]
        )

        corners = np.dot(self.global_rotation_translation, corners.T).T

        # print("Rotated corners: ", corners)
        # Remove third dimension (z)
        corners = corners[0, :2].astype(np.int32)

        if forced_color is not None:
            color = forced_color
        else:
            if object["state"] == "red":
                color = RGB_BY_MASK[BirdViewMasks.RED_LIGHTS]
            elif object["state"] == "yellow":
                color = RGB_BY_MASK[BirdViewMasks.YELLOW_LIGHTS]
            elif object["state"] == "green":
                color = RGB_BY_MASK[BirdViewMasks.GREEN_LIGHTS]
            else:
                raise ValueError(
                    "Unknown traffic light state encountered: ", object["state"]
                )
        """ print(
            "rendering light of color", color, "at location", corners.astype(np.int32)
        ) """

        # Render as a circle with radius 1.5m
        # print("rendering light of color", color, "at location", location)
        cv2.circle(
            canvas,
            center=(corners[0], corners[1]),
            radius=int(1.5 * self._pixels_per_meter),
            color=color,
            thickness=cv2.FILLED,
        )

    def render_polygon(
        self, canvas: RgbCanvas, width, height, x, y, angle, color
    ) -> None:
        """Renders polygon on canvas."""
        corners = self.get_corners(width, height, x, y, angle)
        corners = corners.astype(np.int32)
        # print(corners)
        cv2.fillPoly(canvas, pts=np.int32([corners]), color=color)

    def get_corners(self, width, height, x, y, angle):
        """Returns corners of polygon.
        Corners are first calculated in local coordinate system (i.e. with respect to center of polygon).
        Then, we create a rotation translation matrix to move the polygon to its correct position.
        """
        corners = np.array(
            [
                [
                    -width * self._pixels_per_meter / 2,
                    -height * self._pixels_per_meter / 2,
                    1,
                ],
                [
                    width * self._pixels_per_meter / 2,
                    -height * self._pixels_per_meter / 2,
                    1,
                ],
                [
                    width * self._pixels_per_meter / 2,
                    height * self._pixels_per_meter / 2,
                    1,
                ],
                [
                    -width * self._pixels_per_meter / 2,
                    height * self._pixels_per_meter / 2,
                    1,
                ],
            ]
        )
        # print("Initial corners: ", corners)
        # Scale the polygon
        # corners *= self._pixels_per_meter

        # print("Scaled corners: ", corners)

        rotation_translation_matrix = np.array(
            [
                [
                    np.cos(angle),
                    -np.sin(angle),
                    x * self._pixels_per_meter,
                ],
                [
                    np.sin(angle),
                    np.cos(angle),
                    y * self._pixels_per_meter,
                ],
                [0, 0, 1],
            ]
        )

        corners = np.dot(rotation_translation_matrix, corners.T).T

        # Apply global rotation and translation
        corners = np.dot(self.global_rotation_translation, corners.T).T

        # print("Rotated corners: ", corners)
        # Remove third dimension (z)
        corners = corners[:, :2]

        # print("Final corners: ", corners)
        return corners

    def as_slots(self, objects, randomize=False, filter_lambda=None):
        """Converts a list of objects to slots."""

        objects = self.filter_objects_in_scene(objects)

        canvas = [
            np.zeros(
                shape=(
                    self.target_size.width,
                    self.target_size.height,
                    1,
                ),
                dtype=np.uint8,
            )
            for _ in range(self._num_slots)
        ]
        # Remove ego vehicle
        objects = [
            object
            for object in objects
            if (
                not ("ego_vehicle" in object and object["ego_vehicle"])
                and (filter_lambda is None or filter_lambda(object))
            )
        ]

        if randomize:
            objects = random.sample(objects, len(objects))

        # Single object per slot, last slot contains all remaining objects
        slot_tuples = [
            (object["id"] % self._num_slots, object)
            for i, object in enumerate(objects)  # min(self._num_slots - 1, i)
        ]

        for slot_idx, object in slot_tuples:
            self.render_slot(object, canvas[slot_idx])
            # print("RENDERED")

        canvas = np.concatenate(canvas, axis=2)

        return canvas

    def render_slot(self, object, slot_canvas):
        """Renders object on slot canvas."""
        if object["class"] in ["Vehicle", "Pedestrian"]:
            self.render_polygon(
                slot_canvas,
                object["extent"][1],
                object["extent"][2],
                object["position"][0],
                -object["position"][1],
                object["yaw"],
                (255,),
            )
        elif object["class"] == "TrafficLight":
            self.render_light(slot_canvas, object, forced_color=(255,))
        else:
            raise ValueError(f"Unknown object class: {object['class']}")

    def point_in_bounds(self, x, y):
        point_local = np.array([x, y, 1])

        # Apply global rotation and translation
        point_global = np.dot(self.global_rotation_translation, point_local)

        x, y = point_global[:2]

        return self._in_bounds(x, y)

    def corners_in_bounds(self, corners):
        return any([self._in_bounds(x, y) for x, y in corners])

    def _in_bounds(self, x, y):
        """Checks if x, y is in bounds of the canvas."""
        return 0 <= x < self.target_size.width and 0 <= y < self.target_size.height

    def filter_objects_in_scene(self, objects, exclude_ego_vehicle=True):
        """Filters objects in scene to only include those that are relevant for the bird's eye view."""
        filtered_objects = []
        for object in objects:
            if object["class"] in ["Vehicle", "Pedestrian"]:
                if (
                    exclude_ego_vehicle
                    and "ego_vehicle" in object
                    and object["ego_vehicle"]
                ):
                    continue
                corners = self.get_corners(
                    object["extent"][1],
                    object["extent"][2],
                    object["position"][0],
                    -object["position"][1],
                    object["yaw"],
                )
                if self.corners_in_bounds(corners):
                    filtered_objects.append(object)

            elif object["class"] == "TrafficLight":
                if self.point_in_bounds(
                    object["position"][0] * self._pixels_per_meter,
                    -object["position"][1] * self._pixels_per_meter,
                ):
                    filtered_objects.append(object)
        return filtered_objects
