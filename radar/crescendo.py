"""
Constants for the 2024 Crescendo field.
Always blue origin.
"""

import cv2
import numpy as np
import utils
KEYPOINT_MAP = [
    "red_far_corner",
    "center_far",
    "blue_far_corner",
    "red_far_sub_out",
    "red_close_sub_out",
    "red_far_sub_wall",
    "red_close_sub_wall",
    "center_close",
    "blue_far_sub_wall",
    "blue_far_sub_out",
    "blue_close_sub_out",
    "blue_close_sub_wall",
    "blue_source_far",
    "blue_source_close",
    "red_amp_red",
    "red_amp_blue",
    "blue_amp_red",
    "blue_amp_blue",
    "red_source_close",
    "red_source_far",
]

FIELD_HEIGHT = 8.224
FIELD_WIDTH = 16.541052

SUB_OUT = 0.902
SUB_OUT_FAR_TO_WALL = 2.142757
SUB_IN_FAR_TO_WALL = 1.622058
FROM_SUB_CLOSE_TO_FAR = 1.041400
SUB_OUT_CLOSE_TO_WALL = SUB_OUT_FAR_TO_WALL + FROM_SUB_CLOSE_TO_FAR
SUB_IN_CLOSE_TO_WALL = SUB_OUT_CLOSE_TO_WALL + (
    SUB_OUT_FAR_TO_WALL - SUB_IN_FAR_TO_WALL
)

SOURCE_OUT_CLOSE = 0.57
SOURCE_OUT_CLOSE_TO_FAR = 1.09
SOURCE_OUT_FAR = SOURCE_OUT_CLOSE + SOURCE_OUT_CLOSE_TO_FAR
SOURCE_TO_DS = 1.831628

AMP_TO_DS = 1.259172
AMP_NEAR_WALL_TO_FAR = 1.168400
AMP_FAR_TO_DS = AMP_TO_DS + AMP_NEAR_WALL_TO_FAR

# fmt: off
VERTICES = [
    (FIELD_WIDTH, FIELD_HEIGHT), # Red Far Corner 
    (FIELD_WIDTH/2, FIELD_HEIGHT), # Center Far
    (0, FIELD_HEIGHT), # Blue Far Corner
    (FIELD_WIDTH-SUB_OUT, FIELD_HEIGHT-SUB_OUT_FAR_TO_WALL), # Red Far Subwoofer Out
    (FIELD_WIDTH-SUB_OUT, FIELD_HEIGHT-SUB_OUT_CLOSE_TO_WALL), # Red Close Subwoofer Out
    (FIELD_WIDTH, FIELD_HEIGHT-SUB_IN_FAR_TO_WALL), # Red Far Subwoofer Wall
    (FIELD_WIDTH, FIELD_HEIGHT-SUB_IN_CLOSE_TO_WALL), # Red Close Subwoofer Wall
    (FIELD_WIDTH/2, 0), # Center Close
    (0, FIELD_HEIGHT-SUB_IN_FAR_TO_WALL), # Blue Far Subwoofer Wall
    (SUB_OUT, FIELD_HEIGHT-SUB_OUT_FAR_TO_WALL), # Blue Far Subwoofer Out
    (SUB_OUT, FIELD_HEIGHT-SUB_OUT_CLOSE_TO_WALL), # Blue Close Subwoofer Out
    (0, FIELD_HEIGHT-SUB_IN_CLOSE_TO_WALL), # Blue Close Subwoofer Wall
    (FIELD_WIDTH, SOURCE_OUT_FAR), # Blue Source Far
    (FIELD_WIDTH-SOURCE_TO_DS, SOURCE_OUT_CLOSE), # Blue Source Close
    (FIELD_WIDTH-AMP_TO_DS, FIELD_HEIGHT), # Red Amp Red
    (FIELD_WIDTH-AMP_FAR_TO_DS, FIELD_HEIGHT), # Red Amp Blue
    (AMP_FAR_TO_DS, FIELD_HEIGHT), # Blue Amp Red
    (AMP_TO_DS, FIELD_HEIGHT), # Blue Amp Blue
    (SOURCE_TO_DS, SOURCE_OUT_CLOSE), # Red Source Close
    (0, SOURCE_OUT_FAR), # Red Source Far
]
# fmt: on

# VERTICES = utils.pad_vertices(VERTICES)

EDGES = [
    (0, 12),  # Red Far Corner to Blue Source Far
    (12, 13),  # Blue Source Far to Blue Source Close
    (13, 7),  # Blue Source Close to Center Close
    (7, 18),  # Center Close to Red Source Close
    (18, 19),  # Red Source Close to Red Source Far
    (19, 2),  # Red Source Far to Red Far Corner
    (2, 1),  # Red Far Corner to Center Far
    (1, 0),  # Center Far to Blue Far Corner
]


def draw_2d_field_ex(scale=200):
    image = np.zeros((int(FIELD_HEIGHT * scale), int(FIELD_WIDTH * scale), 3), np.uint8)

    for i, (x, y) in enumerate(VERTICES):
        cv2.circle(image, (int(x * scale), int(y * scale)), 5, (0, 255, 0), -1)
        cv2.putText(
            image,
            KEYPOINT_MAP[i],
            (int(x * scale), int(y * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

    return image
if __name__ == "__main__":
    cv2.imwrite("field.png", draw_2d_field_ex())
