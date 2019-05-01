import getopt
import json
import sys

import cv2

from params import COLOR_RED, COLOR_ORANGE, COLOR_GREEN
from pipeline.observer import CarBehaviourMode
from pipeline.parser import ParametersError

video_input = None

try:
    opts, args = getopt.getopt(sys.argv[1:], "", ["input="])

except getopt.GetoptError:
    raise ParametersError

for opt, arg in opts:

    if opt in "--input":
        video_input = arg

    else:
        print("help")


if video_input is None:
    exit(1)

annotation_input = video_input[:-3] + "txt"

tape = cv2.VideoCapture(video_input)

with open(annotation_input) as file:
    annotation = json.load(file)

frame_counter = 0
while True:
    successful, frame = tape.read()
    if not successful:
        break

    top_left = annotation["top_left"][frame_counter]
    bottom_right = annotation["bottom_right"][frame_counter]
    behaviour = annotation["behaviour"][frame_counter]

    if None not in [top_left, bottom_right, behaviour]:
        if CarBehaviourMode(behaviour) == CarBehaviourMode.NORMAL:
            color = COLOR_GREEN
        elif CarBehaviourMode(behaviour) == CarBehaviourMode.RED_DRIVER:
            color = COLOR_RED
        else:
            color = COLOR_ORANGE

        cv2.rectangle(img=frame, pt1=tuple(top_left), pt2=tuple(bottom_right), color=color, thickness=3)

    cv2.imshow(annotation_input, frame)
    cv2.waitKey(60)
    frame_counter += 1