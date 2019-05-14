#!/usr/bin/env python3

import getopt
import json
import sys
import cv2

from primitives.constants import COLOR_RED, COLOR_ORANGE, COLOR_GREEN
from pipeline.observer import CarBehaviourMode
from primitives.parser import ParametersError


def main():

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

    annotation_input = video_input[:-3] + "json"

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
        key = cv2.waitKey(0)

        if key & 0xFF == ord("q"):
            exit()
        if key & 0xFF == ord("p"):
            cv2.imwrite("/Users/miro/Desktop/bp_photos/red_driver" + str(frame_counter) + ".png", frame)

        frame_counter += 1


if __name__ == '__main__':
    main()
