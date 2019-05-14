#!/usr/bin/env python3

import json
from collections import deque

import numpy as np

from primitives.enums import Color


def main():
    with open("/Users/miro/Desktop/annotations/00004/annotations.json") as file:
        ann = json.load(file)

    cars = deque(ann["cars"])
    red = deque(ann["lights"]["red"])

    red_orange = deque(ann["lights"]["red_orange"])
    green = deque(ann["lights"]["green"])
    orange = deque(ann["lights"]["orange"])

    light_changes = deque(sorted(ann["lights"]["red_orange"] + ann["lights"]["red"] + ann["lights"]["green"] + ann["lights"]["orange"]))
    last_seq = 0

    current_status = None
    if green[0] == 0:
        current_status = Color.GREEN
    if orange[0] == 0:
        current_status = Color.ORANGE
    if red[0] == 0:
        current_status = Color.RED
    if red_orange[0] == 0:
        current_status = Color.RED_ORANGE

    red_drivers = []
    orange_drivers = []
    light_changes.popleft()

    try:
        while True:
            if cars[0] > light_changes[0]:
                last_seq = light_changes.popleft()
                current_status = Color((current_status + 1) % 4)
                continue

            elif current_status == Color.ORANGE:
                if np.abs(cars[0] - light_changes[0]) < 5:
                    red_drivers.append(cars[0])

                orange_drivers.append(cars[0])

            elif current_status == Color.RED:
                if np.abs(cars[0] - last_seq) < 5:
                    orange_drivers.append(cars[0])

                red_drivers.append(cars[0])

            elif current_status == Color.RED_ORANGE:
                if np.abs(cars[0] - last_seq) < 5:
                    orange_drivers.append(cars[0])

                red_drivers.append(cars[0])

            cars.popleft()

    except IndexError:
        pass

    print("red:", len(red_drivers))
    print("orange:", len(orange_drivers))

    ann["violations"] = {
        "red_drivers": red_drivers,
        "orange_drivers": orange_drivers
    }

    with open("/Users/miro/Desktop/annotations/00004/annotations.json", "w") as file:
        json.dump(ann, file)


if __name__ == '__main__':
    main()