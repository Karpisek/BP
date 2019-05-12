import json
from collections import deque

import numpy as np

from repositories.traffic_light_repository import Color

with open("/Users/miro/Desktop/annotations/00004/annotations.json") as file:
    ground_truth = json.load(file)

with open("/Users/miro/Desktop/slow_calibrator/00004/automatic_statistics.json") as file:
    detected = json.load(file)

# car detection
cars_detected = deque(detected["all_drivers"].values())
cars_ground_truth = deque(ground_truth["cars"])

good_detections = []
false_detected = []
not_detected_cars = []

accepted_gt = []

cars_count = len(cars_detected)
ground_truth_cars_count = len(cars_ground_truth)

while True:

    try:
        next_car = cars_detected[0]
        next_ground_truth = cars_ground_truth[0]
    except IndexError:
        break

    if np.abs(next_car - next_ground_truth) < 20:
        good_detections.append(next_car)
        accepted_gt.append(next_ground_truth)
        cars_detected.popleft()
        cars_ground_truth.popleft()

    elif next_car < next_ground_truth:
        false_detected.append(next_car)
        cars_detected.popleft()

    else:
        not_detected_cars.append(next_ground_truth)
        cars_ground_truth.popleft()

print("\n===== CAR DETECTION EVALUATION =====")

print(f"Ground truth cars: {ground_truth_cars_count}")
print(f"Detected cars: {cars_count}")
print(f"False detected cars: {len(false_detected) + len(cars_detected)}")
print(f"Not detected cars: {len(not_detected_cars) + len(cars_ground_truth)}")
print(f"False detected cars percentage: {round((len(false_detected) / cars_count) * 100, 2)}% (from detected cars)")
print(f"Detected cars percentage: {round(((cars_count - len(false_detected) - len(cars_detected)) / ground_truth_cars_count) * 100,2)}% (from the real number of cars)")

# violation detection

red_drivers = deque(detected["red_drivers"].values())
orange_drivers = deque(detected["orange_drivers"].values())

gt_red_drivers = ground_truth["violations"]["red_drivers"]
gt_orange_drivers = ground_truth["violations"]["orange_drivers"]

ground_truth_violation_real_count = len(set(gt_red_drivers + gt_orange_drivers))

for car in gt_red_drivers[:]:
    if car not in accepted_gt:
        gt_red_drivers.remove(car)

for car in gt_orange_drivers[:]:
    if car not in accepted_gt:
        gt_orange_drivers.remove(car)

red_drivers_ground_truth = deque(ground_truth["violations"]["red_drivers"])
orange_drivers_ground_truth = deque(ground_truth["violations"]["orange_drivers"])

red_drivers_count = len(red_drivers)
orange_drivers_count = len(orange_drivers)
gt_red_drivers_count = len(gt_red_drivers)
gt_orange_drivers_count = len(gt_orange_drivers)
violation_count = len(red_drivers + orange_drivers)
ground_truth_violation_count = len(set(gt_red_drivers + gt_orange_drivers))

good_detections = []
false_detected = []
not_detected = []
not_detected_from_detected = []

print("\n===== VIOLATION DETECTION EVALUATION =====")
print(f"Ground truth violations: {ground_truth_violation_real_count}")
print(f"Detected violations: {violation_count}")
print(f"Detected violations percentage: {round((violation_count / ground_truth_violation_real_count) * 100,2)}% (From total number of cars)")

# light state evaluation

red_light = deque(detected["light_states"]["red"])
orange_light = deque(detected["light_states"]["orange"])
green_light = deque(detected["light_states"]["green"])
red_orange_light = deque(detected["light_states"]["red_orange"])

gt_red_light = deque(ground_truth["lights"]["red"])
gt_orange_light = deque(ground_truth["lights"]["orange"])
gt_green_light = deque(ground_truth["lights"]["green"])
gt_red_orange_light = deque(ground_truth["lights"]["red_orange"])

last_light_change = sorted([gt_red_light[-1], gt_orange_light[-1], gt_green_light[-1], gt_red_orange_light[-1]])[-1]

current_status = None
if red_light[0] == 0:
    current_status = Color.RED
    red_light.popleft()

if orange_light[0] == 0:
    current_status = Color.ORANGE
    orange_light.popleft()

if red_orange_light[0] == 0:
    current_status = Color.RED_ORANGE
    red_light.popleft()

if green_light[0] == 0:
    current_status = Color.GREEN
    green_light.popleft()

gt_current_status = None
if gt_red_light[0] == 0:
    gt_current_status = Color.RED
    gt_red_light.popleft()

if gt_orange_light[0] == 0:
    gt_current_status = Color.ORANGE
    gt_orange_light.popleft()

if gt_red_orange_light[0] == 0:
    gt_current_status = Color.RED_ORANGE
    gt_red_light.popleft()

if gt_green_light[0] == 0:
    gt_current_status = Color.GREEN
    gt_green_light.popleft()

wrong_time = 0

list_of_timings = deque(sorted(set(detected["light_states"]["red"] +
                                   detected["light_states"]["orange"] +
                                   detected["light_states"]["green"] +
                                   detected["light_states"]["red_orange"] +
                                   ground_truth["lights"]["red"] +
                                   ground_truth["lights"]["orange"] +
                                   ground_truth["lights"]["green"] +
                                   ground_truth["lights"]["red_orange"])))

last_seq = 0
seq = list_of_timings.popleft()

while True:

    try:
        seq = list_of_timings.popleft()
    except IndexError:
        break

    if current_status != gt_current_status:
        wrong_time += seq - last_seq

    try:
        if green_light[0] == seq:
            green_light.popleft()
            current_status = Color.GREEN
    except IndexError:
        pass

    try:
        if orange_light[0] == seq:
            orange_light.popleft()
            current_status = Color.ORANGE
    except IndexError:
        pass

    try:
        if red_light[0] == seq:
            red_light.popleft()
            current_status = Color.RED
    except IndexError:
        pass

    try:
        if red_orange_light[0] == seq:
            red_orange_light.popleft()
            current_status = Color.RED_ORANGE
    except IndexError:
        pass


    # gt
    try:
        if gt_green_light[0] == seq:
            gt_green_light.popleft()
            gt_current_status = Color.GREEN
    except IndexError:
        pass

    try:
        if gt_orange_light[0] == seq:
            gt_orange_light.popleft()
            gt_current_status = Color.ORANGE
    except IndexError:
        pass

    try:
        if gt_red_light[0] == seq:
            gt_red_light.popleft()
            gt_current_status = Color.RED
    except IndexError:
        pass

    try:
        if gt_red_orange_light[0] == seq:
            gt_red_orange_light.popleft()
            gt_current_status = Color.RED_ORANGE
    except IndexError:
        pass

    last_seq = seq

print("\n===== TRAFFIC LIGHT STATE EVALUATION =====")
print(f"Total number of frames: {last_light_change}")
print(f"Frames with bad classified light state: {wrong_time}")
print(f"Light state detection precision: {round((1 - (wrong_time / last_light_change)) * 100,2)}%")
