#!/usr/bin/env python3
"""
Generate synthetic CAD dataset with flat action language (no JSON).
Each action is a single line with key=value pairs separated by spaces.
Actions are joined by ' | ' and end with ' <EOS>'.
"""

import os
import csv
import random
import math
import re          # <-- ADDED THIS IMPORT
import argparse
from pathlib import Path

random.seed(42)

# ----------------------------------------------------------------------
# Geometric helpers
# ----------------------------------------------------------------------
def random_unit_vector():
    theta = random.uniform(0, 2 * math.pi)
    phi = math.acos(random.uniform(-1, 1))
    return f"{math.sin(phi)*math.cos(theta):.2f},{math.sin(phi)*math.sin(theta):.2f},{math.cos(phi):.2f}"

def random_plane_origin():
    return f"{random.uniform(-50,50):.2f},{random.uniform(-50,50):.2f},{random.uniform(-50,50):.2f}"

def random_sketch_profile():
    choice = random.choices(['rect', 'circle', 'poly', 'slot'], weights=[0.4,0.3,0.2,0.1])[0]
    if choice == 'rect':
        w = random.uniform(10,100)
        h = random.uniform(10,100)
        desc = f"rectangle width {w:.2f} height {h:.2f}"
        params = f"profile=rect width={w:.2f} height={h:.2f}"
    elif choice == 'circle':
        r = random.uniform(5,50)
        desc = f"circle radius {r:.2f}"
        params = f"profile=circle radius={r:.2f}"
    elif choice == 'poly':
        n = random.randint(3,6)
        R = random.uniform(10,60)
        desc = f"regular polygon with {n} sides"
        points = []
        for i in range(n):
            angle = 2*math.pi*i/n
            x = R * math.cos(angle)
            y = R * math.sin(angle)
            points.append(f"{x:.2f},{y:.2f}")
        params = f"profile=poly points=" + " ".join(points)
    else:  # slot
        w = random.uniform(20,80)
        h = random.uniform(10,50)
        desc = f"slot width {w:.2f} height {h:.2f}"
        params = f"profile=slot width={w:.2f} height={h:.2f}"
    return params, desc

def random_point_in_rect(w, h):
    return f"{random.uniform(-w/2, w/2):.2f},{random.uniform(-h/2, h/2):.2f}"

def random_point_in_circle(r):
    angle = random.uniform(0,2*math.pi)
    rad = random.uniform(0, r*0.8)
    return f"{rad*math.cos(angle):.2f},{rad*math.sin(angle):.2f}"

def random_hole_center(profile_params):
    # profile_params is a string like "profile=rect width=32.0 height=19.0"
    if 'rect' in profile_params:
        w_match = re.search(r'width=([0-9.]+)', profile_params)
        h_match = re.search(r'height=([0-9.]+)', profile_params)
        if w_match and h_match:
            w = float(w_match.group(1))
            h = float(h_match.group(1))
            return random_point_in_rect(w, h)
    elif 'circle' in profile_params:
        r_match = re.search(r'radius=([0-9.]+)', profile_params)
        if r_match:
            r = float(r_match.group(1))
            return random_point_in_circle(r)
    # fallback
    return f"{random.uniform(-30,30):.2f},{random.uniform(-30,30):.2f}"

# ----------------------------------------------------------------------
# Action generation
# ----------------------------------------------------------------------
def generate_random_part():
    actions = []
    features_desc = []

    # 1. Sketch
    plane_normal = random_unit_vector()
    plane_origin = random_plane_origin()
    profile_params, profile_desc = random_sketch_profile()
    actions.append(f"CREATE_SKETCH plane_normal={plane_normal} plane_origin={plane_origin} {profile_params}")
    features_desc.append(f"Sketch: {profile_desc} on plane normal {plane_normal}")

    # 2. Extrude (along normal)
    distance = random.uniform(5,100)
    operation = random.choices(["join","cut"], weights=[0.8,0.2])[0]
    actions.append(f"EXTRUDE distance={distance:.2f} direction={plane_normal} operation={operation}")
    features_desc.append(f"extrude {distance:.2f} ({operation})")

    # 3. Holes (only if join)
    num_holes = 0
    if operation == "join":
        num_holes = random.choices([0,1,2,3], weights=[0.3,0.4,0.2,0.1])[0]
        for _ in range(num_holes):
            center = random_hole_center(profile_params)
            radius = random.uniform(2,10)
            depth = random.uniform(distance*0.5, distance*0.95)
            actions.append(f"ADD_HOLE center={center} radius={radius:.2f} depth={depth:.2f} axis={plane_normal}")
        if num_holes:
            features_desc.append(f"{num_holes} holes")

    # 4. Fillet (occasionally)
    if random.random() < 0.3 and operation == "join":
        rad = random.uniform(0.5,5)
        actions.append(f"FILLET radius={rad:.2f}")
        features_desc.append(f"fillet radius {rad:.2f}")

    # 5. Circular pattern (if >=3 holes)
    if num_holes >= 3 and random.random() < 0.4:
        actions.append(f"CIRCULAR_PATTERN center={plane_origin} axis={plane_normal} count={num_holes} radius={random.uniform(20,60):.2f}")
        features_desc.append("circular pattern")

    return actions, features_desc

def generate_text_description(features_list):
    templates = [
        "A mechanical part with {}.", 
        "Construct a CAD model from: {}.",
        "Create a part that has {}.",
        "The component consists of {}."
    ]
    return random.choice(templates).format("; ".join(features_list))

def linearize_actions(actions):
    return " | ".join(actions) + " <EOS>"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--output_dir", type=str, default="flat_dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    actions_dir = os.path.join(args.output_dir, "actions_flat")
    os.makedirs(actions_dir, exist_ok=True)

    rows = []
    for i in range(args.num_samples):
        actions, features_desc = generate_random_part()
        action_str = linearize_actions(actions)
        action_file = os.path.join(actions_dir, f"sample_{i:06d}_actions.txt")
        with open(action_file, "w") as f:
            f.write(action_str)
        text = generate_text_description(features_desc)
        rows.append([text, action_file, action_str])
        if (i+1) % 10000 == 0:
            print(f"Generated {i+1} samples")

    csv_path = os.path.join(args.output_dir, "text_action_pairs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "action_file", "action_str"])
        writer.writerows(rows)

    print(f"Generated {args.num_samples} samples in {args.output_dir}")
    print(f"Example action string:\n{rows[0][2]}")

if __name__ == "__main__":
    main()
