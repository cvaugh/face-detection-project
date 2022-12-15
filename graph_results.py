import os
from colorsys import hsv_to_rgb
from csv import reader
from fnmatch import filter

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import fd_wrapper as wrapper

results_dir = wrapper.relative_path("./results/200_old/hue", root=__file__)
out_file = "hue.png"
subtitle = "Hue (0-255)"
graphing = "hue" # must be one of "hue", "saturation", "value", or None
positive_only = True
negative_only = False

if positive_only and negative_only:
    print("Only one of 'positive_only' and 'negative_only' may be true")
    exit()

paths = []
for root, dirs, files in os.walk(results_dir):
    for path in filter(files, "*.csv"):
        paths.append(os.path.join(root, path))

if len(paths) == 0:
    print("No CSV files found")
    exit()

shifts = [int(os.path.splitext(os.path.basename(path))[0]) for path in paths]

paths_sorted = np.array(paths)[np.argsort(shifts)[::1]]

shifts = np.sort(shifts)

entries = [None] * len(paths_sorted)

for i in tqdm(range(len(entries))):
    entry = {"shift": shifts[i], "results": []}
    with open(paths_sorted[i]) as f:
        next(f) # skip header
        r = reader(f, delimiter=",")
        for line in r:
            e = np.array(line[1:len(line)], dtype=int)
            if positive_only and e[len(e) - 1] == 0:
                continue
            if negative_only and e[len(e) - 1] == 1:
                continue
            entry["results"].append(e)
    entry["results"] = np.array(entry["results"])
    entries[i] = entry

detectors = ["BlazeFace", "MTCNN", "RetinaFace", "SSD"]

results = [None] * len(detectors)

image_count = len(entries[0]["results"])

fig, ax = plt.subplots(len(detectors) + 1, sharex=True)
fig.suptitle(f"Accuracy when classifying {image_count} known{'-positive' if positive_only else '-negative' if negative_only else ''} images")
fig.tight_layout()

graph_xlabel = subtitle

line_color = "blue" if graphing is None else "gray"

y_min = float('inf')
y_max = float('-inf')

for i in range(len(detectors)):
    results[i] = np.array([100 * np.sum(entry["results"][:, i] == entry["results"][:, len(detectors)]) / image_count for entry in entries])
    if np.any(results[i] != 0.0):
        y_min = np.min([y_min, np.min(results[i])])
        y_max = np.max([y_max, np.max(results[i])])

if y_min == y_max:
    y_min -= 0.01
    y_max += 0.01
y_range = y_max - y_min
y_padding = y_range / 10

for i in tqdm(range(len(detectors))):
    ax[i].set_ylim(y_min - y_padding, y_max + y_padding)
    ax[i].set_title(detectors[i])
    ax[i].set_ylabel("Accuracy (percent)")
    ax[i].plot(shifts, results[i], color=line_color, linewidth=1)
    if graphing is not None:
        for j in range(0, np.max(shifts) + 1):
            match graphing:
                case "hue":
                    color = hsv_to_rgb(j / 255, 1, 1)
                case "saturation":
                    color = hsv_to_rgb(0, j / 255, 0.9)
                case "value":
                    color = hsv_to_rgb(0, 1, j / 255)
                case _:
                    color = "red"
            ax[i].plot(j, results[i][j], color=color, marker="o", markersize=3)
    ax[len(detectors)].plot(shifts, results[i], linewidth=1)
    
ax[len(detectors)].set_ylim(y_min - y_padding, y_max + y_padding)
ax[len(detectors)].set_title("All detectors")
ax[len(detectors)].set_ylabel("Accuracy (percent)")
ax[len(detectors)].set_xlabel(graph_xlabel)
ax[len(detectors)].legend(detectors, loc="upper right", prop={"size": 8})

if out_file is None:
    plt.show()
else:
    plt.gcf().set_size_inches(1920/96, 1200/96)
    plt.savefig(out_file)
