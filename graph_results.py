import fd_wrapper as wrapper
import numpy as np
import os
from tqdm import tqdm
from fnmatch import filter
from csv import reader
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb

results_dir = wrapper.relative_path("./results/200_old/hue", root=__file__)

paths = []
for root, dirs, files in os.walk(results_dir):
    for path in filter(files, "*.csv"):
        paths.append(os.path.join(root, path))

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
            entry["results"].append(line)
    entry["results"] = np.array(entry["results"])
    entries[i] = entry

detectors = ["BlazeFace", "MTCNN", "RetinaFace", "SSD"]

results = [None] * len(detectors)

image_count = len(entries[0]["results"])

fig, ax = plt.subplots(len(detectors) + 1, sharex=True)
fig.suptitle("Accuracy when classifying " + str(image_count) + " known images")
fig.tight_layout()

graphing = "hue"
graph_xlabel = "Hue (0-255)"

line_color = "blue" if graphing is None else "gray"

for i in range(len(detectors)):
    results[i] = np.array([100 * np.sum(entry["results"][:, i + 1].astype(int) == entry["results"][:, 5].astype(int)) / image_count for entry in entries])

y_min = np.min(results)

for i in range(len(detectors)):
    ax[i].set_ylim(y_min, 100)
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
    
ax[len(detectors)].set_ylim(y_min, 100)
ax[len(detectors)].set_title("All detectors")
ax[len(detectors)].set_ylabel("Accuracy (percent)")
ax[len(detectors)].set_xlabel(graph_xlabel)
ax[len(detectors)].legend(detectors, loc="upper right", prop={"size": 8})
plt.show()