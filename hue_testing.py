import fd_wrapper as wrapper
import numpy as np
import os
from tqdm import tqdm
from fnmatch import filter
from csv import reader
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb

results_dir = wrapper.relative_path("./results_100", root=__file__)

paths = []
for root, dirs, files in os.walk(results_dir):
    for path in filter(files, "*.csv"):
        paths.append(os.path.join(root, path))

shifts = [int(os.path.splitext(os.path.basename(path))[0]) for path in paths]

paths_sorted = np.array(paths)[np.argsort(shifts)[::1]]

shifts = np.sort(shifts)

entries = [None] * len(paths_sorted)

for i in range(len(entries)):
    entry = {"shift": shifts[i], "results": []}
    with open(paths_sorted[i]) as f:
        next(f)
        r = reader(f, delimiter=",")
        for line in r:
            entry["results"].append(line)
    entry["results"] = np.array(entry["results"])
    entries[i] = entry

detectors = ["BlazeFace", "MTCNN", "RetinaFace", "SSD"]

results = [None] * len(detectors)

image_count = len(entries[0]["results"])

fig, ax = plt.subplots(len(detectors), sharex=True)
fig.suptitle("Accuracy when classifying " + str(image_count) + " known images with varied hues")
fig.tight_layout()

for i in range(len(detectors)):
    results[i] = np.array([100 * np.sum(entry["results"][:, i + 1].astype(int) == entry["results"][:, 5].astype(int)) / image_count for entry in entries])
    ax[i].set_title(detectors[i])
    ax[i].set_ylabel("Accuracy (percent)")
    ax[i].plot(shifts, results[i])
    for hue in range(0, np.max(shifts)):
        ax[i].plot(hue, np.min(results[i]), color=hsv_to_rgb(hue / 255, 1, 1), marker="^", markersize=3)
ax[len(detectors) - 1].set_xlabel("Hue (HSV, 0-255)")
plt.show()
