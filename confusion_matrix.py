import csv

import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import fd_wrapper as wrapper

positive_file = wrapper.relative_path("./results/known_positive.csv", root=__file__)
negative_file = wrapper.relative_path("./results/known_negative.csv", root=__file__)

detectors = ["BlazeFace", "MTCNN", "RetinaFace", "SSD"]

confusion_matrices = [np.zeros((2, 2), dtype=int) for i in detectors]

count = 0

def read_results(file):
    global count
    with open(file) as f:
        next(f) # skip header
        r = csv.reader(f, delimiter=",")
        for row in r:
            ground_truth = int(row[len(row) - 1])
            count += 1
            for i in range(1, len(row) - 1):
                detector = i - 1
                predicted = int(row[i])
                if ground_truth == 0 and predicted == 0:
                    confusion_matrices[detector][1, 1] += 1
                elif ground_truth == 1 and predicted == 0:
                    confusion_matrices[detector][1, 0] += 1
                elif ground_truth == 0 and predicted == 1:
                    confusion_matrices[detector][0, 1] += 1
                elif ground_truth == 1 and predicted == 1:
                    confusion_matrices[detector][0, 0] += 1
                    

read_results(positive_file)
read_results(negative_file)

fig, ax = plt.subplots(1, len(detectors))
fig.suptitle("Results of classifying " + str(count) + " known images")
fig.tight_layout()
text_border = [matplotlib.patheffects.withStroke(linewidth=2, foreground="black")]
cmap = ListedColormap(["#55FF55", "#FF5555"])

for i in range(len(detectors)):
    ax[i].imshow(np.identity(2), cmap=cmap, origin="lower")
    ax[i].set_xlabel("Classification")
    ax[i].set_ylabel("Ground Truth"); 
    ax[i].set_title(detectors[i]); 
    ax[i].set_xticks([0, 1])
    ax[i].xaxis.set_ticklabels(["PN", "PP"])
    ax[i].set_yticks([0, 1])
    ax[i].yaxis.set_ticklabels(["N", "P"])
    for y, x in np.ndindex(np.shape(confusion_matrices[i])):
        ax[i].text(x, y, str(confusion_matrices[i][x, y]), ha="center", va="center", color="white",
            path_effects=text_border)
plt.show()
