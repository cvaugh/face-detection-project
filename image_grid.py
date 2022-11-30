import numpy as np
import matplotlib.pyplot as plt
import fd_wrapper as wrapper
from csv import reader
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm

results_csv = wrapper.relative_path("./results/results_negative.csv", root=__file__)

results = []
with open(results_csv) as f:
    next(f) # skip header
    r = reader(f, delimiter=",")
    for line in r:
        results.append(line)

results = np.array(results)
results = results[:, 0:np.shape(results)[1] - 1]

counts = [[], [], [], [], []]
for row in tqdm(results):
    counts[np.sum(row[1:5], dtype=int)].append(row[0])

for i in range(len(counts)):
    counts[i] = counts[i][:200]

print([len(i) for i in counts])

fig, ax = plt.subplots(1, 5, figsize=(15, 5))

for i in tqdm(range(len(counts))):
    l = int(np.sqrt(len(counts[i])))
    xx, yy = np.meshgrid(np.linspace(0, 1, l), np.linspace(0, 1, l))
    points = np.vstack([xx.ravel(), yy.ravel()])
    for x, y, path in zip(points[0], points[1], counts[i]):
        ab = AnnotationBbox(OffsetImage(plt.imread(path), zoom=0.05), (x, y), frameon=False)
        ax[i].add_artist(ab)
    ax[i].set_yticklabels([])
    ax[i].set_xticklabels([])
    ax[i].set_xlim((-0.1, 1.1))
    ax[i].set_ylim((-0.1, 1.1))
    ax[i].set_xlabel(str(len(counts[i])) + " entries")
    ax[i].set_title(str(i) + " of 4 classify as face")

plt.savefig("image_grid.jpg", dpi=500)
#plt.show()
