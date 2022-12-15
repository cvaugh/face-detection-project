from math import floor

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import fd_wrapper as wrapper
from fd_wrapper import transform

dataset_path = wrapper.relative_path("./known/200/positive", root=__file__)

paths = wrapper.read_dataset(dataset_path)

image = wrapper.load_image(np.random.choice(paths))

psize = 8

fig, ax = plt.subplots(psize, psize)
fig.suptitle("Increasing factor from left to right, top to bottom")

for i in tqdm(range(0, 256, 2)):
    j = int(i / 4)
    ax[floor(j / psize)][j % psize].axis("off")
    ax[floor(j / psize)][j % psize].imshow(np.asarray(transform.low_pass(image, i)))

plt.show()
