import csv
from datetime import timedelta
from os.path import dirname
from pathlib import Path
from time import time

import numpy as np
from tqdm import tqdm

import fd_wrapper as wrapper
from fd_wrapper import transform
from fd_wrapper.detectors import *

if __name__ != "__main__":
    exit()

#dataset_path = wrapper.relative_path("./filtered/flickr_1.2", root=__file__)
dataset_path = wrapper.relative_path("./known/200/positive", root=__file__)
#dataset_path = wrapper.relative_path("./experiments/b", root=__file__)
#dataset_path = wrapper.relative_path("../datasets/celeba/img_align_celeba", root=__file__)

paths = wrapper.read_dataset(dataset_path)
#paths = wrapper.get_ground_truth(wrapper.relative_path("./filtered/labels_00-03.tsv", root=__file__),
#    split_path_at="flickr_1.2", relative_to=wrapper.relative_path("./filtered/flickr_1.2", root=__file__))[0][:5000]

truth_override = None
detectors = [blazeface, insightface, mtcnn, ssd]

paths_start = 0
paths_end = min(-1, len(paths))
if paths_end < 0: paths_end = len(paths)
assert paths_start < paths_end, "paths_start must be < paths_end"

print(f"Truncating paths from {len(paths)} to {paths_end - paths_start} images (paths {paths_start} to {paths_end})")
paths = paths[paths_start:paths_end]

batch_size = 100

# Small subset of images for testing
batches = wrapper.create_batches(paths, batch_size)
batch_count = len(batches)
print(f"Found {len(paths)} images ({batch_count} {'batch' if batch_count == 1 else 'batches'} of size {batch_size})")

def run_batches(start_index=0, set_count=255):
    start_time = time()
    durations = []
    for i in range(max(0, start_index), set_count):
        start_time = time()
        print(f"\n(Set {i}/{set_count}) ", end="")
        wrapper.write_results(paths, wrapper.classify_batches(batches, detectors, transform.hue_rotation, i),
                              detectors, f"results_temp/{str(i)}.csv", True, truth_override)
        duration = time() - start_time
        durations.append(duration)
        remaining = set_count - i - 1
        print(f"{remaining} set(s) remaining ({timedelta(seconds=np.sum(durations))}",
              f" elapsed, ~{timedelta(seconds=np.mean(durations[-5:]) * remaining)} remaining)")
    print(f"\n\nCompleted in {timedelta(seconds=time() - start_time)}")

run_batches()
