import fd_wrapper as wrapper
from fd_wrapper import transform
from fd_wrapper.detectors import *
from time import time
from datetime import timedelta
import numpy as np
from pathlib import Path
from os.path import dirname
from tqdm import tqdm
import csv

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
detectors = [blazeface, insightface, mtcnn, retinaface, ssd]

paths_start = 0
paths_end = min(-1, len(paths))
if paths_end < 0: paths_end = len(paths)
assert paths_start < paths_end, "paths_start must be < paths_end"

print("Truncating paths from", len(paths), "to", paths_end - paths_start, "images (paths", paths_start, "to", str(paths_end) + ")")
paths = paths[paths_start:paths_end]

batch_size = 100

# Small subset of images for testing
batches = wrapper.create_batches(paths, batch_size)
batch_count = len(batches)
print("Found", len(paths), "images (" + str(batch_count), "batch" if batch_count == 1 else "batches", "of size", str(batch_size) + ")")

def run_batches():
    sets_start_time = time()
    set_count = 255
    set_durations = []
    for i in range(set_count):
        set_start_time = time()
        print("\n(Set " + str(i) + "/" + str(set_count) + ") ", end="")
        wrapper.write_results(paths, wrapper.classify_batches(batches, detectors, transform=transform.hue_rotation, transform_offset=i), detectors, f"results_temp/{str(i)}.csv", known=True, truth_override=truth_override)
        set_duration = time() - set_start_time
        set_durations.append(set_duration)
        print(set_count - i - 1, "set(s) remaining (" + str(timedelta(seconds=np.sum(set_durations))), "elapsed, ~" +
            str(timedelta(seconds=np.mean(set_durations[-5:]) * (set_count - i - 1))), "remaining)")
    print("\n\nCompleted in", str(timedelta(seconds=time() - sets_start_time)))

run_batches()
#write_results(paths, classify(), f"results_{paths_start}-{paths_end}.csv", known=True)
