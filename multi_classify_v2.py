import fd_wrapper as wrapper
from fd_wrapper import transform
from fd_wrapper.detectors import *
from time import time
from datetime import timedelta
import numpy as np
from pathlib import Path
from os.path import dirname
from tqdm import tqdm

dataset_path = wrapper.relative_path("./known/200", root=__file__)

paths = wrapper.read_dataset(dataset_path)

batch_size = 128

num_steps = 256

transformer = transform.saturation_rotation

detectors = ["BlazeFace", "MTCNN", "RetinaFace", "SSD", "Ground Truth"]

results = {d: {path: [0 for i in range(num_steps)] for path in paths} for d in detectors}

for path, path_index in zip(paths, range(0, len(paths))):
    print(">>> Image", path_index + 1, "of", len(paths), "<<<")
    image = wrapper.load_image(path)
    transform_progress = tqdm(range(num_steps))
    transform_progress.set_description("Transforming image")
    transformed_images = [transformer(image, step) for step in transform_progress]
    batch_index = 0
    for batch_start in range(0, len(transformed_images), batch_size):
        batch = transformed_images[batch_start:batch_start + batch_size]
        print("> Classifying images", batch_start + 1, "to", batch_start + len(batch), "of", len(transformed_images))
        batch_results = {d: [0] * len(batch) for d in detectors}
        batch_results["BlazeFace"] = blazeface.classify(batch)
        batch_results["SSD"] = ssd.classify(batch)
        batch_results["RetinaFace"] = retinaface.classify(batch)
        batch_results["MTCNN"] = mtcnn.classify(batch)
        batch_results["Ground Truth"] = ["positive" in path for i in range(len(batch))]
        for d in detectors:
            for i in range(0, len(batch)):
                results[d][path][batch_size * batch_index + i] = batch_results[d][i]
        batch_index += 1

# to do: write results
