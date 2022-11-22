import fd_wrapper as wrapper
from fd_wrapper import transform
from fd_wrapper.detectors import *
from time import time
from datetime import timedelta
import numpy as np
from PIL import Image
from pathlib import Path
from os.path import dirname
from tqdm import tqdm

if __name__ != "__main__":
    exit()

#dataset_path = wrapper.relative_path("./filtered/flickr_1.2", root=__file__)
dataset_path = wrapper.relative_path("./known/200", root=__file__)
#dataset_path = wrapper.relative_path("./experiments/b", root=__file__)

paths = wrapper.read_dataset(dataset_path)

batch_size = len(paths)

# Small subset of images for testing
batches = [paths[i:i + batch_size] for i in range(0, len(paths), batch_size)]
batch_count = len(batches)
print("Found", len(paths), "images (" + str(batch_count), "batch" if batch_count == 1 else "batches", "of size", str(batch_size) + ")")

results = [None] * batch_count

def classify(transform=None, **kwargs):
    batch_start_time = time()
    durations = []

    # to do: multithreading
    for i in range(batch_count):
        print("Batch", (i + 1), "of", batch_count)
        start_time = time()
        images = wrapper.load_images(batches[i])

        if transform is not None:
            progress = tqdm(images)
            progress.set_description("Transforming images")
            images = [transform(image, int(kwargs.get("rot"))) for image in progress]

        results_blazeface = blazeface.classify(images)
        results_mtcnn = mtcnn.classify(images)
        results_ssd = ssd.classify(images)
        #results_retinaface = retinaface.classify(images)
        results_retinaface = [False] * len(images)

        results[i] = {
            "blazeface": results_blazeface,
            "mtcnn": results_mtcnn,
            "retinaface": results_retinaface,
            "ssd": results_ssd
        }
        end_time = time()
        duration = end_time - start_time
        durations.append(duration)
        avg = np.mean(durations)
        print("Completed in", str(timedelta(seconds=duration)),
            "(total:", str(timedelta(seconds=end_time - batch_start_time)) + ", avg:", str(timedelta(seconds=avg)) + ",",
            "remaining: ~" + str(timedelta(seconds=avg * (batch_count - i - 1))) + ")")

    print("\nBatches completed in", str(timedelta(seconds=time() - batch_start_time)))

    results_full = {
        "blazeface": [],
        "mtcnn": [],
        "retinaface": [],
        "ssd": []
    }

    for entry in results:
        results_full["blazeface"].extend(entry["blazeface"])
        results_full["mtcnn"].extend(entry["mtcnn"])
        results_full["retinaface"].extend(entry["retinaface"])
        results_full["ssd"].extend(entry["ssd"])
    
    return results_full

def write_results(paths, results, path="results.csv", known=False):
    Path(dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if known:
            lines = "Path, BlazeFace, MTCNN, RetinaFace, SSD, Actual\n"
        else:
            lines = "Path, BlazeFace, MTCNN, RetinaFace, SSD\n"
        for i in range(len(paths)):
            blazeface_result = str(int(results["blazeface"][i]))
            mtcnn_result = str(int(results["mtcnn"][i]))
            retinaface_result = str(int(results["retinaface"][i]))
            ssd_result = str(int(results["ssd"][i]))
            if known:
                expected = str(int("positive" in paths[i]))
                lines += paths[i] + ", " + blazeface_result + ", " + mtcnn_result + ", " + retinaface_result + ", " + ssd_result + ", " + expected + "\n"
            else:
                lines += paths[i] + ", " + blazeface_result + ", " + mtcnn_result + ", " + retinaface_result + ", " + ssd_result + "\n"
        f.writelines(lines)

sets_start_time = time()
set_count = 256
set_durations = []
for i in range(set_count):
    set_start_time = time()
    print("\n(Set " + str(i) + "/" + str(set_count) + ") ", end="")
    write_results(paths, classify(transform=transform.value_rotation, rot=i), path="results_temp/" + str(i) + ".csv", known=True)
    set_duration = time() - set_start_time
    set_durations.append(set_duration)
    print(set_count - i + 1, "sets remaining (" + str(timedelta(seconds=np.sum(set_durations))), "elapsed, ~" +
        str(timedelta(seconds=np.mean(set_durations) * (set_count - i - 1))), "remaining)")

#write_results(paths, classify(), path="results.csv", known=True)

print("\n\nCompleted in", str(timedelta(seconds=time() - sets_start_time)))
