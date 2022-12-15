import fd_wrapper as wrapper
from fd_wrapper import transform
from fd_wrapper.detectors import *
from time import time
from datetime import timedelta
import numpy as np
from pathlib import Path
from os.path import dirname
from tqdm import tqdm

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

results = [None] * batch_count

def classify(transform=None, set_index=0):
    batch_start_time = time()
    durations = []

    for i, batch in batches:
        print("Batch", i + 1, "of", batch_count)
        start_time = time()
        images = wrapper.load_images(batch)

        if transform is not None:
            progress = tqdm(images)
            progress.set_description("Transforming images")
            images = [transform(image, set_index) for image in progress]

        results_blazeface = blazeface.classify(images)
        results_insightface = insightface.classify(images)
        results_mtcnn = mtcnn.classify(images)
        #results_retinaface = retinaface.classify(images)
        results_retinaface = [0] * len(images)
        results_ssd = ssd.classify(images)

        results[i] = {
            "blazeface": results_blazeface,
            "insightface": results_insightface,
            "mtcnn": results_mtcnn,
            "retinaface": results_retinaface,
            "ssd": results_ssd
        }
        end_time = time()
        duration = end_time - start_time
        durations.append(duration)
        avg = np.mean(durations[-5:])
        print("Completed in", str(timedelta(seconds=duration)),
            "(total:", str(timedelta(seconds=end_time - batch_start_time)) + ", avg:", str(timedelta(seconds=avg)) + ",",
            "remaining: ~" + str(timedelta(seconds=avg * (batch_count - i - 1))) + ")")

    print("\nBatches completed in", str(timedelta(seconds=time() - batch_start_time)))

    results_full = {
        "blazeface": [],
        "insightface": [],
        "mtcnn": [],
        "retinaface": [],
        "ssd": []
    }

    for entry in results:
        results_full["blazeface"].extend(entry["blazeface"])
        results_full["insightface"].extend(entry["insightface"])
        results_full["mtcnn"].extend(entry["mtcnn"])
        results_full["retinaface"].extend(entry["retinaface"])
        results_full["ssd"].extend(entry["ssd"])
    
    return results_full

def write_results(paths, results, path="results.csv", known=False):
    Path(dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if known:
            lines = "Path, BlazeFace, InsightFace, MTCNN, RetinaFace, SSD, Ground Truth\n"
        else:
            lines = "Path, BlazeFace, InsightFace, MTCNN, RetinaFace, SSD\n"
        for i in range(len(paths)):
            blazeface_result = str(int(results["blazeface"][i]))
            insightface_result = str(int(results["insightface"][i]))
            mtcnn_result = str(int(results["mtcnn"][i]))
            retinaface_result = str(int(results["retinaface"][i]))
            ssd_result = str(int(results["ssd"][i]))
            if known:
                expected = str(int("positive" in paths[i]) if truth_override is None else truth_override)
                lines += paths[i] + ", " + blazeface_result + ", " + insightface_result + ", " + mtcnn_result + ", " + retinaface_result + ", " + ssd_result + ", " + expected + "\n"
            else:
                lines += paths[i] + ", " + blazeface_result + ", " + insightface_result + ", " + mtcnn_result + ", " + retinaface_result + ", " + ssd_result + "\n"
        f.writelines(lines)

def run_batches():
    sets_start_time = time()
    set_count = 255
    set_durations = []
    for i in range(set_count):
        set_start_time = time()
        print("\n(Set " + str(i) + "/" + str(set_count) + ") ", end="")
        write_results(paths, classify(transform=transform.hue_rotation, set_index=i), path="results_temp/" + str(i) + ".csv", known=True)
        set_duration = time() - set_start_time
        set_durations.append(set_duration)
        print(set_count - i - 1, "set(s) remaining (" + str(timedelta(seconds=np.sum(set_durations))), "elapsed, ~" +
            str(timedelta(seconds=np.mean(set_durations[-5:]) * (set_count - i - 1))), "remaining)")
    print("\n\nCompleted in", str(timedelta(seconds=time() - sets_start_time)))

run_batches()
#write_results(paths, classify(), path=f"results_{paths_start}-{paths_end}.csv", known=True)
