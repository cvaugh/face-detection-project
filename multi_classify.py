import fd_wrapper as wrapper
from fd_wrapper import blazeface
from fd_wrapper import mtcnn
from fd_wrapper import retinaface
from fd_wrapper import ssd
from time import time
from datetime import timedelta
from numpy import mean

if __name__ != "__main__":
    exit()

#dataset_path = wrapper.relative_path("../filtered/flickr_1.2")
dataset_path = wrapper.relative_path("../known")

paths = wrapper.read_dataset(dataset_path)

batch_size = 100

# Small subset of images for testing
batches = [paths[i:i + batch_size] for i in range(0, len(paths), batch_size)]
batch_count = len(batches)
print("Found", len(paths), "images (" + str(batch_count), "batches of size", str(batch_size) + ")")

ssd_instance = ssd.create_instance()
mtcnn_instance = mtcnn.create_instance()
blazeface_instance = blazeface.create_instance()
retinaface_instance = retinaface.create_instance()

results = [None] * batch_count

first_start_time = time()
durations = []

# to do: multithreading
for i in range(batch_count):
    print("\nBatch", (i + 1), "of", batch_count)
    start_time = time()
    images = wrapper.load_images(batches[i])

    results_blazeface = blazeface.classify(blazeface_instance, images)
    results_mtcnn = mtcnn.classify(mtcnn_instance, images)
    results_ssd = ssd.classify(ssd_instance, images)
    results_retinaface = retinaface.classify(retinaface_instance, images)

    results[i] = {
        "blazeface": results_blazeface,
        "mtcnn": results_mtcnn,
        "retinaface": results_retinaface,
        "ssd": results_ssd
    }
    end_time = time()
    duration = end_time - start_time
    durations.append(duration)
    avg = mean(durations)
    print("Completed in", str(timedelta(seconds=duration)),
        "(total:", str(timedelta(seconds=end_time - first_start_time)) + ", avg:", str(timedelta(seconds=avg)) + ",",
        "remaining: ~" + str(timedelta(seconds=avg * (batch_count - i - 1))) + ")")


print("\nBatches completed in", str(timedelta(seconds=time() - first_start_time)))

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

def write_results(paths, results, known=False):
    with open("results.csv", "w") as f:
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

write_results(paths, results_full, known=True)
