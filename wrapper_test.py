import fd_wrapper as wrapper
from fd_wrapper import ssd

# This prevents unexpected behavior when using multithreading
if __name__ != "__main__":
    exit()

#dataset_path = "/path/to/filtered/flicker_1.2"
dataset_path = wrapper.relative_path("../filtered/flickr_1.2")

paths = wrapper.read_dataset(dataset_path)
print("Found", len(paths), "images")

# Small subset of images for testing
paths_subset = paths[:4096]

instance = ssd.create_instance()

results = ssd.classify(instance, paths_subset)

failed_cases = ""

for i in range(len(paths_subset)):
    if not results[i]:
        failed_cases += paths_subset[i] + "\n"

with open("failed_cases.log", "w") as f:
    f.writelines(failed_cases)
