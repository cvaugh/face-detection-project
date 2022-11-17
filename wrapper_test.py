import fd_wrapper as wrapper
from fd_wrapper import blazeface

# This prevents unexpected behavior when using multithreading
if __name__ != "__main__":
    exit()

#dataset_path = "/path/to/filtered/flicker_1.2"
dataset_path = wrapper.relative_path("../filtered/flickr_1.2")

paths = wrapper.read_dataset(dataset_path)
print("Found", len(paths), "images")

# Uncomment this statement to load all images into memory at once
# Not recommended as it requires >64GB of memory
#images = load_images(paths)

# Small subset of images for testing
paths_subset = paths[:4096]

# to do: load images in batches as needed
images = wrapper.load_images(paths_subset)

bf = blazeface.create_instance()

results_blazeface = blazeface.classify(bf, images)

failed_cases = ""

for i in range(len(paths_subset)):
    if not results_blazeface[i]:
        failed_cases += paths_subset[i] + "\n"

with open("failed_cases_blazeface.log", "w") as f:
    f.writelines(failed_cases)
