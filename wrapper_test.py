import fd_wrapper as wrapper
#from fd_wrapper import ssd
from fd_wrapper import mtcnn
#from fd_wrapper import blazeface
#from fd_wrapper import retinaface

# This prevents unexpected behavior when using multithreading
if __name__ != "__main__":
    exit()

#dataset_path = "/path/to/filtered/flickr_1.2"
dataset_path = wrapper.relative_path("../known")

paths = wrapper.read_dataset(dataset_path)
print("Found", len(paths), "images")

# Small subset of images for testing
#paths_subset = paths[:4096]

instance = mtcnn.create_instance()

results = mtcnn.classify(instance, paths)

def write_failed_cases(results):
    failed_cases = ""

    for i in range(len(paths)):
        if not results[i]:
            failed_cases += paths[i] + "\n"

    with open("failed_cases.log", "w") as f:
        f.writelines(failed_cases)

def write_results(results, known=False):
    with open("results.csv", "w") as f:
        if known:
            lines = ""
            for i in range(len(results)):
                expected = "positive" in paths[i]
                lines += paths[i] + ", " + str(int(results[i])) + ", " + str(int(expected)) + ", " + str(int(expected == results[i])) + "\n"
            f.writelines(lines)
        else:
            f.writelines("\n".join([paths[i] + ", " + str(int(results[i])) for i in range(len(results))]))

write_results(results, known=True)
