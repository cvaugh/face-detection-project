from facenet_pytorch import MTCNN
from tqdm import tqdm

__cached_instance = None

def __get_cached_instance():
    global __cached_instance
    if __cached_instance is None:
        __cached_instance = create_instance()
    return __cached_instance

def create_instance(select_largest=False):
    return MTCNN(select_largest=select_largest)

def classify(images, instance=None, resize=(512, 512)):
    if not isinstance(images, list): images = [images]
    if instance is None: instance = __get_cached_instance()

    images = [image.resize(resize) for image in images]

    print(f"[{name()}] Classifying images...", end="\r")
    output = instance(images)

    results = [None] * len(images)
    
    progress = tqdm(total=len(images))
    progress.set_description(f"[{name()}] Processing results")
    for index, (out, path) in enumerate(zip(output, images)):
        results[index] = out is not None
        progress.update()
    progress.close()
    return results

def name():
    return "MTCNN"
