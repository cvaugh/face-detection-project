import os
from PIL import Image
from tqdm import tqdm
from fnmatch import filter

__all__ = ["blazeface", "mtcnn", "retinaface", "ssd"]

# OS-independent way to get a path relative to the current file
def relative_path(path):
    return os.path.join(os.path.dirname(__file__), path)

# Read faces dataset to a list of paths
# Ignores file paths that do not end in .jpg
def read_dataset(root_dir):
    paths = []
    progress = tqdm(os.walk(root_dir))
    progress.set_description("Scanning for images")
    for root, dirs, files in progress:
        for path in filter(files, "*.jpg"):
            path = os.path.join(root, path)
            paths.append(path)
    return paths

def __load_image(file, image_size):
    img = Image.open(file).convert("RGB")
    return img if image_size is None else img.resize(image_size)

def load_images(files, image_size=None, silent=False):
    if silent:
        if not isinstance(files, list): files = [files]
        return [__load_image(file, image_size) for file in files]
    else:
        progress = tqdm(load_images(files, image_size, silent=True), total=len(files))
        progress.set_description("Loading images")
        return [image for image in progress]
