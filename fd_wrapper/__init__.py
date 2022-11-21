import os
from PIL import Image
from tqdm import tqdm
from fnmatch import filter

__all__ = ["blazeface", "mtcnn", "retinaface", "ssd"]

# OS-independent way to get a path relative to the current file
def relative_path(path, root=__file__):
    return os.path.join(os.path.dirname(root), path)

# Read faces dataset to a list of paths
# Ignores file paths that do not end in .jpg
def read_dataset(root_dir, filename_pattern="*.jpg"):
    paths = []
    progress = tqdm(os.walk(root_dir), unit="dir")
    progress.set_description("Scanning for images")
    for root, dirs, files in progress:
        for path in filter(files, filename_pattern):
            paths.append(os.path.join(root, path))
    return paths

def load_image(path, image_size=None):
    img = Image.open(path).convert("RGB")
    return img if image_size is None else img.resize(image_size)

def load_images(paths, image_size=None, silent=False):
    if silent:
        if not isinstance(paths, list): paths = [paths]
        return [load_image(path, image_size) for path in paths]
    else:
        progress = tqdm(load_images(paths, image_size, silent=True), total=len(paths))
        progress.set_description("Loading images")
        return [image for image in progress]
