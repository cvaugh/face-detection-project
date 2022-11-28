import os
from PIL import Image
from tqdm import tqdm
from fnmatch import filter
import numpy as np
from csv import reader

__all__ = ["transform"]

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

def load_image(path, image_size=None, as_ndarray=False):
    img = Image.open(path).convert("RGB")
    img if image_size is None else img.resize(image_size)
    return np.asarray(img) if as_ndarray else img

def load_images(paths, image_size=None, silent=False, as_ndarray=False):
    if not isinstance(paths, list): paths = [paths]
    if not silent:
        progress = tqdm(paths)
        progress.set_description("Loading images")
    iter = paths if silent else progress
    return [load_image(path, image_size, as_ndarray) for path in iter]

def get_ground_truth(path, split_path_at=None, path_separator=os.sep, relative_to=None):
    with open(path) as file:
        r = reader(file, delimiter="\t")
        faces = []
        not_faces = []
        ambiguous = []
        unclassified = []
        for row in r:
            if split_path_at is not None:
                if relative_to is None:
                    row[0] = row[0][row[0].index(split_path_at) + len(split_path_at) + len(path_separator):]
                else:
                    row[0] = os.path.join(relative_to,
                             row[0][row[0].index(split_path_at) + len(split_path_at) + len(path_separator):])
            match row[1]:
                case "FACE":
                    faces.append(row[0])
                case "NOT_FACE":
                    not_faces.append(row[0])
                case "AMBIGUOUS":
                    ambiguous.append(row[0])
                case _:
                    unclassified.append(row[0])
        return faces, not_faces, ambiguous, unclassified

def create_batches(items, batch_size=128):
    batches = []
    index = 0
    for i in range(0, len(items), batch_size):
        batches.append((index, items[i:i + batch_size]))
        index += 1
    return batches

def average_image(images, silent=False):
    y, x, z = np.shape(images[0])
    avg = np.zeros((y, x, z), dtype=float)
    if not silent:
        progress = tqdm(images)
        progress.set_description("Averaging images")
    for image in images if silent else progress:
        avg = avg + (image / len(images))
    avg = np.array(np.round(avg), dtype=np.uint8)
    return avg
