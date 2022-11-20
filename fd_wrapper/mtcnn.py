from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
from tqdm import tqdm

def create_instance(select_largest=False):
    return MTCNN(select_largest=select_largest)

def classify(instance, paths):
    if not isinstance(paths, list): paths = [paths]
    images = [np.array(Image.open(path).convert("RGB")) for path in paths]

    output = instance(images)

    results = [None] * len(paths)
    
    for index, (out, path) in tqdm(enumerate(zip(output, paths)), total=len(paths)):
        results[index] = int(out is not None)
    return results
