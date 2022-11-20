from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
from tqdm import tqdm

def create_instance(select_largest=False):
    return MTCNN(select_largest=select_largest)

def classify(instance, images):
    if not isinstance(images, list): images = [images]

    # to do: resolve "VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences"
    output = instance(images)

    results = [None] * len(images)
    
    for index, (out, path) in tqdm(enumerate(zip(output, images)), total=len(images)):
        results[index] = int(out is not None)
    return results
