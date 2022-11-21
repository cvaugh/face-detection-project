from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
from tqdm import tqdm

def create_instance(select_largest=False):
    return MTCNN(select_largest=select_largest)

def classify(instance, images, resize=(512, 512)):
    if not isinstance(images, list): images = [images]

    images = [image.resize(resize) for image in images]

    print("[MTCNN] Classifying images...", end="\r")
    output = instance(images)

    results = [None] * len(images)
    
    progress = tqdm(total=len(images))
    progress.set_description("[MTCNN] Processing results")
    for index, (out, path) in enumerate(zip(output, images)):
        results[index] = out is not None
        progress.update()
    progress.close()
    return results
