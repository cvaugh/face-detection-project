from retinaface import RetinaFace
from tqdm import tqdm
import numpy as np
import cv2

__cached_instance = None

def __get_cached_instance():
    global __cached_instance
    if __cached_instance is None:
        __cached_instance = create_instance()
    return __cached_instance

def create_instance():
    return RetinaFace.build_model()

def classify(images, instance=None, threshold=0.9, allow_upscaling=True):
    if not isinstance(images, list): images = [images]
    if instance is None: instance = __get_cached_instance()
    results = []
    progress = tqdm(images)
    progress.set_description("[RetinaFace] Classifying images")
    for image in progress:
        result = RetinaFace.detect_faces(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), threshold, instance, allow_upscaling)
        results.append(isinstance(result, dict))
    return results
