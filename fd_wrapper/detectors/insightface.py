import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

import fd_wrapper

__cached_instance = None

def __get_cached_instance():
    global __cached_instance
    if __cached_instance is None:
        __cached_instance = create_instance()
    return __cached_instance

def create_instance(gpu=True):
    instance = FaceAnalysis(providers=["CUDAExecutionProvider" if gpu else "CPUExecutionProvider"])
    instance.prepare(ctx_id=0, det_size=(512, 512))
    return instance


def classify(images, instance=None):
    if not isinstance(images, list): images = [images]
    if instance is None: instance = __get_cached_instance()
    results = []
    progress = tqdm(images)
    progress.set_description(f"[{name()}] Classifying images")
    for image in progress:
        faces = instance.get(np.array(image))
        results.append(len(faces) > 0)
    return results

def name():
    return "InsightFace"
