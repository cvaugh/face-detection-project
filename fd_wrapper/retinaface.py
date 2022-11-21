from retinaface import RetinaFace
from tqdm import tqdm
import numpy as np
import cv2

def create_instance():
    return RetinaFace.build_model()

def classify(instance, images, threshold=0.9, allow_upscaling=True):
    if not isinstance(images, list): images = [images]
    results = []
    progress = tqdm(images)
    progress.set_description("[RetinaFace] Classifying images")
    for image in progress:
        result = RetinaFace.detect_faces(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), threshold, instance, allow_upscaling)
        results.append(isinstance(result, dict))
    return results
