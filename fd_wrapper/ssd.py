import deepface.detectors.SsdWrapper
import numpy as np
import cv2
from tqdm import tqdm

def create_instance():
    return deepface.detectors.SsdWrapper.build_model()["face_detector"]

def __classify_image(instance, image, target_size=(300, 300)):
    img = cv2.resize(np.array(image), target_size)
    imageBlob = cv2.dnn.blobFromImage(img)

    instance.setInput(imageBlob)
    detections = instance.forward()

    return ((detections[:, :, :, 1]==1 ) & (detections[:, :, :, 2]>=0.9)).any()

def classify(instance, images, target_size=(300, 300)):
    if not isinstance(images, list): images = [images]
    results = []
    # to do: multithreading/optimization
    progress = tqdm(images)
    progress.set_description("Classifying images")
    for image in progress:
        results.append(__classify_image(instance, image, target_size))
    return results
