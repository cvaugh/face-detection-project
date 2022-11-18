import fd_wrapper
from deepface.detectors.SsdWrapper import build_model
from numpy import array
import cv2
from tqdm import tqdm

def create_instance():
    return build_model()["face_detector"]

def __classify_image(instance, image, target_size=(300, 300)):
    img = cv2.resize(array(image), target_size)
    imageBlob = cv2.dnn.blobFromImage(img)

    instance.setInput(imageBlob)
    detections = instance.forward()

    return ((detections[:, :, :, 1]==1 ) & (detections[:, :, :, 2]>=0.9)).any()

def classify(instance, paths, target_size=(300, 300)):
    if not isinstance(paths, list): paths = [paths]
    results = []
    # to do: multithreading/optimization
    progress = tqdm(paths)
    progress.set_description("Classifying images")
    for path in progress:
        results.append(__classify_image(instance, fd_wrapper.load_image(path, target_size), target_size))
    return results
