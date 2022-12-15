import cv2
from deepface.detectors.SsdWrapper import build_model
from numpy import array
from tqdm import tqdm

__cached_instance = None

def __get_cached_instance():
    global __cached_instance
    if __cached_instance is None:
        __cached_instance = create_instance()
    return __cached_instance

def create_instance():
    return build_model()["face_detector"]

def __classify_image(instance, image, target_size=(300, 300)):
    img = cv2.resize(array(image), target_size)
    imageBlob = cv2.dnn.blobFromImage(img)

    instance.setInput(imageBlob)
    detections = instance.forward()

    return ((detections[:, :, :, 1]==1 ) & (detections[:, :, :, 2]>=0.9)).any()

def classify(images, instance=None, target_size=(300, 300)):
    if not isinstance(images, list): paths = [paths]
    if instance is None: instance = __get_cached_instance()
    results = []
    # to do: multithreading/optimization
    progress = tqdm(images)
    progress.set_description(f"[{name()}] Classifying images")
    for image in progress:
        results.append(__classify_image(instance, image.resize(target_size), target_size))
    return results

def name():
    return "SSD"
