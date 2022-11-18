from retinaface import RetinaFace
from tqdm import tqdm

def classify(paths):
    if not isinstance(paths, list): paths = [paths]
    results = []
    progress = tqdm(paths)
    progress.set_description("Classifying images")
    for path in progress:
        result = RetinaFace.detect_faces(path)
        results.append(isinstance(result, dict))
    return results
