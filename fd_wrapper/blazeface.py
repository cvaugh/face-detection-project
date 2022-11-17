import fd_wrapper
import numpy as np
import sys
from os import path
blazeface_path = fd_wrapper.relative_path("./BlazeFace_PyTorch")
sys.path.append(blazeface_path)
from blazeface import BlazeFace
from torch import from_numpy
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class BFDataset(Dataset):
    def __init__(self, images, resize=(128, 128)):
        self.images = images
        self.resize = resize
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return from_numpy(np.array(self.images[idx].resize(self.resize))).permute((2, 0, 1))

def create_instance(weights=path.join(blazeface_path, "blazeface.pth"), anchors=path.join(blazeface_path, "anchors.npy"),
                    min_score_threshold=0.75, min_suppression_threshold=0.3):
    instance = BlazeFace()
    instance.load_weights(weights)
    instance.load_anchors(anchors)
    instance.min_score_thresh = min_score_threshold
    instance.min_suppression_threshold = min_suppression_threshold
    return instance


def classify(instance, images, batch_size=512, num_workers=0):
    if not isinstance(images, list): images = [images]
    dl = DataLoader(BFDataset(images), num_workers=num_workers, pin_memory=True, shuffle=False, batch_size=min(len(images), batch_size))
    results = []
    # to do: optimize this loop
    progress = tqdm(dl)
    progress.set_description("Classifying images")
    for image in progress:
        detections = instance.predict_on_batch(image)
        dets = np.array([d.shape[0] for d in detections])
        results.append(dets != 0)
    return np.concatenate(results)
