import csv
import os
from colorsys import hsv_to_rgb
from datetime import timedelta
from fnmatch import filter
from pathlib import Path
from sys import stderr
from time import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

__all__ = ["transform"]

# OS-independent way to get a path relative to the current file
def relative_path(path, root=__file__):
    return os.path.join(os.path.dirname(root), path)

# Read faces dataset to a list of paths
# Ignores file paths that do not end in .jpg
def read_dataset(root_dir, filename_pattern="*.jpg"):
    paths = []
    progress = tqdm(os.walk(root_dir), unit="dir")
    progress.set_description("Scanning for images")
    for root, dirs, files in progress:
        for path in filter(files, filename_pattern):
            paths.append(os.path.join(root, path))
    return paths

def load_image(path, image_size=None, as_ndarray=False):
    img = Image.open(path).convert("RGB")
    img if image_size is None else img.resize(image_size)
    return np.asarray(img) if as_ndarray else img

def load_images(paths, image_size=None, silent=False, as_ndarray=False):
    if not isinstance(paths, list): paths = [paths]
    if not silent:
        progress = tqdm(paths)
        progress.set_description("Loading images")
    iter = paths if silent else progress
    return [load_image(path, image_size, as_ndarray) for path in iter]

def get_ground_truth(path, split_path_at=None, path_separator=os.sep, relative_to=None):
    # to do: load into dictionary of attributes
    with open(path) as file:
        r = csv.reader(file, delimiter="\t")
        faces = []
        not_faces = []
        ambiguous = []
        unclassified = []
        for row in r:
            if split_path_at is not None:
                if relative_to is None:
                    row[0] = row[0][row[0].index(split_path_at) + len(split_path_at) + len(path_separator):]
                else:
                    row[0] = os.path.join(relative_to,
                             row[0][row[0].index(split_path_at) + len(split_path_at) + len(path_separator):])
            match row[1]:
                case "FACE":
                    faces.append(row[0])
                case "NOT_FACE":
                    not_faces.append(row[0])
                case "AMBIGUOUS":
                    ambiguous.append(row[0])
                case _:
                    unclassified.append(row[0])
        return faces, not_faces, ambiguous, unclassified

def create_batches(items, batch_size=128):
    batches = []
    index = 0
    for i in range(0, len(items), batch_size):
        batches.append((index, items[i:i + batch_size]))
        index += 1
    return batches

def average_image(images, silent=False):
    y, x, z = np.shape(images[0])
    avg = np.zeros((y, x, z), dtype=float)
    if not silent:
        progress = tqdm(images)
        progress.set_description("Averaging images")
    for image in images if silent else progress:
        avg = avg + (image / len(images))
    avg = np.array(np.round(avg), dtype=np.uint8)
    return avg

def classify_batches(batches, detectors, transform=None, transform_offset=0):
    results = [None for batch in batches]
    batch_start_time = time()
    durations = []

    for i, batch in batches:
        print(f"Batch {i + 1} of {len(batches)}")
        start_time = time()
        images = load_images(batch)

        if transform is not None:
            progress = tqdm(images)
            progress.set_description("Transforming images")
            # to do: improve performance
            images = [transform(image, transform_offset) for image in progress]

        results[i] = { detector.name(): detector.classify(images) for detector in detectors }

        end_time = time()
        duration = end_time - start_time
        durations.append(duration)
        avg = np.mean(durations[-5:])
        print(f"Completed in {timedelta(seconds=duration)} (total: {timedelta(seconds=end_time - batch_start_time)}," \
              f" avg: {timedelta(seconds=avg)}, remaining: ~{timedelta(seconds=avg * (len(batches) - i - 1))})")

    print(f"\nBatches completed in {timedelta(seconds=time() - batch_start_time)}")

    results_dict = { detector.name(): [] for detector in detectors }

    for entry in results:
        for detector in detectors:
            results_dict[detector.name()].extend(entry[detector.name()])
    
    return results_dict

def classify_sets(paths, detectors, transform=None, batch_size=128, truncate_paths=0, start_index=0, set_count=1, truth_override=None):
    if truncate_paths > 0:
        paths = paths[:truncate_paths]
    batches = create_batches(paths, batch_size)
    batch_count = len(batches)
    print(f"Found {len(paths)} images ({batch_count} {'batch' if batch_count == 1 else 'batches'} of size {batch_size})")
    start_time = time()
    durations = []
    for i in range(max(0, start_index), set_count):
        start_time = time()
        print(f"\n(Set {i + 1}/{set_count}) ", end="")
        write_results(paths, classify_batches(batches, detectors, transform, i),
                              detectors, f"results_temp/{str(i)}.csv", True, truth_override)
        duration = time() - start_time
        durations.append(duration)
        remaining = set_count - i - 1
        print(f"{remaining} set(s) remaining ({timedelta(seconds=np.sum(durations))}",
              f" elapsed, ~{timedelta(seconds=np.mean(durations[-5:]) * remaining)} remaining)")
    print(f"\n\nCompleted in {timedelta(seconds=time() - start_time)}")

def write_results(paths, results, detectors, path, known=False, truth_override=None):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    header = [detector.name() for detector in detectors]
    header.insert(0, "Path")
    if known:
        header.append("Ground Truth")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", lineterminator="\n", quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(header)
        for i in range(len(paths)):
            row = [int(results[detector.name()][i]) for detector in detectors]
            row.insert(0, paths[i])
            if known:
                row.append(int("positive" in paths[i]) if truth_override is None else truth_override)
            writer.writerow(row)

def graph_results(results_dir, detectors, results_file_filter="*.csv", out_file=None, title="Accuracy when classifying {image_count} labeled images",
                  subtitle="", graphing=None, size=(1920, 1200), dpi=96, silent=False):
    assert len(detectors) > 0, "Must pass at least one detector"
    dets = [detector.name() for detector in detectors]
    paths = []
    for root, dirs, files in os.walk(results_dir):
        for path in filter(files, results_file_filter):
            paths.append(os.path.join(root, path))

    if len(paths) == 0:
        print("No results found", file=stderr)
        return

    shifts = [int(os.path.splitext(os.path.basename(path))[0]) for path in paths]

    paths_sorted = np.array(paths)[np.argsort(shifts)[::1]]

    shifts = np.sort(shifts)

    entries = [None] * len(paths_sorted)

    progress = tqdm(range(len(entries)), disable=silent)
    progress.set_description("Loading results")
    for i in progress:
        entry = {"shift": shifts[i], "results": []}
        with open(paths_sorted[i]) as f:
            next(f) # skip header
            r = csv.reader(f, delimiter=",")
            for line in r:
                e = np.array(line[1:len(line)], dtype=int)
                entry["results"].append(e)
        entry["results"] = np.array(entry["results"])
        entries[i] = entry
    progress.close()

    results = [None] * len(dets)

    image_count = len(entries[0]["results"])

    fig, ax = plt.subplots(len(dets) + 1, sharex=True)
    fig.suptitle(title.format(image_count=image_count))
    fig.tight_layout()

    graph_xlabel = subtitle

    line_color = "blue" if graphing is None else "gray"

    y_min = float('inf')
    y_max = float('-inf')

    for i in range(len(dets)):
        results[i] = np.array([100 * np.sum(entry["results"][:, i] == entry["results"][:, len(dets)]) / image_count for entry in entries])
        if np.any(results[i] != 0.0):
            y_min = np.min([y_min, np.min(results[i])])
            y_max = np.max([y_max, np.max(results[i])])

    if y_min == y_max:
        y_min -= 0.01
        y_max += 0.01
    y_range = y_max - y_min
    y_padding = y_range / 10

    progress = tqdm(range(len(dets)), disable=silent)
    progress.set_description("Plotting results")
    for i in progress:
        ax[i].set_ylim(y_min - y_padding, y_max + y_padding)
        ax[i].set_title(dets[i])
        ax[i].set_ylabel("Accuracy (percent)")
        ax[i].plot(shifts, results[i], color=line_color, linewidth=1)
        if graphing is not None:
            for j in range(0, np.max(shifts) + 1):
                match graphing:
                    case "hue":
                        color = hsv_to_rgb(j / 255, 1, 1)
                    case "saturation":
                        color = hsv_to_rgb(0, j / 255, 0.9)
                    case "value":
                        color = hsv_to_rgb(0, 1, j / 255)
                    case _:
                        color = "red"
                ax[i].plot(j, results[i][j], color=color, marker="o", markersize=3)
        ax[len(dets)].plot(shifts, results[i], linewidth=1)
    progress.close()
        
    ax[len(dets)].set_ylim(y_min - y_padding, y_max + y_padding)
    ax[len(dets)].set_title("All detectors")
    ax[len(dets)].set_ylabel("Accuracy (percent)")
    ax[len(dets)].set_xlabel(graph_xlabel)
    ax[len(dets)].legend(dets, loc="upper right", prop={"size": 8})

    if out_file is None:
        plt.show()
    else:
        plt.gcf().set_size_inches(size[0]/dpi, size[1]/dpi)
        plt.savefig(out_file)
