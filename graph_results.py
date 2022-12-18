import fd_wrapper as wrapper
from fd_wrapper.detectors import *

wrapper.graph_results(
    results_dir="./results_temp",
    detectors=[blazeface, insightface, mtcnn, ssd],
    out_file="hue.png",
    subtitle="Hue (0-255)",
    graphing="hue", # must be one of "hue", "saturation", "value", or None
    size=(1920, 1200),
    dpi=96
)
