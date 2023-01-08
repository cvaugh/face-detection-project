import fd_wrapper as wrapper
from fd_wrapper.detectors import *

wrapper.graph_results(
    results_dir="./results_temp",
    detectors=[blazeface, insightface, mtcnn, ssd],
    out_file="posterized.png",
    subtitle="Bit depth (0-8)",
    graphing=None, # must be one of "hue", "saturation", "value", or None
    size=(1920, 1200),
    dpi=96
)
