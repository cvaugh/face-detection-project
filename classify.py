import fd_wrapper as wrapper
from fd_wrapper import transform
from fd_wrapper.detectors import *

paths = wrapper.get_ground_truth(wrapper.relative_path("./filtered/labels.tsv", root=__file__),
    split_path_at="flickr_1.2", relative_to=wrapper.relative_path("./filtered/flickr_1.2", root=__file__))[0]

wrapper.classify_sets(
    paths,
    truncate_paths=50000,
    batch_size=256,
    detectors=[
        blazeface,
        insightface,
        mtcnn,
        ssd
    ],
    transform=None,
    start_index=0,
    set_count=1,
    truth_override=None
)
