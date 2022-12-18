import fd_wrapper as wrapper
from fd_wrapper import transform
from fd_wrapper.detectors import *

dataset_path = wrapper.relative_path("./known/200/positive", root=__file__)

paths = wrapper.read_dataset(dataset_path)
#paths = wrapper.get_ground_truth(wrapper.relative_path("./filtered/labels.tsv", root=__file__),
#    split_path_at="flickr_1.2", relative_to=wrapper.relative_path("./filtered/flickr_1.2", root=__file__))[0]

wrapper.classify_sets(
    paths,
    truncate_paths=0,
    batch_size=100,
    detectors=[
        blazeface,
        insightface,
        mtcnn,
        ssd
    ],
    transform=transform.hue_rotation,
    start_index=0,
    set_count=255,
    truth_override=None
)
