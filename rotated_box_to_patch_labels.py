import argparse
from create_json import CreateJson

parser = argparse.ArgumentParser(description='Create json for dataset input')
parser.add_argument(
    "--dataset_name",
    type=str,
    help='Name of dataset one want to train',
)
parser.add_argument(
    '--split',
    help='train or validation dataset',
    type=str,
)
parser.add_argument(
    '--input_json',
    help='absolute path to annotations file'
)
parser.add_argument(
    '--image_root',
    default='',
    help='absolute path to image root if in annotations file there are only relative paths',
)
parser.add_argument(
    '--intersection_threshold',
    default=0.3,
    type=float,
    help='threshold for patch class',
)
parser.add_argument(
    '--patch_size',
    default=16,
    type=int,
)
parser.add_argument(
    '--polygon',
    default='box',
    type=str,
    help='only 2 types of polygons are available: "box" - for object detection and "seg" for segmentation tasks'
)
parser.add_argument(
    '--save_root',
    default='./annotations',
)

args = parser.parse_args()


instance = CreateJson(
    annotation_path=args.input_json, 
    image_path=args.image_root,
    patch_size=args.patch_size, 
    intersection_threshold=args.intersection_threshold)

instance.fillpoly(
    path_save=args.save_root, 
    polygon=args.polygon, 
    split=args.split, 
    dataset_name=args.dataset_name)