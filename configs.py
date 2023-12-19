from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'

SOURCES_LIST = [IMAGE, VIDEO]

# Model Configs
# MODELS_DIR = ROOT/ 'pretrained-models'
# MODELS_DETECT = MODELS_DIR / 'blaze_face_short_range.tflite'
# MODELS_LANDMARKS = MODELS_DIR / 'face_landmarker.task'
# MODELS_SEGMENT = MODELS_DIR / 'selfie_segmenter.tflite'
# MODELS_STYLE = MODELS_DIR / 'face_stylizer_color_sketch.task'
# model_options = ['rtmdet_tiny_8xb32-300e_coco', 'mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco']
detection_model_name = 'rtmdet_tiny_8xb32-300e_coco'
detection_checkpoint = './checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
device = 'cpu:0'
config_path = './checkpoints/rtmdet_tiny_8xb32-300e_coco.py'
checkpoint = './checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'


# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE_MMDETECTION = IMAGES_DIR / 'demo.jpg'
DEFAULT_DETECT_IMAGE = ROOT / 'output/vis/demo.jpg'


# # Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'demo.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'demo_mot.mp4'
VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,
    'video_2': VIDEO_2_PATH,
}


# Detection Styles
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

# Segmentation style
BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white

# Page Layout Configuration
page_config = {}