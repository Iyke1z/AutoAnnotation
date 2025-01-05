from crp.attribution import CondAttribution
from zennit.torchvision import ResNetCanonizer

from utils.crp import CondAttributionLocalization, CondAttributionSegmentation, FeatureVisualizationLocalization, \
    FeatureVisualizationSegmentation
from utils.zennit_canonizers import YoloV5V6Canonizer, DeepLabV2Canonizer, DeepLabV3PlusCanonizer
from utils.zennit_composites import EpsilonPlusFlat, EpsilonGammaFlat

COMPOSITES = {
    # object detectors
    "yolov5": EpsilonPlusFlat,
    "yolov6": EpsilonGammaFlat,
    # segmentation models
    "unet": EpsilonPlusFlat,
    "deeplabv2": EpsilonPlusFlat,
    "deeplabv3plus": EpsilonPlusFlat,
    "yolov8n": EpsilonGammaFlat,
    "yolov5_nc7":EpsilonPlusFlat,
    "yolov5_nc7_attention": EpsilonPlusFlat

}

CANONIZERS = {
    # object detectors
    "yolov5": YoloV5V6Canonizer,
    "yolov6": YoloV5V6Canonizer,
    # segmentation models
    "unet": ResNetCanonizer,
    "deeplabv2": DeepLabV2Canonizer,
    "deeplabv3plus": DeepLabV3PlusCanonizer,
    "yolov8n": YoloV5V6Canonizer,
    "yolov5_nc7":YoloV5V6Canonizer,
    "yolov5_nc7_attention": YoloV5V6Canonizer

}

ATTRIBUTORS = {
    # object detectors
    "yolov5": CondAttributionLocalization,
    "yolov6": CondAttributionLocalization,
    # segmentation models
    "unet": CondAttributionSegmentation,
    "deeplabv2": CondAttributionSegmentation,
    "deeplabv3plus": CondAttributionSegmentation,
    "yolov8n": CondAttributionLocalization,
    "yolov5_nc7":CondAttributionLocalization,
    "yolov5_nc7_attention": CondAttributionLocalization

}

VISUALIZATIONS = {
    # object detectors
    "yolov5": FeatureVisualizationLocalization,
    "yolov6": FeatureVisualizationLocalization,
    # segmentation models
    "unet": FeatureVisualizationSegmentation,
    "deeplabv2": FeatureVisualizationSegmentation,
    "deeplabv3plus": FeatureVisualizationSegmentation,
    "yolov8n": FeatureVisualizationLocalization,
     "yolov5_nc7": FeatureVisualizationLocalization,
    "yolov5_nc7_attention": FeatureVisualizationLocalization

}