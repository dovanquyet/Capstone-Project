from .classification.efficientnet import EfficientNetVariation, EfficientNetV2S
from .detection.rapid import RAPiD

models = {
    "detection": {
        "rapid": RAPiD,
    },
    "classification": {
        "efficientnet": EfficientNetVariation,
        "efficientnetv2s": EfficientNetV2S,
    }
}
