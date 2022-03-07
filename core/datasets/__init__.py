from .custom import CustomizedDataset
from torchvision.datasets import ImageFolder

readers = {
    "custom": CustomizedDataset,
    "image_folder": ImageFolder,
}
