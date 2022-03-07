import yaml
from easydict import EasyDict

with open("./config/config.yaml", "r") as file:
    config = EasyDict(yaml.safe_load(file))
