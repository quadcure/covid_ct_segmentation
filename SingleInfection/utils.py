import time
import torchvision as tv
from PIL import Image
import datetime

from configuration import Config


config = Config()

FORMAT = "%d/%m/%Y %H:%M:%S UTC+0"

transform = tv.transforms.Compose([
    tv.transforms.Resize((config.test_size, config.test_size)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])




# Util Functions
# required by LungInfection.py
# -----------------------------

def prepare_name_based_on_time_seed():
    name = str(time.time())
    name = name.replace(".", "")
    return name


def prepare_image(image):

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = transform(image).unsqueeze(0)

    return image


def prepare_timestamp():
    now = datetime.datetime.utcnow()
    return now.strftime(FORMAT)

# --------------------------------
