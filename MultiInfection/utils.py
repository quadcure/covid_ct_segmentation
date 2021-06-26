import torchvision as tv
from PIL import Image
import requests

import numpy as np

from configuration import Config
config = Config()


transform = tv.transforms.Compose([
    tv.transforms.Resize((config.test_size, config.test_size)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])




def prepare_image(image, psuedo_url):

    psuedo_image = Image.open(requests.get(psuedo_url, stream=True).raw)

    if psuedo_image.mode != "RGB":
        psuedo_image = psuedo_image.convert("RGB")

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = transform(image).unsqueeze(0)
    psuedo_image = transform(psuedo_image).unsqueeze(0)

    return image, psuedo_image



def split_class(path, w, h):
    im = Image.open(path).convert('L')
    im_array_red = np.array(im)  # 0, 38
    im_array_green = np.array(im)  # 0, 75
    
    uniquemidfinder = np.unique(im_array_red)
    mid = uniquemidfinder[1]
    print(np.unique(im_array_red))

    im_array_red[im_array_red != 0] = 1
    im_array_red[im_array_red == 0] = 255
    im_array_red[im_array_red == 1] = 0

    im_array_green[im_array_green != mid] = 0
    im_array_green[im_array_green == mid] = 255

    # Class1 = GroundGlassOpacities
    # Class2 = Consolidation
    class_one = Image.fromarray(im_array_red).convert('1').resize(size=(h, w))
    class_two = Image.fromarray(im_array_green).convert('1').resize(size=(h, w))

    return class_one, class_two
