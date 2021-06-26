import torch
import flask
import torchvision as tv
from model import Inf_Net

from PIL import Image
import time
from google.cloud import storage
from google.cloud import bigquery
from tempfile import NamedTemporaryFile
import os
import imageio
import numpy as np
import datetime


app = flask.Flask(__name__)
model = None
TEST_SIZE = 352
FORMAT = "%d/%m/%Y %H:%M:%S UTC+0"

BUCKET_NAME = "nosensitivebucket"
ORIGINAL_IMAGES = "images/original"
INFECTION_IMAGES = "images/infection"

bucket = None
bigquery_client = None

transform = tv.transforms.Compose([
    tv.transforms.Resize((TEST_SIZE, TEST_SIZE)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])


def prepare_name_based_on_time_seed():
    name = str(time.time())
    name = name.replace(".", "")
    return name


def load_bucket():
    global bucket
    storage_client = storage.Client.from_service_account_json(json_credentials_path="deployment.json")
    bucket = storage_client.get_bucket(BUCKET_NAME)

def load_bigquery():
    global bigquery_client
    bigquery_client = bigquery.Client.from_service_account_json(json_credentials_path="deployment.json")


def load_model():
    # Initializing model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    global model
    model = Inf_Net()

    # loading weights
    model_state = torch.hub.load_state_dict_from_url('https://github.com/saahiluppal/Inf-Net/releases/download/v1.0/Semi-Inf-Net-100.pth', 
                                    map_location=device, progress=True)
    model.load_state_dict(model_state)

    # loading model on specified device
    # and disabling gradient flow.
    model.to(device)
    model.eval()


def save_original_image_to_bucket(name, original_image):
    path = os.path.join(ORIGINAL_IMAGES, name) + ".jpeg"

    with NamedTemporaryFile() as temp:
        iname = "".join([str(temp.name),".jpeg"])
        imageio.imwrite(iname, original_image)

        blob = bucket.blob(path)
        blob.upload_from_filename(iname, content_type="image/jpeg")

        blob.make_public()
        return blob.public_url

#        private_url = os.path.join("https://storage.cloud.google.com", BUCKET_NAME, path)
#
#        return private_url


def save_infection_image_to_bucket(name, infection_image):
    path = os.path.join(INFECTION_IMAGES, name) + ".jpeg"

    with NamedTemporaryFile() as temp:
        iname = "".join([str(temp.name),".jpeg"])
        imageio.imwrite(iname, infection_image)

        blob = bucket.blob(path)
        blob.upload_from_filename(iname, content_type="image/jpeg")

        blob.make_public()
        return blob.public_url

#        private_url = os.path.join("https://storage.cloud.google.com", BUCKET_NAME, path)
#
#        return private_url


def prepare_image(image):

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = transform(image).unsqueeze(0)

    return image


def prepare_timestamp():
    now = datetime.datetime.utcnow()
    return now.strftime(FORMAT)


def insert_row_bigquery_table(UserName, CtScanUrl, MaskUrl):
    query = (
            'INSERT `deployment-317507.covid_ct.single_infection` (UserName, TimeStamp, CtScanUrl, MaskUrl) '
            f'Values("{UserName}", "{prepare_timestamp()}", "{CtScanUrl}", "{MaskUrl}")'
        )
    
    query_job = bigquery_client.query(query)
    return query_job.result()



# ------------------------------------------------------------
# ------------------------------------------------------------
@app.route("/api/ctlunginfection", methods=["POST"])
def predict():

    random_name = prepare_name_based_on_time_seed()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {"success": False, "device": device, "infection_url": None}

    # Fetch Variables from URL
    UserName = flask.request.values.get("UserName")
    if not UserName:
        return flask.jsonify(data)
    # End here

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"]
            image = Image.open(image.stream)

            # TODO: MultiProcessing Here
            original_image_url = save_original_image_to_bucket(random_name, np.array(image))
            # END here

            image = prepare_image(image)

            with torch.no_grad():
                image = image.to(device)

                lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)
                res = lateral_map_2
# res = F.upsample(res, size=(ori_size[1],ori_size[0]), mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            # TODO: Multiprocessing Here
            infection_segmentation_url = save_infection_image_to_bucket(random_name, res)
            # End here

            # TODO: Multiprocessing Here
            insert_row_bigquery_table(UserName, original_image_url, infection_segmentation_url)
            # End Here

            data["infection_url"] = infection_segmentation_url
            data["success"] = True


    return flask.jsonify(data)


#load_bucket()
#import numpy as np
#save_infection_image_to_bucket(prepare_name_based_on_time_seed(), np.random.randn(32, 32))
if __name__ == "__main__":
    print("Loading model and Flask starting server...")
    print("Please wait until the server has fully started")

    load_model()
    load_bucket()
    load_bigquery()

    app.run(host="0.0.0.0")
