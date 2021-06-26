import torch
import flask
from model import Inf_Net

from PIL import Image
from google.cloud import storage
from google.cloud import bigquery
from tempfile import NamedTemporaryFile
import os
import imageio
import numpy as np


# Custom Utility Script
# Used by LungInfection.py
from utils import prepare_image
from utils import prepare_name_based_on_time_seed
from utils import prepare_timestamp


# Custom Configuration
# Optimized for LungInfection.py
from configuration import Config
config = Config()


# Initializing the Flask API
app = flask.Flask(__name__)


# Globals
model = None
bucket = None
bigquery_client = None



# Initial Loading Services
#-------------------------
# Loading GC Buckets
def load_bucket():
    global bucket
    storage_client = storage.Client.from_service_account_json(
                    json_credentials_path=config.service_account_json
                )
    bucket = storage_client.get_bucket(config.bucket_name)

# Loading GBigQuery
def load_bigquery():
    global bigquery_client
    bigquery_client = bigquery.Client.from_service_account_json(
                    json_credentials_path=config.service_account_json
                )

# Loading the Model
def load_model():
    # Initializing model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    global model
    model = Inf_Net()

    # loading weights
    model_state = torch.hub.load_state_dict_from_url(config.inf_net, 
                                    map_location=device, progress=True)
    model.load_state_dict(model_state)

    # loading model on specified device
    # and disabling gradient flow.
    model.to(device)
    model.eval()
# ------------------------



# Functions using global services
# -------------------------------
# Saving original CT-Scan to GC Bucket
def save_original_image_to_bucket(name, original_image):
    path = os.path.join(config.original_images_folder, name) + ".jpeg"

    with NamedTemporaryFile() as temp:
        iname = "".join([str(temp.name),".jpeg"])
        imageio.imwrite(iname, original_image)

        blob = bucket.blob(path)
        blob.upload_from_filename(iname, content_type="image/jpeg")

        blob.make_public()
        return blob.public_url


# Saving Predicted Mask to GC Bucket
def save_infection_image_to_bucket(name, infection_image):
    path = os.path.join(config.infection_images_folder, name) + ".jpeg"

    with NamedTemporaryFile() as temp:
        iname = "".join([str(temp.name),".jpeg"])
        imageio.imwrite(iname, infection_image)

        blob = bucket.blob(path)
        blob.upload_from_filename(iname, content_type="image/jpeg")

        blob.make_public()
        return blob.public_url


# Saving Data to BigQuery
def insert_row_bigquery_table(UserName, CtScanUrl, MaskUrl):
    query = (
            f'INSERT `{config.project_name}.{config.database_name}.{config.table}` (UserName, TimeStamp, CtScanUrl, MaskUrl) '
            f'Values("{UserName}", "{prepare_timestamp()}", "{CtScanUrl}", "{MaskUrl}")'
        )
    
    query_job = bigquery_client.query(query)
    return query_job.result()



# Routing Services
# ----------------
# ----------------
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


if __name__ == "__main__":
    print("Loading model and Flask starting server...")
    print("Please wait until the server has fully started")

    load_model()
    load_bucket()
    load_bigquery()

    app.run(host="0.0.0.0")
