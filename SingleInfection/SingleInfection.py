import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model import Inf_Net
import io

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
from utils import assert_DOB


# Custom Configuration
# Optimized for LungInfection.py
from configuration import Config
config = Config()


# Initializing the Flask API
app = FastAPI(title="Covid CT Segmentation")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    max_age=3600,
)




# Globals
model = None
bucket = None
bigquery_client = None



# Initial Loading Services
#-------------------------
# Loading GC Buckets
@app.on_event("startup")
def load_bucket():
    global bucket
    storage_client = storage.Client.from_service_account_json(
                    json_credentials_path=config.service_account_json
                )
    bucket = storage_client.get_bucket(config.bucket_name)

# Loading GBigQuery
@app.on_event("startup")
def load_bigquery():
    global bigquery_client
    bigquery_client = bigquery.Client.from_service_account_json(
                    json_credentials_path=config.service_account_json
                )

# Loading the Model
@app.on_event("startup")
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
def insert_row_bigquery_table(UserName, DOB, CtScanUrl, InfectionUrl):
    query = (
            f'INSERT `{config.project_name}.{config.database_name}.{config.table}` (UserName, TimeStamp, DOB, CtScanUrl, InfectionUrl) '
            f'Values("{UserName}", "{prepare_timestamp()}", "{DOB}", "{CtScanUrl}", "{InfectionUrl}")'
        )
    
    query_job = bigquery_client.query(query)
    return query_job.result()



# Routing Services
# ----------------
# ----------------
@app.post("/api/ctsingleinfection")
async def predict_ctlunginfection(ctscan: UploadFile = File(...), UserName: str = None, DOB: str = None):

    random_name = prepare_name_based_on_time_seed()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {"success": False, "device": device, "image_url": None, "infection_url": None}

    # Fetch Variables from URL
    if not UserName or not DOB:
        return data
    s, DOB = assert_DOB(DOB)
    if not s:
        return data
    # End here

    # read the image in PIL format
    content = await ctscan.read()
    image = Image.open(io.BytesIO(content))

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
    insert_row_bigquery_table(UserName, DOB, original_image_url, infection_segmentation_url)
    # End Here

    data["image_url"] = original_image_url
    data["infection_url"] = infection_segmentation_url
    data["success"] = True


    return data
