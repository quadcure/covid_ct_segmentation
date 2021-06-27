import torch
from fastapi import FastAPI, File, UploadFile
from model import Inf_Net_UNet
import io

from PIL import Image
from google.cloud import storage
from google.cloud import bigquery
from tempfile import NamedTemporaryFile
import os
import imageio
import numpy as np


# Custom Utility Sciprt
# Used by MultiInfection.py
from utils import prepare_image
from utils import split_class


# Custom Configuration
# Optimized for MultiInfection.py
from configuration import Config
config = Config()


# Initializing the Flask API
app = FastAPI(title="Covid CT Segmentation")


# Globals
model = None
bucket = None
bigquery_client = None



# Initial Loading Services
# ------------------------
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    global model
    model = Inf_Net_UNet(config.input_channels, config.num_classes)

    model_state = torch.hub.load_state_dict_from_url(config.inf_net_unet,
                                    map_location=device, progress=True)
    model.load_state_dict(model_state)

    model.to(device)
    model.eval()
# ------------------------



# Functions using global services
# -------------------------------
# Saving GroundGlass Segmentation to GC Bucket
def save_ground_glass_to_bucket(name, ground_glass_image):
    path = os.path.join(config.ground_glass_images_folder, name) + ".jpeg"

    with NamedTemporaryFile() as temp:
        iname = "".join([str(temp.name),".jpeg"])
        ground_glass_image.save(iname)

        blob = bucket.blob(path)
        blob.upload_from_filename(iname, content_type="image/jpeg")

        blob.make_public()
        return blob.public_url

# Saving Consolidation Segmentation to GC Bucket
def save_consolidation_to_bucket(name, consolidation_image):
    path = os.path.join(config.consolidation_images_folder, name) + ".jpeg"

    with NamedTemporaryFile() as temp:
        iname = "".join([str(temp.name),".jpeg"])
        consolidation_image.save(iname)

        blob = bucket.blob(path)
        blob.upload_from_filename(iname, content_type="image/jpeg")

        blob.make_public()
        return blob.public_url


# Updating Rows in BigQuery
# Adding ground Glass Opacitiy Url
# Along with Consolidation Url
def update_row_bigquery_table(PsuedoUrl, GroundGlassUrl, ConsolidationUrl):
    query = (
            f'UPDATE `{config.project_name}.{config.database_name}.{config.table}` '
            f'SET GroundGlassUrl = "{GroundGlassUrl}", '
            f'ConsolidationUrl = "{ConsolidationUrl}" '
            f'WHERE InfectionUrl = "{PsuedoUrl}"'
        )

    query_job = bigquery_client.query(query)
    return query_job.result()
# ---------------------------------



# Routing Services
# ----------------
# ----------------
@app.post("/api/ctmultiinfection")
async def inference(ImageUrl: str = None, PsuedoUrl: str = None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {"success": False, "device": device, "GroundGlass": None, "Consolidation": None}

    # Fetch Variables from URL
    if not PsuedoUrl or not ImageUrl:
        return data
    # End here

    # Fetching naming convention from PsuedoUrl
    random_name = os.path.basename(PsuedoUrl)
    random_name = os.path.splitext(random_name)[0]

    # Reading Original and Psuedo image
    # using their URL's provided by /api/ctsingleinfection
    image, psuedo_image = prepare_image(ImageUrl, PsuedoUrl)

    with torch.no_grad():
        image = image.to(device)
        psuedo_image = psuedo_image.to(device)

        output = model(torch.cat((image, psuedo_image), dim=1))
        output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
        b, _, w, h = output.size()

        pred = output.cpu().permute(0, 2, 3, 1).contiguous().view(-1, config.num_classes).max(1)[1].view(b, w, h).numpy().squeeze()
        print('Class numbers of prediction in total:', np.unique(pred))


        with NamedTemporaryFile() as temp:
            iname = "".join([str(temp.name),".png"])
            imageio.imwrite(iname, pred)

            class_one, class_two = split_class(iname, w, h)

        
        # TODO: Multiprocessing here
        ground_glass_url = save_ground_glass_to_bucket(random_name, class_one)
        consolidation_url = save_consolidation_to_bucket(random_name, class_two)

        update_row_bigquery_table(PsuedoUrl, ground_glass_url, consolidation_url)
        
        data["GroundGlass"] = ground_glass_url
        data["Consolidation"] = consolidation_url
        # End here

        return data
