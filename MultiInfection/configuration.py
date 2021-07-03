class Config():
    def __init__(self):

        # Image Processing
        self.test_size = 352

        # Model Specifications
        self.input_channels = 6
        self.num_classes = 3

        # Weights
        self.inf_net_unet = "https://storage.googleapis.com/quadcure_api_bucket/weights/unet_model_200.pkl"

        # Google Service Account API JSON
        self.service_account_json = "QuadCureBot.json"

        # GC Bucket
        self.bucket_name = "quadcure_api_bucket"
        self.original_images_folder = "covid_ctscan/original"
        self.infection_images_folder = "covid_ctscan/infection"
        self.ground_glass_images_folder = "covid_ctscan/ground_glass"
        self.consolidation_images_folder = "covid_ctscan/consolidation"

        # Big Query
        self.project_name = "trusty-vim-316404"
        self.database_name = "quadcure_api_query"
        self.table = "covid_ctscan"
