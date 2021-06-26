class Config():
    def __init__(self):

        # Image Processing
        self.test_size = 352

        # Model Specifications
        self.input_channels = 6
        self.num_classes = 3

        # Weights
        self.inf_net_unet = "https://github.com/saahiluppal/Inf-Net/releases/download/v1.0/unet_model_200.pkl"

        # Google Service Account API JSON
        self.service_account_json = "deployment.json"

        # GC Bucket
        self.bucket_name = "nosensitivebucket"
        self.original_images_folder = "images/original"
        self.infection_images_folder = "images/infection"
        self.ground_glass_images_folder = "images/ground_glass"
        self.consolidation_images_folder = "images/consolidation"

        # Big Query
        self.project_name = "deployment-317507"
        self.database_name = "covid_ct"
        self.table = "single_infection"
