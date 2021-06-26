class Config:
    def __init__(self):

        # Image Preprocessing
        self.test_size = 352

        # Weights
        self.res2net50_v1b_26w_4s = "https://github.com/saahiluppal/Inf-Net/releases/download/v1.0/res2net50_v1b_26w_4s-3cf99910.pth"
        self.inf_net = "https://github.com/saahiluppal/Inf-Net/releases/download/v1.0/Semi-Inf-Net-100.pth"

        # Google Service Account API JSON
        self.service_account_json = "deployment.json"

        # GC Bucket
        self.bucket_name = "nosensitivebucket"
        self.original_images_folder = "images/original"
        self.infection_images_folder = "images/infection"

        # Big Query
        self.project_name = "deployment-317507"
        self.database_name = "covid_ct"
        self.table = "single_infection"
