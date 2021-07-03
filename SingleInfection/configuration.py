class Config:
    def __init__(self):

        # Image Preprocessing
        self.test_size = 352

        # Weights
        self.res2net50_v1b_26w_4s = "https://storage.googleapis.com/quadcure_api_bucket/weights/res2net50_v1b_26w_4s-3cf99910.pth"
        self.inf_net = "https://storage.googleapis.com/quadcure_api_bucket/weights/Semi-Inf-Net-100.pth"

        # Google Service Account API JSON
        self.service_account_json = "QuadCureBot.json"

        # GC Bucket
        self.bucket_name = "quadcure_api_bucket"
        self.original_images_folder = "covid_ctscan/original"
        self.infection_images_folder = "covid_ctscan/infection"

        # Big Query
        self.project_name = "trusty-vim-316404"
        self.database_name = "quadcure_api_query"
        self.table = "covid_ctscan"
