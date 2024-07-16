import os
import subprocess
import zipfile
from pathlib import Path
from utils.logging import logger

# Import Kaggle API
from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

class DataIngestion:
    def __init__(self):
        self.source_url = "spsayakpaul/arxiv-paper-abstracts"
        self.local_file_path = "arxiv-paper-abstracts.zip"
        self.unzip_path = "./dataset"

        self.kaggle_api = KaggleApi()
        self.kaggle_api.authenticate()  # Authenticate with Kaggle using your API key

    def download_datasets(self):
        """
        Download all datasets using the Kaggle API.
        """
        api_command = f"kaggle datasets download -d {self.source_url} -p . --force"

        # Run the Kaggle API command to download the data
        subprocess.run(api_command, shell=True, check=True)
        logger.info(f"Downloaded data from {self.source_url} into {self.local_file_path}")
        return self.local_file_path

    def extract_datasets(self, local_file_path):
        """
        Extract all downloaded ZIP files into their respective directories.
        """
        # Create the unzip directory if it does not exist
        os.makedirs(self.unzip_path, exist_ok=True)

        # Unzip the file
        with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.unzip_path)
        logger.info(f"Extracted {local_file_path} into {self.unzip_path}")

        # Remove the local ZIP file to save disk space
        try:
            os.remove(local_file_path)
            logger.info(f"Removed local ZIP file: {local_file_path}")
        except Exception as e:
            logger.error(f"Failed to remove local ZIP file: {local_file_path}", exc_info=True)
            raise e