from src import logger
from GetData.download import DataIngestion

data = DataIngestion()
file_path = data.download_datasets()

extract = data.extract_datasets(file_path)