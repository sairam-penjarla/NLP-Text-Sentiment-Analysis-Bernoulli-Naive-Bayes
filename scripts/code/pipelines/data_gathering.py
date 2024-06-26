import pandas as pd
from code.logging import logger  # Importing logger module for logging
from code.utilities.common_utils import CommonUtils

class DataGathering:
    def __init__(self, config: dict) -> None:
        self.common_utils = CommonUtils()
        self.config = config
        self.DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "text"]
        self.DATASET_ENCODING = "ISO-8859-1"

    def load_data(self, path):
        logger.info(f"Reading data from path: {path}")
        dataset = pd.read_csv(path, encoding=self.DATASET_ENCODING, names=self.DATASET_COLUMNS)
        logger.info("finished reading the dataset")

        logger.info("Removing the unnecessary columns.")
        dataset = dataset[['sentiment', 'text']]
        logger.info("Replacing the values to ease understanding.")
        dataset['sentiment'] = dataset['sentiment'].replace(4, 1)

        logger.info("Storing data in lists.")
        text, sentiment = list(dataset['text']), list(dataset['sentiment'])
        return text, sentiment