import pickle
import pandas as pd
from code.logging import logger  # Importing logger module for logging
from code.utilities.common_utils import CommonUtils

class ModelInferencing():
    def __init__(self, config: dict) -> None:
        self.utils = CommonUtils()
        self.config = config

    def load_models(self):
        logger.info("Load the vectoriser.")
        file = open('../models/vectoriser-ngram-(1,2).pickle', 'rb')
        vectoriser = pickle.load(file)
        file.close()
        logger.info("Load the LR Model.")
        file = open('../models/LinearSVC.pickle', 'rb')
        LRmodel = pickle.load(file)
        file.close()
        
        return vectoriser, LRmodel

    def predict(self, vectoriser, model, text):
        logger.info("Predict the sentiment")
        text = self.utils.preprocess(text, self.config['text_processing_params'])
        textdata = vectoriser.transform(text)
        sentiment = model.predict(textdata)
        
        logger.info("Make a list of text with sentiment.")
        data = []
        for text, pred in zip(text, sentiment):
            data.append((text,pred))
            
        logger.info("Convert the list into a Pandas DataFrame.")
        df = pd.DataFrame(data, columns = ['text','sentiment'])
        df = df.replace([0,1], ["Negative","Positive"])
        return df