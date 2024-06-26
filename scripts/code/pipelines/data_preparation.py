import time
import pickle
from code.logging import logger 
from code.utilities.common_utils import CommonUtils
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class DataPreparation:
    def __init__(self, config: dict) -> None:
        self.utils = CommonUtils()
        self.config = config

    def run(self, text, sentiment):
        t = time.time()
        processedtext = self.utils.preprocess(text, self.config['text_processing_params'])

        logger.info(f'Text Preprocessing complete.')
        logger.info(f'Time Taken: {round(time.time()-t)} seconds')

        logger.info("Splitting the data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(processedtext,
                                                            sentiment,
                                                            test_size = 0.05, random_state = 0)
        
        vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
        vectoriser.fit(X_train)
        logger.info(f'Vectoriser fitted on X_train')
        logger.info(f'No. of feature_words: {len(vectoriser.get_feature_names_out())}')
        
        logger.info("Saving the vectoriser")
        file = open('../models/vectoriser-ngram-(1,2).pickle','wb')
        pickle.dump(vectoriser, file)
        file.close()

        X_train = vectoriser.transform(X_train)
        X_test  = vectoriser.transform(X_test)
        logger.info(f'Data Transformed.')

        return X_train, X_test, y_train, y_test