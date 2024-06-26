from code.config import read_config
from code.pipelines.data_gathering import DataGathering
from code.pipelines.data_preparation import DataPreparation
from code.pipelines.model_training import ModelTraining
from code.pipelines.model_inferencing import ModelInferencing
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

config = read_config()

data_gathering = DataGathering(config)
data_preparation = DataPreparation(config)
model_training = ModelTraining(config)
model_inferencing = ModelInferencing(config)


DATA_PATH = r"../data/training.1600000.processed.noemoticon.csv"

text, sentiment = data_gathering.load_data(path = DATA_PATH)
X_train, X_test, y_train, y_test = data_preparation.run(text, sentiment)
model_training.run(X_train, X_test, y_train, y_test)


inference = config['flow']['inference']['flag']
if inference:
    text = [
        "Its a problem everytime",
        "May the Force be with you.",
        "Mr. Stark, I don't feel so good"]
    vectoriser, model = model_inferencing.load_models()
    df = model_inferencing.predict(vectoriser, model, text)
    print(df.head())