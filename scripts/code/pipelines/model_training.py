import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from code.logging import logger  # Importing logger module for logging
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


class ModelTraining():
    def __init__(self, config: dict) -> None:
        self.BernoulliNB = BernoulliNB(alpha = 2)
        self.SVCmodel = LinearSVC()
        self.LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)

    def train(self, model, X_train, X_test, y_train, y_test, model_name):
        model.fit(X_train, y_train)
        report = self.model_Evaluate(model, X_test, y_test)
        with open(f'../data/{model_name}_report.txt', 'w') as f:
            f.write(report)
    
    def save_model(self, model, name):
        file = open(f'../models/{name}.pickle','wb')
        pickle.dump(model, file)
        file.close()
        logger.info(f"Model saved to '../models/{name}.pickle'")

    def model_Evaluate(self, model, X_test, y_test):
        logger.info("Predict values for Test dataset")
        y_pred = model.predict(X_test)

        logger.info("Compute and plot the Confusion matrix")
        cf_matrix = confusion_matrix(y_test, y_pred)

        categories  = ['Negative','Positive']
        group_names = ['True Neg','False Pos', 'False Neg','True Pos']
        group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

        labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)

        sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                    xticklabels = categories, yticklabels = categories)

        plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
        plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
        plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)

        return classification_report(y_test, y_pred)

    def run(self, X_train, X_test, y_train, y_test) -> None:
        logger.info("Training BernoulliNB Model")
        self.train(self.BernoulliNB, X_train, X_test, y_train, y_test, 'BernoulliNB')
        self.save_model(self.BernoulliNB, 'BernoulliNB')

        logger.info("Training LinearSVC Model")
        self.train(self.SVCmodel, X_train, X_test, y_train, y_test, 'LinearSVC')
        self.save_model(self.SVCmodel, 'LinearSVC') 

        logger.info("Training Logistic Regression Model")
        self.train(self.LRmodel, X_train, X_test, y_train, y_test, 'LogisticRegression')
        self.save_model(self.LRmodel, 'LogisticRegression')