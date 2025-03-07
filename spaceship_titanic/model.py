import argparse
from catboost import CatBoostClassifier
import pandas as pd
import joblib
import os

from spaceship_titanic.utils.data_transformation import process_features
from spaceship_titanic.config import config_reader
from spaceship_titanic.utils.logger import logger


class MyClassifierModel:
    def __init__(self):
        self.target = ['Transported']
        self.hyperparameters = config_reader.load_hyperparameters_from_poetry()
        self.model = CatBoostClassifier(**self.hyperparameters)

    def train(self, dataset_path):
        logger.info(f"Loading dataset from {dataset_path}")
        data = pd.read_csv(dataset_path)
        data = process_features(data)

        X = data.drop(self.target, axis=1)
        y = data[self.target]

        logger.info("Training model...")
        self.model.fit(X, y)

        logger.info("Saving model to spaceship_titanic/data/model/")
        model_path = "spaceship_titanic/data/model/model.pkl"
        joblib.dump(self.model, model_path)
        logger.info("Model saved successfully.")

    def predict(self, dataset_path):
        logger.info(f"Loading dataset from {dataset_path}")
        data = pd.read_csv(dataset_path)
        data = process_features(data)

        logger.info("Loading model from spaceship_titanic/data/model/")
        self.model = joblib.load('spaceship_titanic/data/model/model.pkl')

        logger.info("Making predictions...")
        predictions = self.model.predict(data)
        pd.DataFrame(predictions, columns=['predictions']).to_csv('spaceship_titanic/data/results.csv', index=False)
        logger.info("Predictions saved to spaceship_titanic/data/results.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'predict'])
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    model = MyClassifierModel()
    if args.command == 'train':
        model.train(args.dataset)
    elif args.command == 'predict':
        model.predict(args.dataset)
