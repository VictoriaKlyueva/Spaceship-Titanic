import logging
import argparse
from catboost import CatBoostClassifier
import pandas as pd
import joblib
from spaceship_titanic.config import config_reader


class MyClassifierModel:
    def __init__(self):
        self.target = ['Transported']
        self.hyperparameters = config_reader.load_hyperparameters_from_poetry()
        self.model = CatBoostClassifier(**self.hyperparameters)

    def train(self, dataset_path):
        logging.info(f"Loading dataset from {dataset_path}")
        data = pd.read_csv(dataset_path)
        X = data.drop(self.target, axis=1)
        y = data[self.target]

        logging.info("Training model...")
        self.model.fit(X, y)

        logging.info("Saving model to ./data/model/")
        joblib.dump(self.model, './data/model/model.pkl')

    def predict(self, dataset_path):
        logging.info(f"Loading dataset from {dataset_path}")
        data = pd.read_csv(dataset_path)

        logging.info("Loading model from ./data/model/")
        self.model = joblib.load('./data/model/model.pkl')

        logging.info("Making predictions...")
        predictions = self.model.predict(data)
        pd.DataFrame(predictions, columns=['predictions']).to_csv('./data/results.csv', index=False)
        logging.info("Predictions saved to ./data/results.csv")


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
