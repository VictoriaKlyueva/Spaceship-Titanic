import argparse
from catboost import CatBoostClassifier
import pandas as pd
import joblib

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

        # Save model
        logger.info("Saving model")
        model_path = "./data/model/model.pkl"
        try:
            joblib.dump(self.model, model_path)
            logger.info("Model saved successfully.")
        except Exception as e:
            logger.info(f"Error occurred during the saving: {e}")
            raise

    def predict(self, dataset_path):
        logger.info(f"Loading dataset from {dataset_path}")
        data = pd.read_csv(dataset_path)

        result = data[['PassengerId']].copy()
        data = process_features(data)

        logger.info("Loading model")
        self.model = joblib.load('./data/model/model.pkl')

        # Save prediction
        logger.info("Making predictions...")
        predictions = self.model.predict(data)

        result['Transported'] = predictions

        result.to_csv('data/results.csv', index=False)
        logger.info("Predictions saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'predict'])
    parser.add_argument('--dataset', required=False)

    args = parser.parse_args()

    # Set default paths
    if args.command == 'train' and args.dataset is None:
        args.dataset = "./data/train.csv"
    elif args.command == 'predict' and args.dataset is None:
        args.dataset = "./data/test.csv"

    model = MyClassifierModel()
    if args.command == 'train':
        model.train(args.dataset)
    elif args.command == 'predict':
        model.predict(args.dataset)
