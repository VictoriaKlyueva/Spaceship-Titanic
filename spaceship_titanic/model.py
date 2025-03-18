import argparse
from catboost import CatBoostClassifier
import pandas as pd
import joblib

from spaceship_titanic.utils.data_transformation import process_features
from spaceship_titanic.config import config_reader
from spaceship_titanic.utils.logger import logger


# Class for working with model
class MyClassifierModel:
    def __init__(self):
        self.target = ['Transported']
        self.hyperparameters = config_reader.load_hyperparameters_from_poetry()
        self.model = CatBoostClassifier(**self.hyperparameters)

    def train(self, dataset_path):
        """
            Train model on dataset from dataset_path

            :param dataset_path: path to train dataset
        """

        # Load data
        logger.info(f"Loading dataset from {dataset_path}")
        try:
            data = pd.read_csv(dataset_path)
            data = process_features(data)
        except Exception as e:
            logger.error(f"An error occurred while loading the dataset: {e}")
            return

        X = data.drop(self.target, axis=1)
        y = data[self.target]

        # Train model
        logger.info("Training model...")
        self.model.fit(X, y)

        # Save model
        logger.info("Saving model")
        model_path = "./data/model/model.pkl"
        try:
            joblib.dump(self.model, model_path)
            logger.info("Model saved successfully.")
        except Exception as e:
            logger.error(f"Error occurred while saving: {e}")
            return

    def predict(self, dataset_path):
        """
            Test model on dataset from dataset_path

            :param dataset_path: path to test dataset
        """

        # Load data
        try:
            logger.info(f"Loading dataset from {dataset_path}")
            data = pd.read_csv(dataset_path)
        except Exception as e:
            logger.error(f"An error occurred while loading the dataset: {e}")
            return

        result = data[['PassengerId']].copy()
        data = process_features(data)

        # Load model
        logger.info("Loading model")
        try:
            self.model = joblib.load('./data/model/model.pkl')
        except Exception as e:
            logger.error(f"An error occurred while loading the model: {e}")
            return

        # Save prediction
        logger.info("Making predictions...")
        try:
            predictions = self.model.predict(data)
        except Exception as e:
            logger.error(f"An error occurred while making prediction: {e}")
            return

        result['Transported'] = predictions

        result.to_csv('data/results.csv', index=False)
        logger.info("Predictions saved")


if __name__ == '__main__':
    try:
        # Init argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument('command', choices=['train', 'predict'])
        parser.add_argument('--dataset', required=False)

        args = parser.parse_args()

        # Set default paths
        if args.command == 'train' and args.dataset is None:
            args.dataset = "./data/train.csv"
        elif args.command == 'predict' and args.dataset is None:
            args.dataset = "./data/test.csv"
    except Exception as e:
        logger.error(f"An error occurred while parsing command: {e}")
        raise

    # Start model
    model = MyClassifierModel()
    if args.command == 'train':
        model.train(args.dataset)
    elif args.command == 'predict':
        model.predict(args.dataset)
