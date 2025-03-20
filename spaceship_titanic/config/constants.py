# Data paths
TRAIN_DATA_PATH = "./data/train.csv"
TEST_DATA_PATH = "./data/test.csv"
MODEL_SAVE_PATH = "./data/model/model.pkl"
RESULTS_SAVE_PATH = "./data/results.csv"

# Default params
DEFAULT_HYPERPARAMETERS = {
    'iterations': 500,
    'depth': 6,
    'learning_rate': 0.03,
    'l2_leaf_reg': 3.0,
}
SEED = 42

# Columns
TARGET_COLUMN = "Transported"
CATEGORICAL_FEATURES = [
    'HomePlanet', 'CryoSleep', 'Destination',
    'VIP', 'Deck', 'Side', 'PaidExtra',
    'IsAlone', 'IsTravelWithFamily', 'Gender'
]
NUMERICAL_FEATURES = [
    'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'PassengerGroup',
    'PassengerNumberInGroup', 'Num', 'PaidSum', 'PassengersCountInGroup'
]
