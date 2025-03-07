import numpy as np
import pandas as pd
import gender_guesser.detector as gender

from spaceship_titanic.utils.logger import logger


def predict_gender(name):
    """
        Predict gender by name and surname
    """
    gender_detector = gender.Detector()

    first_name = name
    gender_pred = gender_detector.get_gender(first_name)

    if gender_pred in ['male', 'mostly_male']:
        return 'male'
    elif gender_pred in ['female', 'mostly_female']:
        return 'female'
    else:
        return 'unknown'


def replace_nans(df):
    """
        Replace NANs in dataset with some values
    """

    logger.info("Replacing NaNs in the dataset")

    cat_features = [
        'HomePlanet', 'CryoSleep', 'Destination',
        'VIP', 'Deck', 'Side', 'PaidExtra', 'IsAlone',
        'IsTravelWithFamily', 'Gender'
    ]
    num_features = [
        'Age', 'RoomService', 'FoodCourt', 'ShoppingMall',
        'Spa', 'VRDeck', 'PassengerGroup', 'PassengerNumberInGroup',
        'Num', 'PaidSum', 'PassengersCountInGroup'
    ]

    # Fill number features with medians
    for feature in num_features:
      df[feature] = df[feature].astype(float)

    median_values = df[num_features].median()
    df[num_features] = df[num_features].fillna(median_values)

    # Fill imbalanced categorical features with the most frequent value
    disbalanced_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Gender']

    for feature in disbalanced_features:
      df[feature] = df[feature].fillna(df[feature].mode()[0])

    # Fill other categorical features with random values
    balanced_features = [i for i in cat_features if i not in disbalanced_features]

    for feature in balanced_features:
        non_nan_values = df[feature].dropna().unique()
        df[feature] = df[feature].apply(lambda x: np.random.choice(non_nan_values) if pd.isna(x) else x)

    return df


def generate_features(df):
    """
        Generate new features
    """

    logger.info("Generating new features")

    # Total amount paid by the passenger
    df['PaidSum'] = (df['RoomService'] + df['FoodCourt'] +
                     df['ShoppingMall'] + df['Spa'] +
                     df['VRDeck'])

    # Whether the passenger paid for extra services
    df['PaidExtra'] = (df['RoomService'] + df['FoodCourt'] +
                       df['ShoppingMall'] + df['Spa'] +
                       df['VRDeck'] != 0.0)

    # Number of passengers in the group
    df['PassengersCountInGroup'] = df.groupby('PassengerGroup')['PassengerNumberInGroup'].transform('max')

    # Whether the passenger is traveling alone
    df['IsAlone'] = (df['PassengersCountInGroup'] == 1) & (df['PassengerNumberInGroup'] == 1)

    # Whether the passenger is traveling with family
    df['IsTravelWithFamily'] = df.groupby('PassengerGroup')['Surname'].transform(lambda x: (x == x.iloc[0]).sum() > 1)

    # Predict gender based on the name
    df['Gender'] = 'male' # df['Name'].apply(predict_gender)

    return df


def process_features(df):
    """
        Обработка фичей датасета для подачи в модель
    """

    logger.info("Processing dataset features")

    # Split PassengerId into PassengerGroup and PassengerNumberInGroup
    df['PassengerGroup'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['PassengerNumberInGroup'] = df['PassengerId'].apply(lambda x: int(x.split('_')[1]))

    # Split Cabin into Deck, Num, and Side
    df['Cabin'] = df['Cabin'].astype(str)
    df['Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if x != 'nan' else np.nan)
    df['Num'] = df['Cabin'].apply(lambda x: int(x.split('/')[1]) if x != 'nan' else np.nan)
    df['Side'] = df['Cabin'].apply(lambda x: x.split('/')[2] if x != 'nan' else np.nan)

    # Split Name into Surname and Name
    df['Name'] = df['Name'].fillna('Unknown Unknown') # Если имя неизвестно
    df['Surname'] = df['Name'].apply(lambda x: x.split()[1])
    df['Name'] = df['Name'].apply(lambda x: x.split()[0])

    df = generate_features(df)

    # Drop unnecessary columns
    cols2drop = ['PassengerId', 'Cabin', 'Name', 'Surname']
    df = df.drop(cols2drop, axis=1, errors='ignore')

    df = replace_nans(df)

    return df
