import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Load Dataset
def load_data():

    df = pd.read_csv(
        r"Data/global air pollution dataset.csv"
    )

    return df


# Preprocess
def preprocess_data(df):

    # Missing Values
    df.fillna(
        df.mode().iloc[0],
        inplace=True
    )

    # Create models folder
    os.makedirs(
        "models",
        exist_ok=True
    )

    # Label Encoding
    encoders = {}

    cat_cols = df.select_dtypes(
        include='object'
    ).columns

    for col in cat_cols:

        le = LabelEncoder()

        df[col] = le.fit_transform(
            df[col].astype(str)
        )

        encoders[col] = le

    # Save encoders
    joblib.dump(
        encoders,
        "models/label_encoders.pkl"
    )

    # Features and Target
    X = df.drop(
        [
            'AQI Value',
            'AQI Category'
        ],
        axis=1
    )

    y = df['AQI Value']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X.columns
    )