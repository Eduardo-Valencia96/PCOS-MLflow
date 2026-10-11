import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(df):
    df_clean = df.copy()

    missing_values = ['?', 'NA', 'N/A', 'na', 'n/a', '', ' ']

    df_clean = df_clean.replace(
        missing_values,
        np.nan
    )

    df_clean['II    beta-HCG(mIU/mL)'] = pd.to_numeric(
        df_clean['II    beta-HCG(mIU/mL)'],
        errors='coerce'
    )

    df_clean['AMH(ng/mL)'] = pd.to_numeric(
        df_clean['AMH(ng/mL)'],
        errors='coerce'
    )

    target = 'PCOS (Y/N)'
    features_to_drop = ['Patient File No.', 'Unnamed: 44']

    X = df_clean.drop(columns=[target] + features_to_drop)
    y = df_clean[target]

    X['Cycle(R/I)'] = X['Cycle(R/I)'].map({
        2: 0,
        4: 1
    })

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor(X):
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return preprocessor


def preprocess_data(X_train, X_test):
    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, preprocessor