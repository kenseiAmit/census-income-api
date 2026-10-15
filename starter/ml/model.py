import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from starter.ml.data import process_data


def train_model(X_train, y_train, **kwargs):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    params = {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 10,
        "random_state": 42,
    }
    params.update(kwargs)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def save_model(model, path):
    """ Save model or preprocessing object to disk.

    Inputs
    ------
    model : object
        Trained model or preprocessing object.
    path : str or Path
        Destination path.
    """
    joblib.dump(model, path)


def load_model(path):
    """ Load model or preprocessing object from disk.

    Inputs
    ------
    path : str or Path
        Source path.
    Returns
    -------
    model : object
        Loaded model or preprocessing object.
    """
    return joblib.load(path)


def compute_slice_metrics(
    df, feature, model, encoder, lb, cat_features, label="salary"
):
    """ Computes performance metrics on slices of data for a given feature.

    Inputs
    ------
    df : pd.DataFrame
        Dataframe containing the feature and label.
    feature : str
        Name of the feature to slice on.
    model : object
        Trained model.
    encoder : OneHotEncoder
        Trained encoder.
    lb : LabelBinarizer
        Trained label binarizer.
    cat_features : list[str]
        Categorical features list.
    label : str
        Target label column.
    Returns
    -------
    slice_metrics : list[dict]
        List of metric dictionaries for each unique value in feature.
    """
    slice_metrics = []
    for val in sorted(df[feature].dropna().unique()):
        slice_df = df[df[feature] == val]
        if len(slice_df) == 0:
            continue
        X_slice, y_slice, _, _ = process_data(
            slice_df,
            categorical_features=cat_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        slice_metrics.append({
            "feature": feature,
            "value": str(val),
            "count": len(slice_df),
            "precision": float(precision),
            "recall": float(recall),
            "fbeta": float(fbeta),
        })
    return slice_metrics
