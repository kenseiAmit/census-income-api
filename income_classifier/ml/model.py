import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from income_classifier.ml.data import process_data


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
