import numpy as np


def multiclass_accuracy(prediction: np.array, ground_truth: np.array):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy
    total_samples = prediction.shape[0]
    accurate_predictions = (prediction == ground_truth).sum()  # неактульный варнинг.
    # Результат сравнения двух np.array - это np.array(bool), а не bool
    accuracy = accurate_predictions / total_samples
    return accuracy

