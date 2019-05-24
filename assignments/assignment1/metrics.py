def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    total_population = ground_truth.shape[0]
    condition_positive = ground_truth.sum()
    condition_negative = total_population - condition_positive
    predicted_positive = prediction.sum()

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in range(prediction.shape[0]):
        if prediction[i]:
            if ground_truth[i]:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if ground_truth[i]:
                false_negative += 1
            else:
                true_negative += 1

    accuracy = (true_positive + true_negative) / total_population
    precision = true_positive / predicted_positive
    recall = true_positive / condition_positive
    f1 = (2 * precision * recall) / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return 0
