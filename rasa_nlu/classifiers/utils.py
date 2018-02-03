def transform_labels_str2num(le, labels):
    # type: (List[Text]) -> np.ndarray
    """Transforms a list of strings into numeric label representation.

    :param labels: List of labels to convert to numeric representation"""

    return le.fit_transform(labels)


def transform_labels_num2str(le, y):
    # type: (np.ndarray) -> np.ndarray
    """Transforms a list of strings into numeric label representation.

    :param y: List of labels to convert to numeric representation"""

    return le.inverse_transform(y)