from sklearn.feature_selection import SelectKBest, f_classif
from numpy import ndarray


class FeatureSelector:
    """
    A class for feature selection using the SelectKBest method from scikit-learn.

    This class is designed to select the top K features based on the ANOVA F-value between
    label/feature for classification tasks.

    Attributes:
    - k (int): The number of top features to select.
    - k_best (SelectKBest): An instance of SelectKBest from scikit-learn.
    """

    def __init__(self, k: int) -> None:
        """
        Initializes the FeatureSelector with the specified number of features to select.

        Parameters:
        - k (int): The number of top features to select.

        Returns:
        - None
        """
        self.k = k
        self.k_best = SelectKBest(score_func=f_classif, k=self.k)

    def train(self, X: ndarray, y: ndarray) -> None:
        """
        Fits the SelectKBest model to the data.

        This method computes the ANOVA F-value between each feature and the target for
        the provided dataset, determining the k best features as specified during the
        initialization.

        Parameters:
        - X (ndarray): The feature dataset. A numpy array of shape (n_samples, n_features).
        - y (ndarray): The target values. A numpy array of shape (n_samples,).

        Returns:
        - None
        """
        self.k_best.fit(X, y)

    def transform(self, X: ndarray) -> ndarray:
        """
        Transforms the dataset to select only the k best features.

        This method reduces the input dataset to the set of selected features determined
        during the fitting process. It should be called after the `train` method.

        Parameters:
        - X (ndarray): The feature dataset to transform. A numpy array of shape (n_samples, n_features).

        Returns:
        - X_new (ndarray): The transformed dataset with only the selected features. A numpy array of shape (n_samples, k).
        """
        return self.k_best.transform(X)
