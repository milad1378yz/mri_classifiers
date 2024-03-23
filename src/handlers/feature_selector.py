from sklearn.feature_selection import SelectKBest, f_classif


class FeatureSelector:
    def __init__(self, k):
        self.k = k
        self.k_best = SelectKBest(score_func=f_classif, k=self.k)

    def train(self, X, y):
        self.k_best.fit(X, y)

    def transform(self, X):
        return self.k_best.transform(X)
