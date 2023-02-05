from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import matthews_corrcoef


class Logit_binary(LogisticRegression):
    def __init__(
        self,
        HR_index=None,
        penalty="l2",
        *,
        dual=False,
        tol=0.0001,
        C=1,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None
    ):
        self.index = HR_index
        self.originalclass = []
        self.predictedclass = []
        super().__init__(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )
        if not (isinstance(HR_index, int)):
            raise KeyError(
                "To use this class, you have to give the index where your HR features are (integer)!"
            )

    def get_params(self, deep=False):
        return {"HR_index": self.index}

    def __TransformData(self, X, y=None):
        HR_metrics = X[:, self.index]
        if ((HR_metrics == 0) | (HR_metrics == 1)).all():
            if y is None:
                cond = HR_metrics == 0
                index_arr_noHR = (X[:, self.index] == 0).nonzero()
                index_arr_HR = (X[:, self.index] == 1).nonzero()
                X_HR = X[~cond, :]
                return (
                    X_HR,
                    np.empty(len(index_arr_HR)),
                    index_arr_noHR[0],
                    index_arr_HR[0],
                )
            else:
                cond = HR_metrics == 0
                index_arr_noHR = (X[:, self.index] == 0).nonzero()
                index_arr_HR = (X[:, self.index] == 1).nonzero()
                X_HR = X[~cond, :]
                y_HR = y[~cond]
                return X_HR, y_HR, index_arr_noHR[0], index_arr_HR[0]
        else:
            raise ValueError(
                "The array associated with HR feature must have binary values (0/1)"
            )

    def fit(self, X, y, sample_weight=None):
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "Number of rows of input data matrix different than labels array size"
            )
        X_H, y_H, _, _ = self.__TransformData(X, y=y)
        if X_H.shape[0] != y_H.shape[0]:
            raise ValueError(
                "Number of rows of input data matrix different than labels array size"
            )

        super().fit(X_H, y_H, sample_weight=sample_weight)

    def predict(self, X):
        check_is_fitted(self)
        X_H, _, _, indexes_HR = self.__TransformData(X)
        y_pred = np.zeros(X.shape[0])
        y_pred[indexes_HR] = super().predict(X_H)
        return y_pred

    def predict_proba(self, X):
        check_is_fitted(self)
        X_H, _, indexes_noRH, indexes_HR = self.__TransformData(X)
        y_pred = np.zeros([X.shape[0], 2])
        if X.shape[0] == 1:
            if X[0, self.index] == 1:
                y_pred[:, :] = super().predict_proba(X)
            else:
                y_pred[0, 1] = 1
        else:
            y_pred[indexes_HR, :] = super().predict_proba(X_H)
            y_pred[indexes_noRH, 1] = 1

        return y_pred
