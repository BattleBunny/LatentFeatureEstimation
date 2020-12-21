# BSD 3-Clause License

# Copyright(c) 2020, janvanrijn
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#         SERVICES
#         LOSS OF USE, DATA, OR PROFITS
#         OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import pandas as pd
import numpy as np
import typing
from sklearn.base import clone
from statistics import *
from scipy.linalg import svd


class AbstractMetaLearningModel(object):

    def fit(self, df_features: pd.DataFrame, df_performance: pd.DataFrame) -> None:
        """
        Takes an input (meta-features) and trains the internal model on it.

        :param df_features: pd.DataFrame
        a data frame with size (N, M), where all N rows represent a base-dataset, and all M columns represent a
        meta-feature, calculated over that specific dataset

        :param df_performance: pd.DataFrame
        a data frame with size (N, D), where all N rows represent a base-dataset, and all D columns represent the
        performance of a (base-)model on that specific dataset (predictive accuracy)
        """
        raise NotImplementedError("Abstract Method, please subclass")

    def predict(self, df_features: pd.DataFrame) -> typing.List[str]:
        """
        Predicts for a set of datasets (expressed in meta-features) the performance per (base-)model

        :param df_features: pd.DataFrame
        a data frame with size (N, M), where all N rows represent a base-dataset, and all M columns represent a
        meta-feature, calculated over that specific dataset

        :return: List[str]
        A list of length N, where each item in the list represents the name of the (base-)model that is predicted to
        perform best
        """
        raise NotImplementedError("Abstract Method, please subclass")

    def score(self, y: pd.DataFrame, y_hat: typing.List[str]) -> typing.List[str]:
        """
        Scores the how well the by the meta-model selected classifiers would have performed.

        :param y: pd.DataFrame
        a data frame with size (N, D), where all N rows represent a base-dataset, and all D columns represent the
        performance of a (base-)model on that specific dataset (predictive accuracy)

        :param y_hat: pd.DataFrame
        A list of length N, where each item in the list represents the name of the (base-)model that is predicted to
        perform best

        :return: List[float]
        The performance of the by the meta-model selected classifiers (indicated in y_hat) per task
        """
        result = []
        for idx, classifier in enumerate(y_hat):
            result.append(y[classifier][idx])
        return result


class BSS(AbstractMetaLearningModel):

    def __init__(self, _):
        """
        Baseline method, that determines during fit time which method performs best on average. This method is selected
        always at test time.
        """
        self.avg_performances = None
        self.best_model_name = None

    def fit(self, df_features: pd.DataFrame, df_performance: pd.DataFrame) -> None:
        """
        Takes an input (meta-features) and trains the internal model on it.

        :param df_features: pd.DataFrame
        a data frame with size (N, M), where all N rows represent a base-dataset, and all M columns represent a
        meta-feature, calculated over that specific dataset

        :param df_performance: pd.DataFrame
        a data frame with size (N, D), where all N rows represent a base-dataset, and all D columns represent the
        performance of a (base-)model on that specific dataset (predictive accuracy)
        """
        self.avg_performances = df_performance.mean(axis=0)

    def predict(self, df_features: pd.DataFrame) -> typing.List[str]:
        """
        Predicts for a set of datasets (expressed in meta-features) the performance per (base-)model

        :param df_features: pd.DataFrame
        a data frame with size (N, M), where all N rows represent a base-dataset, and all M columns represent a
        meta-feature, calculated over that specific dataset

        :return: List[str]
        A list of length N, where each item in the list represents the name of the (base-)model that is predicted to
        perform best
        """
        bss = pd.DataFrame(index=df_features.index)
        bss["flow_id"] = self.avg_performances.idxmax()
        return bss["flow_id"]


class MetaRegression(AbstractMetaLearningModel):

    def __init__(self, expected_models, regressor):
        """
        Baseline method, that determines during fit time which method performs best on average. This method is selected
        always at test time.
        """
        self.models = {
            model: clone(regressor) for model in expected_models
        }

    def fit(self, df_features: pd.DataFrame, df_performance: pd.DataFrame) -> None:
        """
        Takes an input (meta-features) and trains the internal model on it.

        :param df_features: pd.DataFrame
        a data frame with size (N, M), where all N rows represent a base-dataset, and all M columns represent a
        meta-feature, calculated over that specific dataset

        :param df_performance: pd.DataFrame
        a data frame with size (N, D), where all N rows represent a base-dataset, and all D columns represent the
        performance of a (base-)model on that specific dataset (predictive accuracy)
        """
        for model in self.models:
            self.models[model].fit(df_features, df_performance[model])

    def predict(self, df_features: pd.DataFrame) -> typing.List[str]:
        """
        Predicts for a set of datasets (expressed in meta-features) the performance per (base-)model

        :param df_features: pd.DataFrame
        a data frame with size (N, M), where all N rows represent a base-dataset, and all M columns represent a
        meta-feature, calculated over that specific dataset

        :return: List[str]
        A list of length N, where each item in the list represents the name of the (base-)model that is predicted to
        perform best
        """
        performance = []
        modelnames = []
        for model in self.models:
            performance.append(
                self.models[model].predict(df_features).tolist())
            modelnames.append(model)
        performance = np.array(performance)
        bestidx = performance.argmax(axis=0)
        df = pd.DataFrame({"modelnames": np.array(modelnames)[
                          bestidx], "datasets": df_features.index}).set_index("datasets")
        return df["modelnames"]

class SVDPortfolio(MetaRegression):
    def __init__(self, expected_models, regressor, estimator):
        """
        Baseline method, that determines during fit time which method performs best on average based on estimated latent features. This method is selected
        always at test time.
        """
        self.models = {
            model: clone(regressor) for model in expected_models
        }
        self.estimator = estimator

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """
        Takes an input (meta-features) and trains the estimator and internal model on it.

        :param X_train: pd.DataFrame
        a data frame with size (N, M), where all N rows represent a base-dataset, and all M columns represent a
        meta-feature, calculated over that specific dataset

        :param y_train: pd.DataFrame
        a data frame with size (N, D), where all N rows represent a base-dataset, and all D columns represent the
        performance of a (base-)model on that specific dataset (predictive accuracy)
        """
        U, s, Vh = svd(y_train)
        df_U = pd.DataFrame(index=y_train.index)
        #m > n dus alleen eerste n cols zijn nodig
        for i in range(y_train.shape[1]):
            df_U[i] = U[:, i]
        self.estimator.fit(X_train,df_U)
        latent_features = self.estimator.predict(X_train)
        for model in self.models:
            self.models[model].fit(latent_features, y_train[model])

    def predict(self, df_features: pd.DataFrame) -> typing.List[str]:
        """
        Predicts for a set of datasets (expressed in meta-features) the performance per (base-)model

        :param df_features: pd.DataFrame
        a data frame with size (N, M), where all N rows represent a base-dataset, and all M columns represent a
        meta-feature, calculated over that specific dataset

        :return: List[str]
        A list of length N, where each item in the list represents the name of the (base-)model that is predicted to
        perform best
        """
        latent_features = self.estimator.predict(df_features)
        performance = []
        modelnames = []
        for model in self.models:
            performance.append(
                self.models[model].predict(latent_features).tolist())
            modelnames.append(model)
        performance = np.array(performance)
        bestidx = performance.argmax(axis=0)
        df = pd.DataFrame({"modelnames": np.array(modelnames)[
                          bestidx], "datasets": df_features.index}).set_index("datasets")
        return df["modelnames"]
