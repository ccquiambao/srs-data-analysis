import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import sklearn.pipeline
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from sklearn.pipeline import TransformerMixin
from datetime import datetime
from sklearn.preprocessing import RobustScaler

class MeanGradeEncoder(TransformerMixin):
    def fit(self, X, y=None):
        self.user_training_mean_values = X.groupby('user_id')['grade'].agg('mean').to_dict()
        self.global_mean_grade = X['grade'].mean()

        return self

    def transform(self, X, y=None):
        X.set_index('user_id',inplace=True)
        X['past_grade'].fillna(self.user_training_mean_values,inplace=True)
        X['past_grade'].fillna(self.global_mean_grade,inplace=True)
        X.reset_index(inplace=True)

        return X

class RobustEncoder(TransformerMixin):
    def fit(self, X, y=None):
        self.scaler = RobustScaler()
        self.scaler.fit(X[['thinking_time', 'actual_interval']])

        return self

    def transform(self, X, y=None):
        X[ ['thinking_time', 'actual_interval'] ] = self.scaler.transform(X[['thinking_time', 'actual_interval']])

        return X

class CleanEncoderDropUser(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(['user_id'], axis=1).as_matrix()


if __name__ == '__main__':

    train_csv = 'no_outliers_full.csv'
    X_cols = ['grade', 'easiness', 'acq_reps', 'ret_reps', 'lapses',
              'actual_interval', 'thinking_time', 'times_seen', 'past_grade',
              'acq_reps_since_lapse', 'ret_reps_since_lapse']
    X = pd.read_csv(train_csv, usecols=X_cols)

    mge = MeanGradeEncoder()
    re = RobustEncoder()
    ce = CleanEncoderDropUser()
    kmeans = KMeans(n_clusters=6, random_state=42, n_jobs=-2)

    pipe = sklearn.pipeline.Pipeline([('mean_grade', mge), ('robust', re), ('clean', ce) ('kmeans', kmeans) ])
    pipe.fit_transform(X)

    with open('kmeans.txt', 'w') as f:
        for col in X.columns:
            f.write(col)
        for center in kmeans.cluster_centers_:
            f.write(center)
        for label in kmeans.labels_:
            f.write(label)

    joblib.dump(kmeans, 'kmeans.pkl')
