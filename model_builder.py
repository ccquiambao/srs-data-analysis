import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.pipeline
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from sklearn.pipeline import TransformerMixin
from sklearn.preprocessing import RobustScaler
import sys

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
        X['past_grade_rounded'] =  X['past_grade'].round()


        return X

class RobustEncoder(TransformerMixin):
    def fit(self, X, y=None):
        self.scaler = RobustScaler()
        self.scaler.fit(X[['thinking_time', 'actual_interval']])

        return self

    def transform(self, X, y=None):
        X[ ['thinking_time', 'actual_interval'] ] = self.scaler.transform(X[['thinking_time', 'actual_interval']])

        return X

class DummyEncoder(TransformerMixin):
    def fit(self, X, y=None):
        self.index_ = X.index
        self.columns_ = X.columns
        self.cat_columns_ = ['past_grade_rounded']

        for col in self.cat_columns_:
            X.loc[:, col] = X.loc[:, col].astype('category')

        self.cat_dict_ = {col:X[col].cat.categories for col in self.cat_columns_}


        return self

    def transform(self, X, y=None):
        for col in self.cat_columns_:
            if X[col].dtype.name != 'category':
                X.loc[:, col] = X.loc[:, col].astype('category', categories=self.cat_dict_[col])

        return pd.get_dummies(X, columns=['past_grade_rounded'])


class CleanEncoderDropUser(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(['user_id', 'grade'], axis=1).as_matrix()

def split_users(df, train_size=0.8):
    all_users = np.asarray(df.groupby('user_id').groups.keys())
    num_users = len(all_users)
    train_user_size = int(round(num_users * train_size))
    train_users = np.random.choice(all_users, train_user_size, replace=False)

    train_df = df[df['user_id'].isin(train_users)]
    test_df = df[~df['user_id'].isin(train_users)]

    return train_df, test_df

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'enter data input, file output, and model'
        quit()

    train_csv = sys.argv[1]
    X_cols = ['user_id', 'grade', 'easiness', 'acq_reps', 'ret_reps', 'lapses',
              'actual_interval','thinking_time', 'times_seen', 'past_grade',
              'acq_reps_since_lapse', 'ret_reps_since_lapse']
    X = pd.read_csv(train_csv, usecols=X_cols)
    y = X['grade'].copy()

    mge = MeanGradeEncoder()
    de = DummyEncoder()
    re = RobustEncoder()
    ce = CleanEncoderDropUser()

    logit = LogisticRegression(C=10, n_jobs=-1)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1)

    if sys.argv[4] == 'logit':
        model = logit
    elif sys.argv[4] == 'rf':
        model = rf
    else:
        print "enter 'logit' or 'rf'"
        quit()
        
    kf = KFold(n_splits=5, random_state=42)
    pipe = sklearn.pipeline.Pipeline([('mean_grade', mge), ('dummies', de),
                                      ('robust', re), ('cleaning', ce),
                                      ('model', model) ])

    scores = []
    all_users = df.groupby('user_id').groups.keys()

    for train_users, test_users in kf.split(all_users):
        X_train = df[df['user_id'].isin(train_users)]
        y_train = X_train.loc[:, 'grade']

        X_test = df[~df['user_id'].isin(test_users)]
        y_test = X_test.loc[:, 'grade']

        pipe.fit(X_train, y_train)
        score = pipe.score(X_test,  y_test)
        scores.append(score)

    with open('{fn}.txt'.format(fn=sys.argv[2]), 'w') as f:
        for score in scores:
            f.write(str(score) + '\n')

    joblib.dump(logit, '{fn}.pkl'.format(fn=sys.argv[2]))
