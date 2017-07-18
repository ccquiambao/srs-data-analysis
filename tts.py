import pandas as pd
import numpy as np
import sys

def split_users(df, train_size=0.75):
    all_users = df.groupby('user_id').groups.keys()
    num_users = len(df.groupby('user_id').groups)
    train_user_size = int(round(num_users * train_size))

    train_users = np.random.choice(all_users, train_user_size, replace=False)
    train_df = df[df['user_id'].isin(train_users)]
    test_df = df[~df['user_id'].isin(train_users)]

    return train_df, test_df

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'enter input csv and percent of users'

    input_csv = sys.argv[1]
    X = pd.read_csv(input_csv, parse_dates=['timestamp'])
    y = X['grade'].copy()

    train_size = sys.argv[2] / 100.
    X_train, X_test = split_users(X, train_size=train_size)

    train_file = 'user_{}_train.csv'.format(sys.argv[2])
    with open(train_file,'w') as f:
        X_train.to_csv(f)

    test_file = 'user_{}_test.csv'.format(100 - int(sys.argv[2])
    with open(test_file, 'w') as g:
        X_test.to_csv(g)
