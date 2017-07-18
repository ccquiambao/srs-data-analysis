import pandas as pd
import sqlite3
import numpy as np
import scipy.stats
import itertools
import sys

if len(sys.argv) !=3:
    print 'enter input db and output csv'
    quit()

input_db = sys.argv[1]
conn = sqlite3.connect(input_db)
query = '''SELECT * FROM log WHERE event == 9'''

df = pd.read_sql(query, conn, parse_dates=['timestamp', 'next_rep'])

df['dow'] = df['timestamp'].dt.dayofweek
df['times_seen'] = df['acq_reps'] + df['ret_reps']

df['past_grade'] = df.groupby(['user_id','object_id'])['grade'].shift(1)
df['past_grade'] = df['past_grade'].fillna(df.groupby(['user_id'])['grade'].
                                              transform('mean'))

df['past_timestamp'] = df.groupby(['user_id', 'object_id'])['timestamp'].shift(1)
mask = (df['times_seen'] > 1) & (df['actual_interval'] == 0)
df.loc[mask, 'actual_interval'] = (df.loc[mask, 'timestamp'] -
                                        df.loc[mask, 'past_timestamp']).dt.total_seconds()

mask = (df['thinking_time'] >= 0) | (df['actual_interval'] >= 0)
df = df.loc[mask, :]

mask = df['thinking_time'] < 3600
df = df.loc[mask, :]

think_mean = df['thinking_time'].mean()
think_std = df['thinking_time'].std()
act_int_mean = df['actual_interval'].mean()
act_int_std = df['actual_interval'].std()

mask = ( (df['thinking_time'] < (think_mean + 3 * think_std) ) &
         (df['actual_interval'] < (act_int_mean + 3 * act_int_std)) )
df = df.loc[mask, :]

output_csv = sys.argv[2]
with open(output.csv, 'w') as f:
    df.to_csv(f)
