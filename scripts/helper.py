import numpy as np
import pandas as pd


def balanced_sample(df):
	pos = df[df.solved_status==1]
	neg = df[df.solved_status==0]

	pos = pos[:len(neg)]

	df = pd.concat([pos, neg], axis=0)

	return df.iloc[np.random.permutation(len(df))]
