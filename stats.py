import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#sns.set(style="whitegrid")
#plt.title("Statistics")

data = pd.DataFrame({'clf' : [] , 'path_finder' : [], 'size' : [], 'average_apple' : [], 'average_score' : [] })
clfs = ['Forest','MLP']

for clf in clfs:
    datapath = "data/supervised_{}.csv".format(clf)
    df = pd.read_csv(datapath, index_col=0)
    print(clf)
    print(df.groupby(by='path_finder').mean()[['score','apple_score']])
    print("\n")

#ax = sns.barplot(x="clf", y="averagescore", hue="path_finder",col="path_finder", order=["Forest", "MLP"], estimator=np.mean, data=data, palette="RdBu", ci="sd", edgecolor=".2")