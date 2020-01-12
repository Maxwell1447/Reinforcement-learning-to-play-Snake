import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


sns.set(style="whitegrid")
plt.title("Statistics")

data = pd.DataFrame({'clf' : [] , 'path_finder' : [], 'size' : [], 'average_apple' : [], 'average_score' : [] })
clfs = ['Forest','MLP']
path_finders = ['greedy', 'a_star']
sizes = [10]

for clf in clfs:
    for size in sizes:
        for path_finder in path_finders:
            datapath = ".\\data\\supervised_{}_{}_{}.csv".format(clf, size, path_finder)
            df = pd.read_csv(datapath, index_col=0)
            average_apple, average_score = df['apple_score'].mean(), df['score'].mean()
            s = pd.Series([clf, average_apple, path_finder, size, average_apple, average_score], index=[size], name= clf + " " + path_finder + " " + size)
            

ax = sns.barplot(x="clf", y="averagescore", hue="path_finder",col="path_finder", order=["Forest", "MLP"], estimator=np.mean, data=data, palette="RdBu", ci="sd", edgecolor=".2")