import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



data = pd.DataFrame({'score': [], 'apple_score': [], 'steps' : [], 'clf' : [] , 'path_finder' : [], 'nb_parameter' : [], 'size' : [] })
clfs = ['logreg','SVM','Forest','MLP']


for clf in clfs:
    path = "data/supervised_{}.csv".format(clf)
    df = pd.read_csv(path)
    print('******************  ' + clf + '  ************************')
    print('greedy')
    s_g = df.loc[df['path_finder'] == 'greedy'].groupby(['nb_parameter']).max()[['score','apple_score','steps']]
    print(s_g)

    print('\n','a_star')
    s_a = df.loc[df['path_finder'] == 'a_star'].groupby(['nb_parameter']).max()[['score','apple_score','steps']]
    print(s_a)
    
    df['clf'] = clf
    data = pd.concat([data, df[['score', 'apple_score', 'steps', 'clf', 'path_finder', 'nb_parameter', 'size']] ])
    print("\n")
    






### mean all clf
fig1, (ax1) = plt.subplots(1, 1)
sns.barplot(x="clf", y="apple_score", hue="path_finder", order=clfs, estimator=np.mean, data=data, palette="RdBu", ci=None, edgecolor=".2", ax=ax1)
sns.set(style="whitegrid")
plt.title("Statistics mean all clf")


### mean score MLP, Forest
fig2, (ax2) = plt.subplots(1, 1)
data2 = data.groupby(['nb_parameter', 'clf']).mean()['score']
data2 = data2.drop([ [0.0, 'logreg'], [0, 'SVM']])

data2 = data2.reset_index(level=['nb_parameter','clf'])
data2 = data2.pivot('nb_parameter','clf', 'score')

sns.heatmap( data = data2, annot=True, center = data2.median().median(), linewidths=.5, cmap='Reds', ax= ax2)
ax2.set_ylim(top=0, bottom=data2.index.size)
ax2.set_xlim(left=0, right=2)
plt.title("Statistics mean score MLP, Forest")


### max score MLP, Forest
fig3, (ax3) = plt.subplots(1, 1)
data3 = data.groupby(['nb_parameter', 'clf']).max()['score']
data3 = data3.drop([ [0.0, 'logreg'], [0, 'SVM']])

data3 = data3.reset_index(level=['nb_parameter','clf'])
data3 = data3.pivot('nb_parameter','clf', 'score')

sns.heatmap( data = data3, annot=True, center = data3.median().median(), linewidths=.5, cmap='Blues', ax= ax3)
ax3.set_ylim(top=0, bottom=data3.index.size)
ax3.set_xlim(left=0, right=2)
plt.title("Statistics max score MLP, Forest")



### path_finder
fig4, (ax4) = plt.subplots(1, 1)
path_finder_data = pd.read_csv('data/path_finder.csv')
sns.barplot(x=path_finder_data['path_finder'], y='score', estimator=np.mean, data=path_finder_data, palette="RdBu", ci='sd', edgecolor=".2", ax=ax4)
sns.set(style="whitegrid")
plt.title("Statistics path finder")