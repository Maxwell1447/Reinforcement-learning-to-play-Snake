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
    s_g = df.loc[df['path_finder'] == 'greedy'].groupby(['nb_parameter']).mean()[['score','apple_score','steps']]
    print(s_g)

    print('\n','a_star')
    s_a = df.loc[df['path_finder'] == 'a_star'].groupby(['nb_parameter']).mean()[['score','apple_score','steps']]
    print(s_a)
    
    df['clf'] = clf
    data = pd.concat([data, df[['score', 'apple_score', 'steps', 'clf', 'path_finder', 'nb_parameter', 'size']] ])
    print("\n")
    


fig1, (ax1) = plt.subplots(1, 1)

sns.barplot(x="clf", y="apple_score", hue="path_finder", order=clfs, estimator=np.mean, data=data, palette="RdBu", ci=None, edgecolor=".2", ax=ax1)
sns.set(style="whitegrid")
plt.title("Statistics")


fig2, (ax2) = plt.subplots(1, 1)
data = data.groupby(['nb_parameter', 'clf']).mean()['score']
data = data.drop([ [0.0, 'logreg'], [0, 'SVM']])
data = data.reset_index(level=['nb_parameter','clf'])
data = data.pivot('nb_parameter','clf', 'score')



sns.heatmap( data = data, annot=True, center = data.median().median(), linewidths=.5, cmap='Reds', ax= ax2)
ax2.set_ylim(top=0, bottom=data.index.size)
ax2.set_xlim(left=0, right=2)
plt.title("Statistics")