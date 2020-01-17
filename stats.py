import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



data = pd.DataFrame({'score': [], 'apple_score': [], 'steps' : [], 'clf' : [] , 'path_finder' : [], 'nb_parameter' : [], 'size' : [] })
clfs = ['logreg', 'SVM', 'kNN', 'Forest', 'MLP']


for clf in clfs:
    path = "data/supervised_{}.csv".format(clf)
    df = pd.read_csv(path)
    print('******************  ' + clf + '  ************************')
    print('greedy')
    s_g = df.loc[df['all_data'] == True].loc[df['path_finder'] == 'greedy'].groupby(['nb_parameter']).mean()[['score','apple_score','steps']]
    print(s_g)

    print('\n','a_star')
    s_a = df.loc[df['all_data'] == True].loc[df['path_finder'] == 'a_star'].groupby(['nb_parameter']).mean()[['score','apple_score','steps']]
    print(s_a)
    
    df['clf'] = clf
    data = pd.concat([data, df[['score', 'apple_score', 'steps', 'clf', 'path_finder', 'nb_parameter', 'size']] ])
    print("\n")



### mean and median per clf
fig1, (ax1, ax11) = plt.subplots(1, 2)
sns.barplot(x="clf", y="apple_score", hue="path_finder", order=clfs, estimator=np.median, data=data, palette="ocean", ci="sd", edgecolor=".2", ax=ax1).set_title("Mean apple score per clf")
sns.set(style="darkgrid")

sns.barplot(x="clf", y="apple_score", hue="path_finder", order=clfs, estimator=np.mean, data=data, palette="ocean", ci="sd", edgecolor=".2", ax=ax11).set_title('Median apple score per clf')
sns.set(style="darkgrid")



### mean score MLP, Forest, kNN
fig2, (ax2, ax21) = plt.subplots(1, 2)

data2 = data.groupby(['nb_parameter', 'clf']).mean()['score']
data2 = data2.drop([ [0.0, 'logreg'], [0, 'SVM']])
data2 = data2.drop(index = 'kNN',level=1)
data2 = data2.reset_index(level=['nb_parameter','clf'])
data2 = data2.pivot('nb_parameter','clf', 'score')

data21 = data.groupby(['nb_parameter', 'clf']).max()['score']
data21 = data21.drop([ [0.0, 'logreg'], [0, 'SVM']])
data21 = data21.drop(index = 'kNN',level=1)
data21 = data21.reset_index(level=['nb_parameter','clf'])
data21 = data21.pivot('nb_parameter','clf', 'score')

sns.heatmap( data = data2, annot=True, center = data2.mean().mean(), fmt='g', linewidths=.5, cmap='coolwarm', ax= ax2).set_title('Mean score MLP, Forest')
ax2.set_ylim(top=0, bottom=data2.index.size)
ax2.set_xlim(left=0, right=2)

sns.heatmap( data = data21, annot=True, center = data21.max().mean(), fmt='g', linewidths=.5, cmap='coolwarm', ax= ax21).set_title("Max score MLP, Forest")
ax21.set_ylim(top=0, bottom=data2.index.size)
ax21.set_xlim(left=0, right=2)



### mean apple_score MLP, Forest, kNN
fig3, (ax3, ax31) = plt.subplots(1, 2)
data3 = data.groupby(['nb_parameter', 'clf']).mean()['apple_score']
data3 = data3.drop([ [0.0, 'logreg'], [0, 'SVM']])
data3 = data3.drop(index = 'kNN',level=1)
data3 = data3.reset_index(level=['nb_parameter','clf'])
data3 = data3.pivot('nb_parameter','clf', 'apple_score')

data31 = data.groupby(['nb_parameter', 'clf']).max()['apple_score']
data31 = data31.drop([ [0.0, 'logreg'], [0, 'SVM']])
data31 = data31.drop(index = 'kNN',level=1)
data31 = data31.reset_index(level=['nb_parameter','clf'])
data31 = data31.pivot('nb_parameter','clf', 'apple_score')

sns.heatmap( data = data3, annot=True, center = data3.mean().mean(), fmt='g', linewidths=.5, cmap='coolwarm', ax= ax3).set_title('Mean apple score MLP, Forest')
ax3.set_ylim(top=0, bottom=data2.index.size)
ax3.set_xlim(left=0, right=2)
plt.title("Mean apple score MLP, Forest")

sns.heatmap( data = data31, annot=True, center = data31.mean().mean(), fmt='g', linewidths=.5, cmap='coolwarm', ax= ax31).set_title("Max apple score MLP, Forest")
ax31.set_ylim(top=0, bottom=data2.index.size)
ax31.set_xlim(left=0, right=2)



### path_finder
fig4, (ax4) = plt.subplots(1, 1)
path_finder_data = pd.read_csv('data/path_finder.csv')
sns.barplot(x=path_finder_data['path_finder'], y='apple_score', estimator=np.mean, data=path_finder_data, palette="ocean", ci='sd', edgecolor=".2", ax=ax4)
sns.set(style="darkgrid")
plt.title("Apple score path finder")