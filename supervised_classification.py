from __future__ import division
import argparse
import pandas as pd
import numpy as np
import os

from env.a_star_path_finder_env import AStarEnv
from env.greedy_path_finder_env import GreedyEnv
from env.ai_game_env import *
from snake import *
import supervised.pick_clf as pick_clf
import supervised.preprocess_and_accuracy as preprocess_and_accuracy
from env.classifier_env import ClassifierEnv

'''
--mode:   feed: generate data for the supervised classification 
          train-test: fit the model to the data and test the model
--no_feed_data: if True no data collected
--episode: number of iterations
--clf: name of the classifier
--all_data: If True use all the data to fit the model. If False, the model will not use the data of the body
--grid: size of the grid
--poly_features: if True add polynomial features to the data. WARNING use this only when all_data = False
--predict_and_test: if True test on the training set and display the training accuracy
--nb_parameter: only apply when clf =  kNN, MLP, Forest
                for kNN number of neighbors
                for MLP number of layers
                for Forest number of trees
--path_finder: type of path_finder to use, two choices, greedy and a_star. a_star is better but more complex. 
               a_star is an implementation of the A star algorithm.
               greedy is a greedy algorithm that takes the shortest path to the apple and turns whenever he can.
'''




parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['feed', 'train-test'], default='train-test')
parser.add_argument('--no_feed_data', action='store_true', default=False)
parser.add_argument('--episode', type=int, default=1)
parser.add_argument('--clf', choices=['logreg', 'kNN', 'SVM', 'Nusvm', 'MLP', 'Forest', 'multiclass'], default='logreg')
parser.add_argument('--all_data', action='store_true', default=False)
parser.add_argument('--grid', type=int, default=20)
parser.add_argument('--poly_features', action='store_true', default=False)
parser.add_argument('--predict_and_test', action='store_true', default=True)
parser.add_argument('--nb_parameter', type=int, default=50)
parser.add_argument('--path_finder', choices=['greedy', 'a_star'], default='greedy')

args = parser.parse_args()

if args.mode == 'feed':
    grid = Grid(args.grid, args.grid, 30)
    if args.path_finder == "greedy":
        env = GreedyEnv(grid)
    elif args.path_finder == "a_star":
        env = AStarEnv(grid)
    else:
        raise ValueError("wrong pathfinder arg")
    for i in range(args.episode):
        print("episode ", i + 1)
        steps, score = env.play(data_feeding=not args.no_feed_data, wait=not bool(i))
        print("steps = ", steps)
        print("apple score = ", score)

if args.mode == 'train-test':
    clf = pick_clf.pick_clf(args.clf, args.nb_parameter)
    datapath = "data\\data_{}_{}.csv".format(args.path_finder, args.grid)

    data = pd.read_csv(datapath)

    # load x_train
    if args.all_data:
        x_train = data.drop(['Action', 'Unnamed: 0'], axis=1)
    else:
        x_train = data[['Headx', 'Heady', 'Applex', 'Appley', 'x+', 'x-', 'y+', 'y-']]

    if args.poly_features:
        x_train = preprocess_and_accuracy.prepare_data(x_train)

    # load y_train
    y_train = np.array(data[['Action']])[:, 0]

    clf.fit(x_train, y_train)

    if args.predict_and_test:
        acc_test = preprocess_and_accuracy.predict_and_test(clf, x_train, y_train)
        print('******************  Training accuracy *********************')
        print('ACC multinomial: ', acc_test)

    grid = Grid(args.grid, args.grid, 30)
    
    for i in range(args.episode):
        ai_class = ClassifierEnv(grid, clf, args.all_data, args.poly_features)
        steps, score = ai_class.play(wait=not bool(i))
        print("episode ", i + 1)
        print("steps = ", steps)
        print("apple score = ", score)
        path = '.\\data\\supervised_{}.csv'.format(args.clf)
        header = not os.path.exists(path)
        df = pd.DataFrame({'steps': [steps], 'apple_score': [score-1], 'score': [(score-1)*50-steps-100], 'size': [args.grid], 
                           'path_finder': [args.path_finder], 'poly_features': [args.poly_features], 'all_data': [args.all_data] })
        if args.clf in ['kNN', 'MLP', 'Forest']:
            df['nb_parameter'] = [args.nb_parameter]
        else:
            df['nb_parameter'] = [0]
        df.to_csv(path, mode='a', header=header, index=False)
