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

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['feed', 'train-test'], default='train-test')
parser.add_argument('--no_feed_data', action='store_true', default=False)
parser.add_argument('--episode', type=int, default=1)
parser.add_argument('--clf', choices=['logreg', 'kNN', 'SVM', 'Nusvm', 'MLP', 'Forest', 'multiclass'], default='logreg')
parser.add_argument('--all_data', action='store_true', default=True)
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
        env.play(data_feeding=not args.no_feed_data, wait=not bool(i))

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
        df = pd.DataFrame({'steps': [steps], 'apple_score': [score-1], 'score': [(score-1)*100-steps], 'size': [args.grid], 
                           'path_finder': [args.path_finder], 'poly_features': [args.poly_features], 'all_data': [args.all_data] })
        if args.clf in ['kNN', 'MLP', 'Forest']:
            df['nb_parameter'] = [args.nb_parameter]
        else:
            df['nb_parameter'] = [0]
        df.to_csv(path, mode='a', header=header, index=False)
