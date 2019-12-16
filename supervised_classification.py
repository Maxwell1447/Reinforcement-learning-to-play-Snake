from __future__ import division
import argparse
import pandas as pd
from env.ai_game_env import *
from snake import *
import supervised.pick_clf as pick_clf
import supervised.pick_data_path as pick_data_path
import supervised.preprocess_and_accuracy as preprocess_and_accuracy
import env.greedy_path_finder_env as path_finder
import env.ai_classification as ai_classification


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='test')
parser.add_argument('--clf', choices=['logreg', 'kNN', 'SVM', 'Nusvm', 'MLP'], default='logreg')
parser.add_argument('--all_data', type=bool, default=False)
parser.add_argument('--grid', type=int, default=20)
parser.add_argument('--poly_features', type=bool, default=False)
parser.add_argument('--predict_and_test', type=bool, default=True)
parser.add_argument('--n_neighbor', type=int, default=20)
parser.add_argument('--data', choices=['greedy', 'a_star'], default='greedy')

args = parser.parse_args()

if args.mode == 'train':
    grid = Grid(args.grid, args.grid,30)
    pf = path_finder.PathFinder(grid)
    pf.play()

if args.mode == 'test':
    clf = pick_clf.pick_clf(args.clf,args.n_neighbor)
    datapath = pick_data_path.pick_data_path(args.data, args.grid)
    
    
    data = pd.read_csv(datapath)
    
    # load x_train
    if args.all_data:
        x_train = data.drop(['Action', 'Unnamed: 0'], axis=1)
    else:
        x_train = data[['Headx','Heady','Applex','Appley','x+','x-','y+','y-']]
        
    if args.poly_features:
        x_train = preprocess_and_accuracy.prepare_data(x_train)
    
    # load y_train
    y_train = np.array(data[['Action']])[:,0]
    
    clf.fit(x_train, y_train)
    
    if args.predict_and_test:
        acc_test = preprocess_and_accuracy.predict_and_test(clf,x_train, y_train)
        print('******************  Training accuracy *********************')
        print('ACC multinomial: ', acc_test)
    
    grid = Grid(args.grid, args.grid,30)
    
    
    
    ai_class = ai_classification.ai_classification(grid, clf, args.all_data, args.poly_features)
    ai_class.play()