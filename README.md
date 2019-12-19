# Comparing different techniques of Machine Learning to play Snake

## How to run locally

clone the repository in a floder

```
git clone https://github.com/Maxwell1447/Reinforcement-learning-to-play-Snake.git
```

create a python virtual env in the cloned repository (if you run it on Windows, make sure python is referenced in you environnement variables. More info on [this page](https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/))

```
python -m venv venv . venv/bin/activate
```

install requirements (make sure that pip is upgraded to version 19 or higher)

```
venv\Scripts\pip install -r requirements.txt
```

Make sure your run python scripts in this virtual env then.

***

### Run supervised methods

You can either run and test different models on already-built datasets, or you can create your own dataset.

#### Generate a new dataset

```
python supervised_classification.py --mode feed --episode <Number of games> --grid <size of the grid> --path_finder <greedy || a_star>
```
Note that the grid is square. The pathfinder corresponds to the deterministic algorithm chosen to show the example and feed the data.
In the ```data``` folder, a ```.csv``` file should have been created whith the following name ```data_<pathfinder>_<grid size>.csv```

You can keep on feeding the same data set with the same command. You can freely change the number of episodes then.

#### Fit a model to an existing dataset

Run the following command in a terminal:
```
python supervised_classification.py --mode train-test --grid <size of the grid> --path_finder <greedy || a_star> [--poly_features] [--predict_and_test] [--all_data] --clf <model> [--n_neighbor <kNN tolerance>]
```
This will access the dataset file with the corresponding pathfinder and grid size.

+ ```--poly_feature``` create polynomial features for all features to the second order

+ ```--predict_and_test``` gives the accuracy on the training set

+ ```--all_data``` allows to take into account the tail of the snake. Otherwise, there are only info about the head, the apple and the current direction. You will note that not considering the tail provides better results most of the time.


There is a bunch of models that are proposed by *Scikit Learn*. Here is a list of the classifier you can use

+ **Logistic regression**

use ```--clf logreg```

+ **SVM**

use ```--clf SVM```

+ **SVM with RBF kernel**

use ```--clf Nusvm```

+ **k-Nearest Neighbors**

use ```--clf kNN [--n_neighbor <Number of allowed neighbors>]```

+ **MLP**

use ```--clf MLP```

+ **One vs Rest with SVM + kernel**

use ```--clf multiclass```

***

### Run Deep Q Reinforcement learning

There are already trained networks that you can test, but you can also run the training of a given network model

#### Train a model

Run the following command in a terminal:
```
python snake_train_test.py --mode train --version <model version> [--retrain <training number>] --step <number of step> [--initial_eps <eps value>] [--weights <file name>]
```
+ The model versions are ```v1``` ```v2``` ```v3```. You can [add a custom version model](#custom) if you wish.

+ You have the possiblility to train again the same model. This is with ```--retrain```. You have to increment the training number every time you want to retrain a given model. Every trained version is saved in ```data\``` with  a name in the format: ```dqn_snake_weights_<model version>_<training number>.hf5```

By defalut the training number is set to -1 so that the first file is number 0, so don't indicate the training number the first time you train a model. Then you can specify the trained version you want to retrain, checking if there is a corresponding ```.hf5``` folder in ```data\```.

**Be carful** if the training number has no corresponding file, it will nonetheless train but with new random weights.

+ ```--weights <file name>``` allows to use custom file names to train with. But after the training, a new file will be created according to the training number!

+ ```--step <number of steps>``` corresponds to the number of steps for the training (and not the number of games!). Generally you will need millions of steps to have results. It can take several hours, so be patient :)

+ ```--initial_eps <eps value>``` is the exploration parameter at the beginning of the training. It will linearly deacrease down to ```0.01``` after 2,000,000 steps. Set it to 1 at the first training, and then decrease it to around ```0.3```

#### Add a costum model <a name="custom"></a>

Open and edit the file use ```dqn\models.py```

Create a new function and fill it, given the other models as templates.
```python
def model_custom(input_shape, nb_actions): # you can choose any other name
  ...
```

then edit the ```model``` function by adding:
```python
elif version == "custom_name": # choose the name you want
  return model_custom(input_shape, nb_actions)
```

Now you just have to call ```--version custom_name``` to use this model.

#### Test a model

Run the following command in a terminal:
```
python snake_train_test.py --mode test --version <model version> [--retrain <training number>] [--weights <file name>] --episodes <number of games>
```

+ Use a version and a training number corresponding to an existing .hf5 file.

+ With ```--weights <file name>``` you can use a custom file name for testing. But make sure it is corresponding to the right model version!

+ ```--episodes <number of games>``` corresponds to the number of games that will be played.

***

