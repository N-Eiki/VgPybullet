# colab notebooks for tossingsub

## Installation

    Upload tossingsub to your google driver folder (e.g. MyDrive/Colab Notebook/)

## Usage

### Train with Q-learning, DQN, DDQN, or AC

 - train_q_ur5etosser.ipynb: this has almost the same functionality as src/train_q_ur5etosser.py
 - train_dqn_ur5etosser.ipynb: this has almost the same functionality as src/train_q_ur5etosser.py
 - train_ddqn_ur5etosser.ipynb: this has almost the same functionality as src/train_q_ur5etosser.py 
 - train_ac_ur5etosser.ipynb: this has almost the same functionality as src/train_q_ur5etosser.py

##### Use pre-trained weights

Please specify the weight file path as below.

    weightpath = weight/object/learning_algorithm/reward_function/filename

##### Train from scratch

Please set the variable `weightpath` as None.

    weightpath = None
