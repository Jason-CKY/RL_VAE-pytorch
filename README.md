# Deep RL policies on Pybullet Environments

This repo is a pytorch implementation of training a variational autoencoder (VAE). This is written to train a VAE for use in a RL environment, and contains code to generate images from random exploration of various RL environments from [pybullet](https://github.com/bulletphysics/bullet3) and [RLBench](https://github.com/stepjam/RLBench) for training.

## Dependencies:
* CUDA >= 10.2
* [RLBench](https://github.com/stepjam/RLBench), only if you want to use RLBench environments to train VAE

## How to use
* Clone this repo
* pip install -r requirements.txt

### Training model for openai gym environment
* Edit training parameters in ./Algorithms/<algo>/config.json
```
python train.py
usage: train.py [-h] [--env ENV] [--agent {ddpg,trpo,ppo,td3,random}]
                [--arch {mlp,cnn}] --timesteps TIMESTEPS [--seed SEED]
                [--num_trials NUM_TRIALS] [--normalize] [--rlbench] [--image]

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             environment_id
  --agent {ddpg,trpo,ppo,td3,random}
                        specify type of agent
  --arch {mlp,cnn}      specify architecture of neural net
  --timesteps TIMESTEPS
                        specify number of timesteps to train for
  --seed SEED           seed number for reproducibility
  --num_trials NUM_TRIALS
                        Number of times to train the algo
  --normalize           if true, normalize environment observations
  --rlbench             if true, use rlbench environment wrappers
  --image               if true, use rlbench environment wrappers
```

### Testing trained model performance
```
python test.py
usage: test.py [-h] [--env ENV] [--agent {ddpg,trpo,ppo,td3,random}]
               [--arch {mlp,cnn}] [--render] [--gif] [--timesteps TIMESTEPS]
               [--seed SEED] [--normalize] [--rlbench] [--image]

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             environment_id
  --agent {ddpg,trpo,ppo,td3,random}
                        specify type of agent
  --arch {mlp,cnn}      specify architecture of neural net
  --render              if true, display human renders of the environment
  --gif                 if true, make gif of the trained agent
  --timesteps TIMESTEPS
                        specify number of timesteps to train for
  --seed SEED           seed number for reproducibility
  --normalize           if true, normalize environment observations
  --rlbench             if true, use rlbench environment wrappers
  --image               if true, use rlbench environment wrappers
```