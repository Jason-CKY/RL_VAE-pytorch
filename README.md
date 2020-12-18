# Deep RL policies on Pybullet Environments

This repo is a pytorch implementation of training a variational autoencoder (VAE). This is written to train a VAE for use in a RL environment, and contains code to generate images from random exploration of various RL environments from [pybullet](https://github.com/bulletphysics/bullet3) and [RLBench](https://github.com/stepjam/RLBench) for training.

## Dependencies:
* CUDA >= 10.2
* [RLBench](https://github.com/stepjam/RLBench), only if you want to use RLBench environments to train VAE

## How to use
* Clone this repo
* pip install -r requirements.txt

### Generating data from openai gym environment
```
python generate_data.py
usage: generate_data.py [-h] --env ENV --num_samples NUM_SAMPLES
                        [--max_ep_len MAX_EP_LEN] [--seed SEED] [--rlbench]
                        [--view {wrist_rgb,front_rgb,left_shoulder_rgb,right_shoulder_rgb}]

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             environment_id
  --num_samples NUM_SAMPLES
                        specify number of image samples to generate
  --max_ep_len MAX_EP_LEN
                        Maximum length of an episode
  --seed SEED           seed number for reproducibility
  --rlbench             if true, use rlbench environment wrappers
  --view {wrist_rgb,front_rgb,left_shoulder_rgb,right_shoulder_rgb}
                        choose the type of camera view to generate image (only
                        for RLBench envs)
```
