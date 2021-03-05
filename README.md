# Udacity Deep Reinforcement Learning NanoDegree
## Project Navigation

### Introduction
This project aims to train an agent to navigate inside a finite environment where the goal is to collect as much
yellow bananas as possible while avoiding blue bananas.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file under this repository, and unzip (or decompress) the file. 
3. The code is written in Python 3 and uses the PyTorch library. You can install the requirements with:

```
conda create -n drlnd python=3.6
conda activate drlnd
pip install torch
pip install unityagents
```


### Environment Details

The observation given to the agent is a 37 dimensions vector and contains the agent's velocity, as well as a ray-based
perception of objects around the agent's forward direction.

The agent is able to choose between 4 possible discrete actions:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.

The task is episodic, and the environment is considered solved, when the agent gets an average score of +13 over 100 consecutive episodes.

### Solving with DQN

In this section, we try to solve the problem using DQN.
Details of the model are provided in `Report.md`. 

Note: you can skip the training part and use the provided pretrained `model.pt` to rollout a trained agent.

#### Training

To launch training, you need to run the train script with the file path to Unity environment as argument:

`python train.py <path_to_unity_env_here>`

For example on Linux: `python train.py Banana_Linux/Banana.x86_64`

You can customize the different options for training by modifying the variables at the top of file `train.py`.

### Rollout

To launch rollout, you need to run the rollout script with the file path to Unity environment as argument:

`python rollout.py <path_to_unity_env_here>`

For example on Linux: `python rollout.py Banana_Linux/Banana.x86_64`

You can customize the different options for rollout by modifying the variables at the top of file `rollout.py`.