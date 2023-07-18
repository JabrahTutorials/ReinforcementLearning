# Overview
This is the code accompanying the lecture on self-driving with carla that can be found here https://www.youtube.com/watch?v=MNiqlHC6Kn4&t
You should only run this after installing carla and getting a grasp on how to run the simulator and calling the python API, which is documented very well on their site.

# How to Run

## Setup
I did not create a dependency or yml file (will do so at a later time), but you need carla, pygame, pytorch, opencv and numpy to run this project

You should ensure that you have a `weights` folder when you run the project. If you do not have one, then just run `initial_setup.py` and it will create it for you. If you just cloned the repository, I reccomend you run this file first.

## main.py
Run this file if you want to evaluate the performance of your agent
A couple of items to note

```
env = SimEnv(visuals=False)
```

The call above initializes our simulation environment. You should set visuals to `False` if you do not want to open this with pygame, or to `True` if you want a pygame window to open along with the simulator.

```
model.load('weights/model_ep_4400')
```

This loads a trained/pre-trained model. The program will not run unless it can load this model.

## train.py
This is for training the model. The model only starts learning after a certain number of episodes, and it can take from 8-10 hours (at least on my setup) before we see signs of learning. I will now describe a few variables you can set to configure your training process. You can modify them yourself in `config.py`.

`target_speed` --> Speed you want the car to move at in km/h

`max_iter` --> Maximum number of steps before starting a new episode

`start_buffer` --> Number of episodes to run before starting training

`train_freq` --> How often to train (set to 1 to train every step, 2 to train every 2 steps etc)

`save_freq`: --> Frequency of saving our model

`start_ep` --> Which episode should we start on (just a counter which you can update if program crushes while training for example)

`max_dist_from_waypoint` --> Maximum distance from waypoint/road before we decide to terminate the episode
