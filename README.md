# Atari_Game_Playing
Use policy gradient for pong and deep Q networks for breakouts
Q network includes deep Q learning, double Q learning, and duel Q learning

## Reqiurements
Keras==2.0.7
Tensorflow==1.3

## Installation
Type the following command to install OpenAI Gym Atari environment.

`$ pip3 install opencv-python gym gym[atari]`

Please refer to [OpenAI's page](https://github.com/openai/gym) if you have any problem while installing.

## How to run :
training policy gradient:
* `$ python3 main.py --train_pg`

testing policy gradient:
* `$ python3 test.py --test_pg`

training DQN:
* `$ python3 main.py --train_dqn`

testing DQN:
* `$ python3 test.py --test_dqn`

## Results
Get an average of 5 points in pong and 73 points in breakouts.
