from agent_dir.agent import Agent
from keras.models import Sequential, Input, Model
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Conv2D, Flatten, Reshape, Lambda, MaxPooling2D, BatchNormalization
import numpy as np


def compute_discounted_R(R, discount_rate=.96):
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    for t in reversed(range(R.shape[0])):
        if R[t] == -1: running_add = 0
        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add
    # discounted_r = (discounted_r - discounted_r.mean()) / (discounted_r.std() + 0.00001)
    return discounted_r

class Agent_DQN(Agent):
    def __init__(self, env, args):
        super(Agent_DQN,self).__init__(env)
        self.env = env
        self.n_action = 4
        self.__build_network()
        self.__build_train_fn()
        self.gamma = 0.99
        self.batch_size = 32
        self.S = None
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            self.policy_network.load_weights('breakouts_dqn_weights333_80.0_15190_1902194.h5')
            #'double_dqn_weights.h5')#
            # breakouts_dqn_weights000.h5
    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        self.observation = self.env.reset()
        pass

    def __build_network(self):
        model = Sequential()
        model.add(Conv2D(32, [8,8], strides=(4,4), input_shape=(84, 84, 4),
                         activation='relu'))
        model.add(Conv2D(64, [4,4], strides=(2,2),
                         activation='relu'))
        model.add(Conv2D(64, [3,3], strides=(1,1),
                         activation='relu'))
        model.add(Flatten())
        # model.add(Dense(64, activation='relu', init='he_uniform'))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(self.n_action+1, activation='linear'))
        model.add(Lambda(lambda x:K.expand_dims(x[:,4],-1)+x[:,0:4]-K.mean(x[:,0:4],keepdims=True),
                         output_shape=(self.n_action,)))
        print(model.summary())
        self.policy_network = model
        model1 = Sequential()
        model1.add(Conv2D(32, [8,8], strides=(4,4), input_shape=(84, 84, 4),
                         activation='relu'))
        model1.add(Conv2D(64, [4,4], strides=(2,2),
                         activation='relu'))
        model1.add(Conv2D(64, [3,3], strides=(1,1),
                         activation='relu'))
        model1.add(Flatten())
        # model.add(Dense(64, activation='relu', init='he_uniform'))
        model1.add(Dense(512))
        model1.add(LeakyReLU(alpha=0.1))
        model1.add(Dense(self.n_action+1, activation='linear'))
        model1.add(Lambda(lambda x:K.expand_dims(x[:,4],-1)+x[:,0:4]-K.mean(x[:,0:4],keepdims=True),
                         output_shape=(self.n_action,)))
        self.target_network = model1
    def __build_train_fn(self):
        action_Q_placeholder = self.policy_network.output
        action_onehot_placeholder = K.placeholder(shape=(None, self.n_action))
        target_Q_placeholder = K.placeholder(shape=(None,))

        action_Qvalue = K.sum(action_Q_placeholder * action_onehot_placeholder, axis=1)
        loss = K.max(K.square(action_Qvalue - target_Q_placeholder))
        adam = Adam(lr=0.0001)
        rmsprop = RMSprop(lr=0.0001, rho=0.99)
        updates = rmsprop.get_updates(params=self.policy_network.trainable_weights,
                                    loss=loss)
        self.train_fn = K.function(inputs=[self.policy_network.input,
                                           action_onehot_placeholder,
                                           target_Q_placeholder],
                                   outputs=[],
                                   updates=updates)
    def fit(self, S, next_S, A, R):
        shuffle = np.random.permutation(R.shape[0])
        S = S[shuffle[:self.batch_size]]; next_S = next_S[shuffle[:self.batch_size]];
        A = A[shuffle[:self.batch_size]]; R = R[shuffle[:self.batch_size]];
        Done = self.done[shuffle[:self.batch_size]]
        A_onehot = to_categorical(A, num_classes=self.n_action)
        def get_target(next_S, R):
            next_Qarray = self.target_network.predict(next_S.reshape(-1,84,84,4))
            # select_Q_onehot = to_categorical(
            #     np.argmax(self.policy_network.predict(next_S.reshape(-1,84,84,4)),axis=1),
            #     num_classes=2)
            # next_Q = np.sum(next_Qarray * select_Q_onehot, axis=1)
            next_Q = np.max(next_Qarray, axis=1)
            target_Q = R + self.gamma * next_Q * Done
            return target_Q

        target_Q = get_target(next_S, R)#.reshape(-1, 1)
        self.train_fn([S.reshape(-1,84,84,4), A_onehot, target_Q])
        # print(loss)


    def train(self):
        reward_history = []
        explore = 1.
        history = 10000
        accumulate = 0
        test = False
        self.done = np.ones([history,])
        S = np.zeros([history, 84, 84, 4])
        next_S = np.zeros_like(S)
        A = np.zeros([history, ])
        R = np.zeros([history, ])
        for i in range(500000):
            self.init_game_setting()
            state = self.observation
            done = False
            episode_reward = 0.0
            j = 0
            total_A = 0
            # playing one game
            while (not done):
                # self.env.render()
                num = accumulate%history
                action = self.make_action(state, test=test)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                S[num] = self.S
                next_S[num] = state
                A[num] = action
                R[num] = reward
                # if action == 2:
                #     total_A = total_A + 1
                j = j + 1
                self.done[num] = 0 if done else 1

                if accumulate > history and (accumulate+1)%4 == 0:
                    # history_R = compute_discounted_R(history_R)
                    self.fit(S, next_S, A, R)
                    if (accumulate+1)%100 ==0:
                        self.policy_network.save_weights('breakouts_dqn_weights444.h5')
                if accumulate > history and (accumulate+1) % 1000 == 0:
                    self.target_network.set_weights(self.policy_network.get_weights())
                if accumulate > history and explore>0.05:
                    explore = explore* 0.9999982#- 9.6e-7 #
                test = False if np.random.choice(2, p=[explore, 1 - explore]) == 0 else True
                accumulate = accumulate + 1
            print('episode:', i, 'episode reward:', episode_reward, 'steps:', accumulate,
                  #'down:', total_A/j,
                    'number of states:', j)
            ### save and print
            reward_history.append(episode_reward)
            np.save('reward_history_dqn444.npy', reward_history)
            if episode_reward > 30:
                self.policy_network.save_weights('breakouts_dqn_weights444_' +
                                                 str(episode_reward) + '_' +
                                                 str(i) + '_'+str(accumulate)+'.h5')
        pass
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        # self.env.env.render()


        # Qarray = np.squeeze(Qarray, axis=0)
        # action = np.random.choice(2, p=Qarray)
        # self.observation = prev_observation
        # test = np.random.choice([True, False], p=[0.999, 0.001])
        self.S = observation
        if test:
            Qarray = self.policy_network.predict(observation.reshape(1, 84, 84, 4))
            return Qarray.argmax()
        else:
            return np.random.choice(4,p=[0.25,0.25,0.25,0.25])

