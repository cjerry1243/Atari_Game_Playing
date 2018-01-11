from agent_dir.agent import Agent
from keras.models import Sequential, Input, Model
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
from keras import backend as K
from keras.layers import Dense, Conv2D, Flatten, Reshape, Lambda, MaxPooling2D, BatchNormalization
import numpy as np
import scipy

def RGB2gray(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return 1/3 * R + 1/3 * G + 1/3 * B

def prepro(o, image_size=[80, 80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    Input:
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array
        Grayscale image, shape: (80, 80, 1)

    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32), axis=2).ravel()

def preprocess(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_PG,self).__init__(env)
        self.env = env
        self.S = None
        self.mean = 0.
        self.std = 1.
        self.batch_size = 32
        self.__build_network()
        self.__build_train_fn()

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.model.load_weights('pong_pg_weights_13.0_5867.h5')

            # self.model.load_weights('pong_pg_weights.h5') pong_pg_weights10_8035.h5

    def init_game_setting(self):
        self.observation = self.env.reset()
        pass
    def __build_network(self):
        model = Sequential()

        model.add(Reshape((80, 80, 1), input_shape=(6400,)))
        model.add(Conv2D(16, [8, 8], strides = (4, 4), #subsample=(3, 3), border_mode='same',
                                activation='relu', init='lecun_uniform'))
        model.add(Conv2D(32, [4, 4], strides = (2, 2), #subsample=(3, 3), border_mode='same',
                                activation='relu', init='lecun_uniform'))
        model.add(Flatten())
        # model.add(Dense(64, activation='relu', init='he_uniform')), init='lecun_uniform'
        model.add(Dense(128, activation='relu', init='lecun_uniform'))
        model.add(Dense(2, activation='softmax'))
        print(model.summary())
        self.model = model
    def __build_train_fn(self):
        # def loss(discount_r):
        #     def f(y_true, y_pred):
        #         action_prob = K.sum(y_true*y_pred, axis=1)
        #         action_prob = K.log(action_prob)
        #         policy_loss = -K.sum(discount_r) * K.mean(action_prob)
        #         policy_loss = K.print_tensor(policy_loss)
        #         return policy_loss
        #     return f
        # discount_reward_ = Input(shape=(1,))
        # state = Input(shape=(6400,))
        # pi_action = self.model(state)
        # model = Model([state, discount_reward_], pi_action)
        # adam = Adam(lr=1e-4)
        # rmsprop = RMSprop(lr=1e-4 ,clipnorm=1) #10
        # model.compile(optimizer=rmsprop, loss=loss(discount_reward_))
        action_prob_placeholder = self.model.output
        action_onehot_placeholder = K.placeholder(shape=(None, 2))
        discount_reward_placeholder = K.placeholder(shape=(None,))
        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)
        loss = - log_action_prob * discount_reward_placeholder
        loss = K.sum(loss)
        adam = Adam(lr=1e-4)#,decay = 0.99)
        rmsprop = RMSprop(lr=1e-4, decay=0.99)
        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)
        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
                                           discount_reward_placeholder],
                                   outputs=[loss],
                                   updates=updates)
        # self.update_model = model

    def fit(self, S, A, discount_reward):
        action_onehot = to_categorical(A.reshape(-1), num_classes=2)
        loss = self.train_fn([S.reshape(S.shape[0], -1), action_onehot, discount_reward])
        print(loss)
    def run_episode(self,i):  ####### playing one episode
        state = self.observation
        done = False
        episode_reward = 0.0
        # S, A, R , sample_R= [], [], [], []
        S = np.zeros([10000, 6400])
        A = np.zeros([10000,])
        R = np.zeros([10000,])
        j = 0
        while (not done):
            # if (i+1)%100 ==0 and i<500:
            #     self.env.env.render()
            # if (i+1)%200 ==0 and i>499:
            #     self.env.env.render()
            action = self.make_action(state, test=False)
            state, reward, done, info = self.env.step(action)
            episode_reward += reward
            S[j] = self.S
            A[j] = 0 if action == 2 else 1
            R[j] = reward
            # if reward != 0:
            #     print('episode:', i, 'action:', j, 'reward:', reward)
            j = j + 1
            # if reward != 0:
        print('number of down action:', sum(A)/j, 'number of states:', j)
        if i==0 and (0.45> sum(A)/j or sum(A)/j >0.55): exit()
        # Rn = (R-R.mean())/(R.std()+1e-6)
        def compute_discounted_R(R, discount_rate=.99):
            discounted_r = np.zeros_like(R, dtype=np.float32)
            running_add = 0
            for t in reversed(range(R.shape[0])):
                if R[t] != 0: running_add = 0
                running_add = running_add * discount_rate + R[t]
                discounted_r[t] = running_add
            discounted_r = (discounted_r-discounted_r.mean()) / (discounted_r.std()+0.00001)
            return discounted_r
        RR = R[:j]
        RR = compute_discounted_R(RR)
        # def makebatches(batchsize, S, A, R):
        #     trajectory = R.shape[0]
        #     shuffle = np.random.permutation(trajectory)
        #     S = S[shuffle]
        #     A = A[shuffle]
        #     R = R[shuffle]
        #     iteration = int(trajectory/batchsize)
        #     self.iteration = iteration
        #     S = S[:iteration*batchsize].reshape(iteration, batchsize, -1)
        #     A = A[:iteration*batchsize].reshape(iteration, batchsize)
        #     R = R[:iteration*batchsize].reshape(iteration, batchsize)
        #     '''''''''
        #     sum = R.sum(axis=1)
        #     sum = (sum - sum.mean())/(sum.std()+1e-4)
        #     R_out = np.zeros_like(R)
        #     for it in range(iteration):
        #         R_out[it,0] = sum[it]
        #     '''
        #     return S, A, R
        # S, A, R = makebatches(self.batch_size, S, A, R)
        return S[:j], A[:j], RR-0.01, episode_reward
    def train(self):
        reward_history = []
        for i in range(15000):
            self.init_game_setting()
            S, A, discount_reward, episode_reward = self.run_episode(i)
            self.fit(S, A, discount_reward)

            ########### print and save
            print('episode:', i, 'episode reward:', episode_reward)
            reward_history.append(episode_reward)
            np.save('reward_history_pg.npy', reward_history)
            self.model.save_weights('pong_pg111_weights.h5')
            if episode_reward > 5:
                self.model.save_weights('pong_pg_weights111_' + str(episode_reward) + '_' + str(i) + '.h5')
        pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        # print(self.env.get_random_action())
        # action: 2 = up, 5 = down
        # print(observation)
        # if test:
        #     self.model.load_weights('pong_pg_weights.h5')
        prev_observation = observation
        observation = prepro(observation - self.observation)
        pi_action = self.model.predict(observation.reshape(1,-1))
        pi_action = np.squeeze(pi_action, axis=0)
        if test:
            action = pi_action.argmax()
        else:
            action = np.random.choice(2, p=pi_action)
        self.observation = prev_observation
        self.S = observation
        return 2 if action == 0 else 3