import numpy as np
import tensorflow as tf
import os
import datetime

n = 1116 #1 , 5 , 7 , 9 best yet  1016

class MyModel(tf.keras.Model): 
    def __init__(self, num_states, hidden_units, num_actions): #num_states = states dimensions
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.losses = []
        self.meanloss = 1000
        self.checkpoint_path = f"tmp/log/checkpoint/cp-best{n}.ckpt"
        self.checkpoint_path_testing = f"tmp1/log/checkpoint/cp-best{n}.ckpt"
        self.checkpoint_path_precautious = f"tmp/log/checkpoint/cp-best{n+1}.ckpt"

        self.checkpoint_dir = os.path.dirname("tmp/log")
        self.checkpoint_dir_testing = os.path.dirname("tmp1/log")
        self.checkpoint_dir_precautious = os.path.dirname("tmp/log")

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(    filepath=self.checkpoint_path,     verbose=1,     save_weights_only=True,    save_freq=5*200)
        self.cp_callback_1 = tf.keras.callbacks.ModelCheckpoint(    filepath=self.checkpoint_path_testing,     verbose=1,     save_weights_only=True,    save_freq=5*200)
        self.cp_callback_2 = tf.keras.callbacks.ModelCheckpoint(    filepath=self.checkpoint_path_precautious,     verbose=1,     save_weights_only=True,    save_freq=5*200)
   
    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.model.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        #self.model.load_weights(f"tmp/log/checkpoint/cp-best{n-1}.ckpt")

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.model(states) * tf.one_hot(actions, self.num_actions), axis=1)
            #loss = tf.math.reduce_mean((actual_values - selected_action_values))
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        variables_earlystop = self.model.trainable_variables.copy()
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.losses.insert(0,loss)
        n_factor = 100
        if (len(self.losses)>n_factor and len(self.losses)%100 == 0) :
            self.meanloss =min( tf.math.reduce_mean(self.losses[0:n_factor ]) , self.meanloss)
            if tf.math.reduce_mean(self.losses[0:n_factor ])>self.meanloss and len(self.losses)> n_factor*100 and  len(self.losses)%300 == 0:
                #self.model.load_weights(self.checkpoint_path)
                #print("                                                                                             load")
                return loss
            else:
                self.model.save_weights(self.checkpoint_path.format(epoch=0))
                self.model.save_weights(self.checkpoint_path_testing.format(epoch=0))
                self.model.save_weights(self.checkpoint_path_precautious.format(epoch=0))

                print("                                                                                             save")
                return loss
        return loss


    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            print(f"                                    random  {epsilon}")
            return np.random.choice(self.num_actions)
        else:
            print(f"                              prediction  {epsilon}")
            return np.argmax(self.model.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()
    losses = list()
    while not done:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        rewards += reward
        if done:
            reward = -200
            env.reset()

        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        loss = TrainNet.train(TargetNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return rewards, mean(losses)
