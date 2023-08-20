# AI for self-driving car

# Importing the libraries
import numpy as np
import random      # to select random action for the car
import os      # when you need to save and load the model when the system has shut down
import torch
import torch.nn as nn
import torch.nn.functional as F        # it contains different functions for the NN, typically loss function
import torch.optim as optim
import torch.autograd as autograd      # conveting tensor to variable containing tensor
from torch.autograd import Variable

# Creating architecture of the Neural Network
class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):       # this function will activate the neurons and return Q-values for each possible actions
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size)) # using random.sample we pick a random sample from memory, of fixed size = batch_size
        # zip function works like reshape function
        # if List = ((1,2,3),(4,5,6)), then zip(*List) = ((1,4),(2,5),(3,6))
        # say we have List = ((s1,a1,r1),(s2,a2,r2))        s=states, a=action, r=reward
        # but we want in this format ((s1,s2),(a1,a2),(r1,r2)) so we use zip

        return map(lambda x: Variable(torch.cat(x, 0)), samples) # we cant directly return the list, so convert it into pyTorch variable first

# Implementing Deep Q Learning

class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # fake dimension has to be first dimension of the last state
        # it contains five vectors, 3 signals and 2 orientations (plus and minus)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state): # tells AI which action to play at what time
        probs = F.softmax(self.model(Variable(state, volatile = True))*7) # temperature_parameter = 7 - it's a parameter to tell surety of an event
        action = probs.multinomial()
        return action.data[0,0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad() # zero_grad re-initialises your opitmizer at the start of each loop
        td_loss.backward(retain_variables = True)
        self.optimizer.step() # this updates the weight

    def update(self, reward, new_signal):       # it is actually last_reward and last_signal from map.py
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_state)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1) # +1 to avoid reward_window = 0, denominator cant be 0

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict}
                   , 'last_brain.pth') # new file will be created named 'last_brain.pth'

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print('=> loading checkpoint...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('done !')
        else:
            print('no checkpoint found...')