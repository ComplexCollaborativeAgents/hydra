from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import math
import pred_model
import pickle
import pdb


LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:

    def __init__(self, model = None, state_dim = 4, action_dim = 1, output_dim=4):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        if model == None:
            self.model = pred_model.PredModel(self.state_dim, self.action_dim, self.output_dim).to(device)
        else:
            self.model = model
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

    def train(self, X_state, X_Action, Y_state, num_epochs = 200, BATCH_SIZE = 128):

        trainset = TensorDataset(torch.tensor(X_state).float().to(device), torch.tensor(X_Action).float().to(device), torch.tensor(Y_state).float().to(device))
        trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
        for i in range(num_epochs):
            print('Epoch: {}'.format(i+1))

            loss_sum = 0
            iter_num = 0
            for X_batch, A_batch, Y_batch in trainloader:
                iter_num += 1
                y_hat = self.model.forward(X_batch, A_batch)
                # print(y_batch.dtype, y_hat.dtype)
                loss = self.loss_fn(y_hat, Y_batch)
                self.model_optimizer.zero_grad()
                loss.backward()
                self.model_optimizer.step()
                loss_sum = loss_sum + loss.item()

            epoch_loss = loss_sum/iter_num
            print('Average Epoch Loss: {}'.format(epoch_loss))


    def evaluate(self, X_test, A_test, Y_test):
        self.model.eval()
        with torch.no_grad():
            testset = TensorDataset(torch.Tensor(X_test).float().to(device), torch.tensor(A_test).float().to(device), torch.tensor(Y_test).float().to(device))
            testloader = DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=0, pin_memory=False)

            for x,a,y in testloader:
                y_hat = self.model.forward(x,a)

                # print(y_batch.dtype, y_hat.dtype)
                loss = self.loss_fn(y_hat, y)
                print('TEST LOSS: {}'.format(loss.item()))

    def predict(self, X, A):
        self.model.eval()
        with torch.no_grad():

            testset = TensorDataset(torch.Tensor(X).float().to(device), torch.tensor(A).float().to(device))
            testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

            for x,action in testloader:
                y_hat = self.model.forward(x,action)

        return y_hat




def load_data():
    with open('data_trained.pickle','rb') as f:
        data = pickle.load(f)
    X_state = []
    Y_state = []
    X_action = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            X_state.append(data[i][j][0])
            X_action.append(data[i][j][1])
            Y_state.append(data[i][j][2])
    X_state = np.array(X_state)
    X_action = np.array(np.reshape(X_action,(-1,1)))
    Y_state = np.array(Y_state)
    return X_state, X_action, Y_state

if __name__ =="__main__":
    pretrained_model = True
    X,A,Y = load_data()
    num_train = int(0.9* len(X))
    idx_shuffle = np.random.permutation(len(X))

    print('Number of Train Data Points: {}'.format(num_train))
    print('Number of Test Data Points: {}'.format(len(X) - num_train))

    X_train = X[idx_shuffle[:num_train]]
    A_train = A[idx_shuffle[:num_train]]
    Y_train = Y[idx_shuffle[:num_train]]

    X_test = X[idx_shuffle[num_train:]]
    A_test = A[idx_shuffle[num_train:]]
    Y_test = Y[idx_shuffle[num_train:]]


    num_states = 4
    num_actions = 1
    num_outputs = 4
    if pretrained_model == False:
        trainer_obj = Trainer(num_states,num_actions, num_outputs)
        trainer_obj.train(X_train,A_train,Y_train)
        trainer_obj.evaluate(X_test, A_test)
        torch.save(trainer_obj.model, 'model/pred_model.pkl')
    if pretrained_model == True:
        model = torch.load('model/pred_model.pkl')
        trainer_obj = Trainer(num_states,num_actions, num_outputs,model)
        trainer_obj.evaluate(X_test, A_test, Y_test)
