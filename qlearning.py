#---------#
# Imports #
#---------#

import matplotlib.pyplot as plt
import random

import numpy as np
import torch

import neuralnetwork as nn
import mlutils as ml

#-----------#
# Functions #
#-----------#

def epsilon_greedy(state, valid_moves_F, Qnet, epsilon, turn):
    moves = valid_moves_F(state)
    if np.random.uniform() < epsilon: # Random move
        move = moves[random.sample(range(len(moves)), 1)[0]]
        Q = Qnet.use_pytorch(np.array(state + [move])) if Qnet.Xmeans is not None else 0
    else: # Greedy move
        Qs = []
        for m in moves:
            Qs.append(Qnet.use_pytorch(np.array(state + [m])) if Qnet.Xmeans is not None else 0)
        move = moves[np.argmax(Qs)] if turn else moves[random.sample(range(len(moves)), 1)[0]] # Train agent against random player
        #move = moves[np.argmax(Qs)] if turn else moves[np.argmin(Qs)] # Train agent against itself
        Q = np.max(Qs) if turn else np.min(Qs)
    return move, Q

def train_Qnet(valid_moves_F, make_move_F, boxes_created_F, n_batches, n_reps_per_batch, network, n_iterations, learning_rate, epsilon_decay_factor, use_ReLU=False, use_SGD=False):
    # Use CUDA cores if available, else use CPU
    Qnet = nn.NN(25, network, 1, use_ReLU).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).double()
    repk = -1
    epsilon = 1 # Degree of randomness of move as a fraction => initially 1, i.e. 100% random move
    outcomes = np.zeros(n_batches*n_reps_per_batch)
    errors = []
    for batch in range(n_batches):
        if batch > 0:
            epsilon *= epsilon_decay_factor
            epsilon = max(0.01, epsilon) # Minimum probability of a random move is 0.01
        samples = []
        for reps in range(n_reps_per_batch):
            repk += 1
            # Initialize game
            state = [0]*24
            boxes = [0]*9
            score = [0]*2
            done = False
            # Start game; player 1's turn initially
            turn = True
            move, _ = epsilon_greedy(state, valid_moves_F, Qnet, epsilon, turn)
            # Continue to play the game
            while not done:
                r = 0
                state_next = make_move_F(state, move)
                created, boxes = boxes_created_F(state_next, boxes) # Check how many boxes are created after making a move
                if created > 0: # If a box is created, update score of the player who made the move
                    if turn:
                        score[0] += created
                    else:
                        score[1] += created
                else: # Else give turn to the other player
                    turn = not turn
                if 0 not in state_next: # If there are no more edges remaining, the game is over
                    done = True # Set termination flag
                    Qnext = 0 # Determine the reinforcement
                    r = 1 if score[0] > score[1] else -1
                    outcomes[repk] = r
                    #move_next = -1
                else: # Else, determine next move and add current sample with reinforcement 0
                    move_next, Qnext = epsilon_greedy(state_next, valid_moves_F, Qnet, epsilon, turn)
                samples.append([*state, move/23, r, Qnext]) # Collect turn results as a sample
                state = state_next
                move = move_next
        samples = np.array(samples) # Samples contains the training inputs and the targets
        X = samples[:, :25] # Training inputs to the neural network
        T = samples[:, 25:26]+samples[:, 26:27] # Target values for the neural network
        Qnet, error = Qnet.train_pytorch(X, T, learning_rate, n_iterations, use_SGD) # Training the neural network
        errors.append(np.array(error))
    print('TRAINED')
    return Qnet, outcomes, np.array(errors)

def print_prediction(state):
    for i in range(4):
        for j in range(4):
            print('    \u2022    ', end = '')
            print() if j == 3 else print(state[(3*i)+j], end = '')
        if i != 3:
            print(state[(4*i)+12], '    ', state[(4*i)+13], '    ', state[(4*i)+14], '    ', state[(4*i)+15])
    return

def print_Qs(Qs):
    '''Print Q values as a list to observe if they reduce/change across edges'''
    print('Q values: {}'.format([float('%.3f' % i) for i in list(np.array(Qs).flatten())]))
    return

def plot_outcomes(outcomes, batches):
    temp = outcomes.reshape(batches, -1)
    results = []
    for i in range(temp.shape[0]):
        result = np.sum(temp[i] == 1) # Aggregating results per batch
        results.append(result)
    plt.figure(figsize=(15, 12))
    #plt.xticks(np.arange(0, batches, step=1))
    plt.title('Outcomes')
    plt.xlabel('Batch')
    plt.ylabel('Games won by P1\n(out of {})'.format(temp.shape[1]))
    plt.plot(results);
    return

def print_error(errors):
    err = np.array([errors[i][-1].cpu().item() for i in range(errors.shape[0])])
    print('Lowest error during training: {:.5f}'.format(np.min(err)))
    print('Error at training end: {:.5f}'.format(err[-1]))
    plt.figure(figsize=(15, 12))
    #plt.xticks(np.arange(0, errors.shape[0], step=1))
    plt.title('Errors')
    plt.xlabel('Batch')
    plt.ylabel('Error')
    plt.plot(err);
    return

if __name__ == "__main__":
    
    import game as g
    import pickle
    
    # Parameters
    n_batches = [1]
    n_reps_per_batch = [1]
    networks = [[50, 50, 50, 50, 50]]
    n_iterations = [10]
    learning_rates = [10**-3]
    epsilon = 1
    epsilon_decay_factors = [0.999]
    
    # Results storage
    all_parameters = []
    all_Qnets = []
    all_outcomes = []
    all_errors = []
    
    # Training
    print('Starting training')
    for i in n_batches:
        for j in n_reps_per_batch:
            for k in networks:
                for l in n_iterations:
                    for m in learning_rates:
                        for n in epsilon_decay_factors:
                            # Train
                            Qnet, outcomes, errors = train_Qnet(g.valid_moves, g.make_move, g.box_created, i, j, k, l, m, n)
                            # Collate parameters
                            all_parameters.append([i, j, k, l, m, n])
                            all_Qnets.append(Qnet)
                            all_outcomes.append(outcomes)
                            all_errors.append(errors)
                            # Save to pickle file
                            pickle.dump(all_parameters, open('all_parameters.p', 'wb'))
                            pickle.dump(all_Qnets, open('all_Qnets.p', 'wb'))
                            pickle.dump(all_outcomes, open('all_outcomes.p', 'wb'))
                            pickle.dump(all_errors, open('all_errors.p', 'wb'))
                            print()