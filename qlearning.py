import matplotlib.pyplot as plt
import random
import numpy as np
import neuralnetworks as nn
#import mlutils as ml # Import when plotting network weights, etc.; mlutils.py currently in progress

def epsilon_greedy(state, valid_moves_F, Qnet, epsilon, turn):
    moves = valid_moves_F(state)
    if np.random.uniform() < epsilon: # Random move
        move = moves[random.sample(range(len(moves)), 1)[0]] if turn else moves[0] # Dumb opponent
        #move = moves[random.sample(range(len(moves)), 1)[0]] # Random opponent
        Q = Qnet.use_pytorch(np.array(state + [move])) if Qnet.processed is True else 0
    else: # Greedy move
        Qs = [Qnet.use_pytorch(np.array(state + [m/23])) if Qnet.processed is True else 0 for m in moves]
        move = moves[np.argmax(Qs)] if turn else moves[0] # Dumb opponent
        #move = moves[np.argmax(Qs)] if turn else moves[random.sample(range(len(moves)), 1)[0]] # Random opponent
        Q = np.max(Qs) if turn else Qnet.use_pytorch(np.array(state + [move])) if Qnet.processed is True else 0
    return move, Q

def train_Qnet(valid_moves_F, make_move_F, boxes_created_F, n_epochs, n_reps_per_epoch, network, batch_size, learning_rate, epsilon_decay_factor, use_ReLU=False, use_SGD=False):
    Qnet, outcomes = nn.NN(False, 25, network, 1, use_ReLU), np.zeros(n_epochs*n_reps_per_epoch)
    repk, epsilon = -1, 1 # Degree of randomness of move as a fraction => initially 1, i.e. 100% random move
    for epoch in range(n_epochs):
        if epoch > 0:
            epsilon *= epsilon_decay_factor
            epsilon = max(0.01, epsilon) # Minimum probability of a random move is 0.01
        samples = []
        for reps in range(n_reps_per_epoch):
            repk += 1
            # Initialize game
            state, boxes, score, done = [0]*24, [0]*9, [0]*2, False
            # Start game; player 1's turn initially
            turn = True
            move, _ = epsilon_greedy(state, valid_moves_F, Qnet, epsilon, turn)
            # Continue to play the game
            while not done:
                r = 0 # Default reinforcement for a turn is 0
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
                    r = 1 if score[0] > score[1] else -1 # Determine the reinforcement
                    outcomes[repk], done = r, True # Store game outcome and set termination flag
                    Qnext, move_next = 0, -1 # Qnext value will be zero at end of game
                else: # Else, determine next move and add current sample with reinforcement 0
                    move_next, Qnext = epsilon_greedy(state_next, valid_moves_F, Qnet, epsilon, turn)
                if (created > 0 and turn) or (created == 0 and not turn): # Add player 1 turns only
                    samples.append([*state, move, r, Qnext]) # Collect turn results as a sample
                state, move = state_next, move_next
        samples = np.array(samples) # Samples contains the training inputs and the targets
        X, T = samples[:, :25], samples[:, 25:26]+samples[:, 26:27] # Training inputs and target values for the neural network
        Qnet, _ = Qnet.train_pytorch(X, T, n_reps_per_epoch, batch_size, learning_rate, use_SGD) # Training the neural network
        print(f'Epoch {epoch+1} done')
    print('TRAINED')
    return Qnet, outcomes

def compute_results(outcomes, n_epochs):
    temp, x = outcomes.reshape(n_epochs, -1), []
    for i in range(temp.shape[0]):
        result = np.sum(temp[i] == 1) # Aggregating results per epoch
        x.append(result)
    y, s = [0]*len(x), 0
    for i in range(len(x)):
        s += x[i]
        y[i] = 100*s/((i+1)*temp.shape[1])
    return y

def plot_results(win_rate, experiment_no, results_path):
    plt.figure(figsize=(12, 9))
    plt.gca().set(title='Outcomes', xlabel='Epoch', ylabel='Averaged Win %')
    plt.plot(win_rate)
    plt.savefig(os.path.join(results_path, f'{experiment_no}. Results.png'))
    return

def plot_epsilons(epsilon_decay_factor, n_epochs, experiment_no, results_path):
    epsilons, epsilon = [], 1
    for epoch in range(n_epochs):
        epsilon *= epsilon_decay_factor
        epsilon = max(0.01, epsilon)
        epsilons.append(epsilon)
    plt.figure(figsize=(12, 9))
    plt.gca().set(title='Epsilons', xlabel='Epoch', ylabel='Epsilon')
    plt.plot(epsilons)
    plt.savefig(os.path.join(results_path, f'{experiment_no}. Epsilons.png'))
    return

if __name__ == "__main__":
    
    import sys, os, time, pickle
    import game as g
    
    experiment_no, results_path, reuse = int(sys.argv[1]), os.getcwd()+'/Results', False
    
    # Parameters
    n_epochs = [400]
    n_reps_per_epoch = [1000]
    networks = [[20, 20, 20]]
    batch_sizes = [50]
    learning_rates = [10**-4]
    epsilon_decay_factors = [0.97]
    use_SGD = True
    
    if not reuse:
        # Training
        print('Starting training')
        for i in range(len(n_epochs)):
            start_time = time.time()
            Qnet, outcomes = train_Qnet(g.valid_moves, g.make_move, g.box_created, n_epochs[i], n_reps_per_epoch[i], networks[i], batch_sizes[i], learning_rates[i], epsilon_decay_factors[i], use_SGD)
            # Outputs
            train_time = round(time.time()-start_time, 2)
            print(f'Time to train: {train_time} seconds\n')
            pickle.dump({'n_epochs': n_epochs[i], 'n_reps_per_epoch': n_reps_per_epoch[i], 'network': networks[i], 'batch_size': batch_sizes[i], 'learning_rate': learning_rates[i], 'epsilon_decay_factor': epsilon_decay_factors[i], 'use_SGD': use_SGD}, open(os.path.join(results_path, f'{experiment_no}. Parameters.pth'), 'wb'))
            pickle.dump(Qnet, open(os.path.join(results_path, f'{experiment_no}. Qnet.pt'), 'wb'))
            win_rate = compute_results(outcomes, n_epochs[i])
            print(f'Highest win rate during training: {np.max(win_rate)}\nWin rate at training end: {win_rate[-1]}')
            plot_results(win_rate, experiment_no, results_path)
            pickle.dump({'train_time': train_time, 'win_rate': win_rate}, open(os.path.join(results_path, f'{experiment_no}. Outputs.pth'), 'wb'))
            plot_epsilons(epsilon_decay_factors[i], n_epochs[i], experiment_no, results_path)
            experiment_no += 1
    else:
        pass # Reuse to be coded
