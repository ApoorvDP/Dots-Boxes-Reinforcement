import numpy as np, matplotlib.pyplot as plt, random
import neuralnetworks as nn, mlutils as ml # mlutils currently in progress

def move_to_onehot(move):
    onehot_move = [0]*24
    onehot_move[move] = 1
    return onehot_move

def epsilon_greedy(state, valid_moves_F, Qnet, epsilon, turn):
    moves = valid_moves_F(state)
    if np.random.uniform() < epsilon: # Random move
        #move = moves[random.sample(range(len(moves)), 1)[0]] if turn else moves[0] # Dumb opponent
        move = moves[random.sample(range(len(moves)), 1)[0]] # Random opponent
    else: # Greedy move
        Qs = [Qnet.use_pytorch(np.array(state + move_to_onehot(m)))[0] if Qnet.processed is True else 0 for m in moves] # Q values for deciding greedy move
        #move = moves[np.argmax(Qs)] if turn else moves[0] # Dumb opponent
        move = moves[np.argmax(Qs)] if turn else moves[random.sample(range(len(moves)), 1)[0]] # Random opponent
    #Q = Qnet.use_pytorch(np.array(state + move_to_onehot(move)))[0] if Qnet.processed is True else 0
    return move, Qnet.use_pytorch(np.array(state + move_to_onehot(move)))[0] if Qnet.processed is True else 0

def train_Qnet(valid_moves_F, make_move_F, boxes_created_F, parameters, use_ReLU=False, verbose=False):
    Qnet, outcomes = nn.NN(False, 48, parameters['network'], 1, use_ReLU), np.zeros(parameters['n_epochs']*parameters['n_reps_per_epoch'])
    repk, epsilon = -1, 1 # Degree of randomness of move as a fraction => initially 1, i.e. 100% random move
    for epoch in range(parameters['n_epochs']):
        if epoch > 0:
            epsilon *= parameters['epsilon_decay_factor']
            epsilon = max(parameters['min_epsilon'], epsilon) # Minimum probability of a random move is min_epsilon
        samples = []
        for reps in range(parameters['n_reps_per_epoch']):
            repk += 1
            # Initialize game
            state, boxes, score, done = [0]*24, [0]*9, [0]*2, False
            # Start game; player 1's turn initially
            turn = True
            move, _ = epsilon_greedy(state, valid_moves_F, Qnet, epsilon, turn)
            # Play game
            while not done:
                if turn: # Store player 1 state and move for retroactively appending to sample when turn switches back to player 1
                    state_p1, move_p1 = state.copy(), move
                r = 0 # Default reinforcement for a turn is 0
                state_next = make_move_F(state, move)
                created, boxes = boxes_created_F(state_next, boxes) # Check how many boxes are created after making a move
                if created > 0: # If a box is created, update score of the player who made the move
                    if turn:
                        score[0] += created
                    else:
                        score[1] += created
                else: # Else, give turn to the other player
                    turn = not turn
                if 0 not in state_next: # If there are no more edges remaining, the game is over
                    r = 1 if score[0] > score[1] else -1 # Determine the reinforcement
                    outcomes[repk], done = r, True # Store game outcome and set termination flag
                    move_next, Qnext = -1, 0 # Qnext value will be zero at end of game
                else: # Else, determine next move and add current sample with reinforcement 0
                    move_next, Qnext = epsilon_greedy(state_next, valid_moves_F, Qnet, epsilon, turn)
                if (created > 0 and turn) or (created == 0 and turn) or done: # Add player 1 turns and game end state
                    samples.append([*state_p1, *move_to_onehot(move_p1), r, Qnext]) # Collect turn results as a sample
                state, move = state_next, move_next
        samples = np.array(samples) # Samples contains the training inputs and the targets
        X, T = samples[:, :48], samples[:, 48:49]+samples[:, 49:50] # Training inputs and target values for the neural network
        Qnet, _ = Qnet.train_pytorch(X, T, parameters['n_trains'], X.shape[0], parameters['learning_rate'], parameters['use_SGD']) # Training the neural network
        if verbose:
            print(f'(Epoch: {epoch+1}, Mean Outcome: {outcomes.reshape(-1, parameters["n_reps_per_epoch"])[epoch, :].mean():.2f}, Epsilon: {epsilon:.2f})')
    return Qnet, outcomes

def compute_results(n_epochs, outcomes, bin_size):
    temp, x = outcomes.reshape(n_epochs, -1), []
    for i in range(temp.shape[0]):
        result = np.sum(temp[i] == 1) # Aggregating results per epoch
        x.append(result)
    y, s = [0]*len(x), 0
    for i in range(bin_size, len(x)):
        s = sum(x[i - bin_size:i])
        y[i] = 100*s/(bin_size*temp.shape[1]) # Aggregating results for (bin_size)-latest epochs
    return y

def plot_results(experiment_no, run, win_rate, bin_size, results_path):
    plt.figure(figsize=(12, 9))
    plt.gca().set(title=f'Highest win rate during training: {np.max(win_rate)}\nWin rate at training end: {win_rate[-1]}', xlabel='Epoch', ylabel='Averaged Win %', ylim=(50, 100)) # Set standard range for y
    plt.plot(range(bin_size, bin_size+len(win_rate)), win_rate)
    plt.savefig(os.path.join(results_path, f'{experiment_no}. Results - Run {run}.png'))
    return

def compute_epsilons(n_epochs, min_epsilon, epsilon_decay_factor):
    epsilons, epsilon = [], 1
    for epoch in range(n_epochs):
        epsilon *= epsilon_decay_factor
        epsilon = max(min_epsilon, epsilon)
        epsilons.append(epsilon) # Storing epsilons
    return epsilons

def plot_epsilons(experiment_no, epsilons, min_epsilon, results_path):
    plt.figure(figsize=(12, 9))
    plt.gca().set(title='Epsilons', xlabel='Epoch', ylabel='% chance of a random move')
    plt.plot(epsilons)
    plt.axhline(y=min_epsilon, color='r', linestyle='--') # Plotting minimum randomness visual threshold
    plt.savefig(os.path.join(results_path, f'{experiment_no}. Epsilons.png'))
    return

def plot_wins(experiment_no, runs, win_rates, parameters, bin_size, results_path):
    plt.figure(figsize=(12, 9))
    plt.gca().set(title=f'Parameters: {parameters}\nRuns: {runs}', xlabel='Epoch', ylabel='Win % Range', ylim=(50, 100)) # Set standard range for y
    plt.plot(range(bin_size, bin_size+win_rates.shape[1]), np.mean(win_rates, axis=0)) # Plotting win rate averaged over 5 runs
    plt.fill_between(range(bin_size, bin_size+win_rates.shape[1]), np.min(win_rates, axis=0), np.max(win_rates, axis=0), color='orange', alpha=0.3) # Plotting minimum and maximum values for individual runs
    plt.savefig(os.path.join(results_path, f'{experiment_no}. Wins.png'))
    return

if __name__ == "__main__":
    
    import sys, os, time, pickle
    import game as g
    
    # Control flags
    reuse, store, play = False, True, False
    
    # Metadata
    experiment_no, results_path, runs, bin_size = int(sys.argv[1]), os.getcwd()+'/Results', 5, 10
    
    # Parameters
    parameters = {'n_epochs': 50, 'n_reps_per_epoch': 200, 'network': [100, 100], 'n_trains': 100, 'learning_rate': 0.001, 'min_epsilon': 0.02, 'epsilon_decay_factor': 0.92, 'use_ReLU': False, 'use_SGD': True}
    
    # Q-learning
    if not reuse: # Training
        win_rates = []
        for i in range(runs): # Multiple runs for a set of parameters
            pickle.dump({'runs': runs, 'bin_size': bin_size}, open(os.path.join(results_path, f'{experiment_no}. Metadata.meta'), 'wb')) # Save metadata
            print(f'Training => Run: {i+1}')
            start_time = time.time()
            Qnet, outcomes = train_Qnet(g.valid_moves, g.make_move, g.box_created, parameters, verbose=True)
            train_time = round(time.time()-start_time, 2)
            print(f'Trained => Time to train: {train_time} seconds')
            pickle.dump(parameters, open(os.path.join(results_path, f'{experiment_no}. Parameters.pth'), 'wb')) # Save parameters
            pickle.dump(Qnet, open(os.path.join(results_path, f'{experiment_no}. Qnet - Run {i+1}.pt'), 'wb')) # Save network
            win_rate = compute_results(parameters['n_epochs'], outcomes, bin_size)[bin_size:]
            plot_results(experiment_no, i+1, win_rate, bin_size, results_path)
            epsilons = compute_epsilons(parameters['n_epochs'], parameters['min_epsilon'], parameters['epsilon_decay_factor'])
            plot_epsilons(experiment_no, epsilons, parameters['min_epsilon'], results_path)
            pickle.dump({'train_time': train_time, 'win_rate': win_rate, 'epsilons': epsilons}, open(os.path.join(results_path, f'{experiment_no}. Outputs - Run {i+1}.pth'), 'wb')) # Save outputs
            win_rates.append(win_rate)
        plot_wins(experiment_no, runs, np.array(win_rates).reshape(runs, -1), parameters, bin_size, results_path)
    else: # Testing
        pass # Reuse to be coded
    
    # Interactive game
    if play:
        pass #g.play_game() # Starting game
