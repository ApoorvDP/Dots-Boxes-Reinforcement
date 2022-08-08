import sys, os, time, pickle, numpy as np, matplotlib.pyplot as plt, random, json
import dqn, mlutils as ml, game as g

def get_move_heuristic(state, moves):
    boxes_edges_index = g.boxes_edges_index_generate() # Generate index of all edges for all boxes
    boxes_edges, boxes_3s, boxes_2s = [], [], []
    for box_edges in boxes_edges_index: # Generate binary 0/1 of whether each edge in index exists in state
        boxes_edges.append([state[i] for i in box_edges])
    for i in range(len(boxes_edges)):
        if sum(boxes_edges[i]) == 3: # Find boxes with 3 edges completed
            boxes_3s.append(i)
        if sum(boxes_edges[i]) == 2: # Find boxes with 2 edges completed
            boxes_2s.append(i)
    moves_3s = [boxes_edges_index[box_3s][boxes_edges[box_3s].index(0)] for box_3s in boxes_3s] # List all available moves in boxes with 3 edges complete
    moves_2s = list(set([i for x in [[boxes_edges_index[i][j] for j in list(np.where(np.array(boxes_edges[i]) == 0)[0])] for i in boxes_2s] for i in x])) # List all available moves in boxes with 2 edges complete
    if len(boxes_3s) > 0: # If there are boxes with 3 edges complete, choose a move from those
        move = random.choice(moves_3s)
    elif len(boxes_2s) > 0: # If there are boxes with 2 edges complete, try not to choose a move from those
        avail_moves = [m for m in moves if m not in moves_2s] # Check if there are available moves to not make a 3rd edge to a 2-edge box
        move = random.choice(avail_moves) if len(avail_moves) > 0 else random.choice(moves)
    else: # Else choose a move at random
        move = random.choice(moves)
    return move

def opponent_move(state, moves, Qnet):
    if Qnet is not None:
        Qs = [Qnet.evaluate(np.array(state + ml.move_to_onehot_24(m)))[0] if Qnet.processed is True else 0 for m in moves] # Q values for Qnet_adv
        move = moves[np.argmax(Qs)]
    else:
        move = get_move_heuristic(state, moves)
    return move

def epsilon_greedy(state, valid_moves_F, Qnet, Qnet_adv, epsilon, turn):
    moves = valid_moves_F(state)
    Qs = [Qnet.evaluate(np.array(state + ml.move_to_onehot_24(m)))[0] if Qnet.processed is True else 0 for m in moves] # Q values for Qnet
    if np.random.uniform() < epsilon: # Random move
        move = moves[random.sample(range(len(moves)), 1)[0]] if not turn else opponent_move(state, moves, Qnet_adv)
    else: # Greedy move
        move = moves[np.argmax(Qs)] if not turn else opponent_move(state, moves, Qnet_adv)
    return move, np.argmax(Qs) if not turn else 0 # 0 because don't care about opponent's Q value prediction

def compute_results(epochs, outcomes, bin_size):
    temp, x = outcomes.reshape(epochs, -1), []
    for i in range(temp.shape[0]):
        result = np.sum(temp[i] == 1) # Aggregating results per epoch
        x.append(result)
    y, s = [0]*len(x), 0
    for i in range(bin_size, len(x)):
        s = sum(x[i - bin_size:i])
        y[i] = 100*s/(bin_size*temp.shape[1]) # Aggregating results for (bin_size)-latest epochs
    return y

def plot_results(results_path, experiment_no, run, win_rate, bin_size):
    plt.figure(figsize=(10, 8))
    plt.gca().set(title=f'Highest win rate during training: {np.max(win_rate)}\nWin rate at training end: {win_rate[-1]}', xlabel='Epoch', ylabel='Averaged Win %', ylim=(50, 100)) # Set standard range for y
    plt.plot(range(bin_size, bin_size+len(win_rate)), win_rate)
    plt.savefig(os.path.join(results_path, f'{experiment_no}. Results - Run {run}.png'))
    return

def compute_epsilons(epochs, min_epsilon, epsilon_decay_factor):
    epsilons, ep = [], 1
    for epoch in range(epochs):
        if epoch > 0:
            ep *= epsilon_decay_factor
        epsilon = max(min_epsilon, ep)
        epsilons.append(epsilon) # Storing epsilons
    return epsilons

def plot_epsilons(results_path, experiment_no, epsilons, min_epsilon):
    plt.figure(figsize=(10, 8))
    plt.gca().set(title='Epsilons', xlabel='Epoch', ylabel='Epsilon', ylim=(0, 1))
    plt.plot(epsilons)
    plt.axhline(y=min_epsilon, color='r', linestyle='--') # Plotting minimum randomness visual threshold
    plt.savefig(os.path.join(results_path, f'{experiment_no}. Epsilons.png'))
    return

def plot_wins(results_path, experiment_no, runs, win_rates, parameters, bin_size):
    plt.figure(figsize=(10, 8))
    plt.gca().set(title=f'Parameters: {str(list(parameters.items())[:len(list(parameters.items()))//3])}\n{str(list(parameters.items())[len(list(parameters.items()))//3:(2*len(list(parameters.items())))//3])}\n{str(list(parameters.items())[(2*len(list(parameters.items())))//3:])}\nRuns: {runs}\n', xlabel='Epoch', ylabel='Win % Range', ylim=(50, 100)) # Set standard range for y
    plt.plot(range(bin_size, bin_size+win_rates.shape[1]), np.mean(win_rates, axis=0)) # Plotting win rate averaged over 5 runs
    plt.fill_between(range(bin_size, bin_size+win_rates.shape[1]), np.min(win_rates, axis=0), np.max(win_rates, axis=0), color='orange', alpha=0.3) # Plotting minimum and maximum values for individual runs
    plt.savefig(os.path.join(results_path, f'{experiment_no}. Wins.png'))
    return

def plot_testing(results_path, testing_results):
    plt.figure(figsize=(10, 8))
    plt.gca().set(title='Testing Results', xlabel='Test no.', ylabel='Win %', ylim=(50, 100)) # Set standard range for y
    plt.plot(range(1, 1+len(testing_results)), testing_results)
    plt.savefig(os.path.join(results_path, 'Testing.png'))
    return

if __name__ == "__main__":
    
    # Controls
    with open('config.json') as f:
        config = json.load(f)
    results_path, adv_path, Qnet_adv = os.path.join(os.getcwd(), 'Results'), os.path.join(os.getcwd(), 'Adversaries'), None
    if config['adversary'] == True: # Load Q-network as adversary
        with open(os.path.join(adv_path, 'Qnet_adv.pt'), 'rb') as f:
            while True:
                try:
                    Qnet_adv = pickle.load(f).after_load_model()
                except EOFError:
                    break
    
    # Reinforcement Learning algorithm
    if config['algorithm'] == 'dqn': # Deep Q-learning
        if config['train'] == True: # Training
            experiment_no, runs, bin_size = int(sys.argv[1]), 5, 10 # Meta parameters
            with open('parameters.json') as f:
                parameters = json.load(f) # Parameters
                parameters['inputs'], parameters['outputs'] = 4*(2*config['board']*(config['board']-1)), 1 # Edges in board: 2n(n-1)
            win_rates = []
            for i in range(runs): # Multiple runs for a set of parameters
                pickle.dump({'inputs': parameters['inputs'], 'outputs': parameters['outputs'], 'runs': runs, 'bin_size': bin_size}, open(os.path.join(results_path, f'{experiment_no}. Metadata.pth'), 'wb')) # Save metadata
                print(f'Training => Run: {i+1}')
                start_time = time.time()
                Qnet, outcomes = dqn.train_DQN(epsilon_greedy, g.valid_moves, g.make_move, g.box_created, parameters, Qnet_adv, verbose=True)
                train_time = round(time.time()-start_time, 2)
                Qnet.training_time = train_time
                print(f'Trained => Time to train: {train_time} seconds')
                pickle.dump(parameters, open(os.path.join(results_path, f'{experiment_no}. Parameters.pth'), 'wb')) # Save parameters
                pickle.dump(Qnet, open(os.path.join(results_path, f'{experiment_no}. Qnet - Run {i+1}.pt'), 'wb')) # Save network
                win_rate = compute_results(parameters['epochs'], outcomes, bin_size)[bin_size:]
                plot_results(results_path, experiment_no, i+1, win_rate, bin_size)
                epsilons = compute_epsilons(parameters['epochs'], parameters['min_epsilon'], parameters['epsilon_decay_factor'])
                plot_epsilons(results_path, experiment_no, epsilons, parameters['min_epsilon'])
                pickle.dump({'train_time': train_time, 'win_rate': win_rate, 'epsilons': epsilons}, open(os.path.join(results_path, f'{experiment_no}. Outputs - Run {i+1}.pth'), 'wb')) # Save outputs
                win_rates.append(win_rate)
            plot_wins(results_path, experiment_no, runs, np.array(win_rates).reshape(runs, -1), parameters, bin_size)
        else: # Testing
            num_tests, runs, num_games = int(sys.argv[1]), 5, 100 # Meta parameters
            with open(os.path.join(results_path, 'Qnet_best.pt'), 'rb') as f:
                while True:
                    try:
                        Qnet_best = pickle.load(f).after_load_model()
                    except EOFError:
                        break
            testing_results = dqn.test_DQN(epsilon_greedy, g.valid_moves, g.make_move, g.box_created, num_tests, runs, num_games, Qnet_best, Qnet_adv, verbose=True)
            plot_testing(results_path, testing_results)
    
    # Play
    if config['play'] == True:
        g.play()

