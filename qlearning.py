import sys, os, time, pickle, numpy as np, matplotlib.pyplot as plt, random, json
import neuralnetworks as nn, mlutils as ml, game as g

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

def train_Qnet(epsilon_greedy_F, valid_moves_F, make_move_F, boxes_created_F, parameters, Qnet_adv, ep=1, verbose=False):
    Qnet = nn.FCNN(False, 48, parameters['network'], 1, parameters['use_ReLU'])
    outcomes, repk = np.zeros(parameters['epochs']*parameters['games_per_epoch']), -1
    for epoch in range(parameters['epochs']):
        start_time = time.time()
        if epoch > 0:
            ep *= parameters['epsilon_decay_factor'] # Degree of randomness of move as a fraction => by default ep=1 for epoch 0, i.e. 100% random move
        samples, epsilon = [], max(parameters['min_epsilon'], ep) # Minimum probability of a random move is min_epsilon
        for reps in range(parameters['games_per_epoch']):
            repk += 1
            # Initialize game
            state, boxes, score, done = [0]*24, [0]*9, 0, False
            # Start game; player 1's turn initially
            turn, state_p = True, None # state_p tells if there is a previous state to use as sample
            move, _ = epsilon_greedy_F(state, valid_moves_F, Qnet, Qnet_adv, epsilon, turn)
            # Play game
            while not done:
                if not turn: # Store agent state and move for retroactively appending to sample when turn switches back to player 1
                    state_p, move_p = state.copy(), move
                r = 0 # Default reinforcement for a turn is 0
                state_next = make_move_F(state, move)
                created, boxes = boxes_created_F(state_next, boxes) # Check how many boxes are created after making a move
                if created > 0: # If a box is created, update score of the player who made the move
                    if not turn:
                        score += created
                else: # Else, give turn to the other player
                    turn = not turn
                if 0 not in state_next: # If there are no more edges remaining, the game is over
                    r = score # Reinforcement is the number of boxes, i.e. the score of the player
                    outcomes[repk], done = 1 if score > 4 else -1, True # Store game outcome and set termination flag
                    move_next, Qnext = -1, 0 # Qnext value will be zero at end of game
                else: # Else, determine next move and add current sample with reinforcement 0
                    move_next, Qnext = epsilon_greedy_F(state_next, valid_moves_F, Qnet, Qnet_adv, epsilon, turn)
                # Note to future self: The if condition below is CORRECT; Do NOT confuse it again
                if (created > 0 and not turn) or (created == 0 and not turn and state_p is not None) or done: # Add player 2 turns and game end state
                    samples.append([*state_p, *ml.move_to_onehot_24(move_p), r, Qnext]) # Collect not turn results as a sample
                state, move = state_next, move_next
        samples = np.array(samples) # Samples contains the training inputs and the targets
        X, T = samples[:, :48], samples[:, 48:49]+samples[:, 49:50] # Training inputs and target values for the neural network
        Qnet, _ = Qnet.train(X, T, parameters['updates_per_epoch'], X.shape[0], parameters['learning_rate'], parameters['use_SGD']) # Training the neural network
        if verbose:
            #print(f'(Epoch: {epoch+1}, Wins: {len(np.where(outcomes.reshape(-1, parameters["games_per_epoch"]) == 1)[0])}, Losses: {len(np.where(outcomes.reshape(-1, parameters["games_per_epoch"]) == -1)[0])}')
            print(f'(Epoch: {epoch+1}, Win %: {(round(outcomes.reshape(-1, parameters["games_per_epoch"])[epoch, :].mean(), 4)+1)*100/2:.2f}, Epsilon: {epsilon:.2f}), Time taken: {time.time() - start_time:.2f}s')
    return Qnet.before_save_model(), outcomes

def test_Qnet(epsilon_greedy_F, valid_moves_F, make_move_F, boxes_created_F, num_tests, runs, num_games, Qnet, Qnet_adv, ep=0, verbose=False):
    testing = []
    for i in range(num_tests):
        print(f'Testing => Test {i+1}')
        percentages = []
        for j in range(runs):
            outcomes = []
            for k in range(num_games):
                state, boxes, score, done = [0]*24, [0]*9, 0, False
                turn = True
                move, _ = epsilon_greedy_F(state, valid_moves_F, Qnet, Qnet_adv, ep, turn)
                while not done:
                    state_next = make_move_F(state, move)
                    created, boxes = boxes_created_F(state_next, boxes)
                    if created > 0: # If a box is created, update score of the player who made the move
                        if not turn:
                            score += created
                    else:
                        turn = not turn
                    if 0 not in state_next:
                        outcome, done = 1 if score > 4 else -1, True
                        move_next, Qnext = -1, 0
                    else:
                        move_next, Qnext = epsilon_greedy_F(state_next, valid_moves_F, Qnet, Qnet_adv, ep, turn)
                    state, move = state_next, move_next
                outcomes.append(outcome)
            percentages.append(sum(outcomes[k] == -1 for k in range(len(outcomes)))/num_games*100)
        testing.append(sum(percentages)/runs)
        if verbose:
            print(f'Finished test {i+1}; Percent: {testing[-1]}')
    return testing

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
    plt.gca().set(title=f'Parameters: {str(list(parameters.items())[:len(list(parameters.items()))//2])}\n{str(list(parameters.items())[len(list(parameters.items()))//2:])}\nRuns: {runs}\n', xlabel='Epoch', ylabel='Win % Range', ylim=(50, 100)) # Set standard range for y
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
    
    # Q-learning
    if config['train'] == True: # Training
        experiment_no, runs, bin_size = int(sys.argv[1]), 5, 10 # Meta parameters
        with open('parameters.json') as f:
            parameters = json.load(f) # Parameters
        win_rates = []
        for i in range(runs): # Multiple runs for a set of parameters
            pickle.dump({'runs': runs, 'bin_size': bin_size}, open(os.path.join(results_path, f'{experiment_no}. Metadata.meta'), 'wb')) # Save metadata
            print(f'Training => Run: {i+1}')
            start_time = time.time()
            Qnet, outcomes = train_Qnet(epsilon_greedy, g.valid_moves, g.make_move, g.box_created, parameters, Qnet_adv, verbose=True)
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
        testing_results = test_Qnet(epsilon_greedy, g.valid_moves, g.make_move, g.box_created, num_tests, runs, num_games, Qnet_best, Qnet_adv, verbose=True)
        plot_testing(results_path, testing_results)
    
    # Play
    if config['play'] == True:
        g.play()

