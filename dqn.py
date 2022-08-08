import time, numpy as np
import neuralnetworks as nn, mlutils as ml

def train_DQN(epsilon_greedy_F, valid_moves_F, make_move_F, boxes_created_F, parameters, Qnet_adv, ep=1, verbose=False):
    Qnet = nn.FCNN(False, parameters['inputs'], parameters['network'], parameters['outputs'], parameters['use_ReLU'])
    if verbose:
        print(f'Neural network\n{Qnet}\ncreated on {Qnet.device}.')
    board_edge_count, outcomes, repk = parameters['inputs']//2, np.zeros(parameters['epochs']*parameters['games_per_epoch']), -1
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
            move, _ = epsilon_greedy_F(board_edge_count, state, valid_moves_F, Qnet, Qnet_adv, epsilon, turn)
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
                    move_next, Qnext = epsilon_greedy_F(board_edge_count, state_next, valid_moves_F, Qnet, Qnet_adv, epsilon, turn)
                # Note to future self: The if condition below is CORRECT; Do NOT confuse it again
                if (created > 0 and not turn) or (created == 0 and not turn and state_p is not None) or done: # Add player 2 turns and game end state
                    samples.append([*state_p, *ml.move_to_onehot(move_p, board_edge_count), r, Qnext]) # Collect not turn results as a sample
                state, move = state_next, move_next
        samples = np.array(samples) # Samples contains the training inputs and the targets
        X, T = samples[:, :48], samples[:, 48:49]+samples[:, 49:50] # Training inputs and target values for the neural network
        Qnet, _ = Qnet.train(X, T, parameters['updates_per_epoch'], X.shape[0], parameters['learning_rate'], parameters['use_SGD']) # Training the neural network
        if verbose:
            #print(f'(Epoch: {epoch+1}, Wins: {len(np.where(outcomes.reshape(-1, parameters["games_per_epoch"]) == 1)[0])}, Losses: {len(np.where(outcomes.reshape(-1, parameters["games_per_epoch"]) == -1)[0])}')
            print(f'(Epoch: {epoch+1}, Win %: {(round(outcomes.reshape(-1, parameters["games_per_epoch"])[epoch, :].mean(), 4)+1)*100/2:.2f}, Epsilon: {epsilon:.2f}), Time taken: {time.time() - start_time:.2f}s')
    return Qnet.before_save_model(), outcomes

def test_DQN(epsilon_greedy_F, valid_moves_F, make_move_F, boxes_created_F, board_edge_count, num_tests, runs, num_games, Qnet, Qnet_adv, ep=0, verbose=False):
    testing = []
    for i in range(num_tests):
        print(f'Testing => Test {i+1}')
        percentages = []
        for j in range(runs):
            outcomes = []
            for k in range(num_games):
                state, boxes, score, done = [0]*24, [0]*9, 0, False
                turn = True
                move, _ = epsilon_greedy_F(board_edge_count, state, valid_moves_F, Qnet, Qnet_adv, ep, turn)
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
                        move_next, Qnext = epsilon_greedy_F(board_edge_count, state_next, valid_moves_F, Qnet, Qnet_adv, ep, turn)
                    state, move = state_next, move_next
                outcomes.append(outcome)
            percentages.append(sum(outcomes[k] == -1 for k in range(len(outcomes)))/num_games*100)
        testing.append(sum(percentages)/runs)
        if verbose:
            print(f'Finished test {i+1}; Percent: {testing[-1]}')
    return testing
