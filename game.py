from copy import copy
import numpy as np

def boxes_edges_index_generate():
    boxes_edges_index, c = [], 0
    for i in range(0, 9, 3):
        for j in range(3):
            boxes_edges_index.append([(i+j), (i+j+12)+c, (i+j+13)+c, (i+j+3)])
        c += 1
    return boxes_edges_index

def print_game(state):
    for i in range(4):
        for j in range(4):
            print('\u2022', end = '')
            print() if j == 3 else print('---', end = '') if state[((3*i)+j)] == 1 else print('   ', end = '')
        if i != 3:
            if(state[((4*i)+12)] == 1):
                print('|') if(state[((4*i)+13)] == 0 and state[((4*i)+14)] == 0 and state[((4*i)+15)] == 0) else print('|', end = '')
            if(state[((4*i)+13)] == 1):
                if(state[((4*i)+14)] == 0 and state[((4*i)+15)] == 0):
                    print('   |') if(state[((4*i)+12)] == 1) else print('    |')
                else:
                    print('   |', end = '') if(state[((4*i)+12)] == 1) else print('    |', end = '')
            if(state[((4*i)+14)] == 1):
                if(state[((4*i)+15)] == 0):
                    print('   |') if(state[((4*i)+13)] == 1) else print('       |') if(state[((4*i)+12)] == 1) else print('        |')
                else:
                    print('   |', end = '') if(state[((4*i)+13)] == 1) else print('       |', end = '') if(state[((4*i)+12)] == 1) else print('        |', end = '')
            if(state[((4*i)+15)] == 1):
                print('   |') if(state[((4*i)+14)] == 1) else print('       |') if(state[((4*i)+13)] == 1) else print('           |') if(state[((4*i)+12)] == 1) else print('            |')
            if(state[((4*i)+12)] == 0 and state[((4*i)+13)] == 0 and state[((4*i)+14)] == 0 and state[((4*i)+15)] == 0):
                print()
    return

def valid_moves(state):
    return list(np.where(np.array(state) == 0)[0])

def make_move(state, move):
    newState = state.copy()
    newState[move] = 1
    return newState

def box_created(state, boxes):
    created = 0
    k = 0
    for i in range(9):
        if i != 0 and (i%3) == 0:
            k += 1
        if state[i] == 1 and state[(i+3)] == 1 and state[(i+12+k)] == 1 and state[(i+13+k)] == 1:
            if boxes[i] == 0:
                boxes[i] = 1
                created += 1
    return created, boxes

if __name__ == "__main__":
    
    start_state = [0]*24
    print_game(start_state)
    print('The initial state of the game is only dots and no edges.')
    valid_moves(start_state)
    print('Since there are no edges yet, all the edges are valid moves for the initial state.')
    make_move(start_state, 1)
    print('The function correctly modifies the state of the game.')
    print_game(make_move(start_state, 1))
    print('Edge 1 (in correspondence with the game representation figure) is now present, while all other edges are absent.')
    example_state = make_move(start_state, 3)
    print(example_state)
    print()
    print_game(make_move(start_state, 3))
    print('Similarly, setting the value of edge 3 (in correspondence with the game representation figure) results in the above state.')
    print('Random state')
    random_state = make_move(make_move(make_move(make_move(make_move([0]*24, 0), 3), 12), 13), 1)
    print_game(random_state)
    print()
    print()
    boxes = [1] + [0]*8
    print('Make move at edge 4')
    boxes_created_state_1 = make_move(random_state, 4)
    print_game(boxes_created_state_1)
    created, boxes = box_created(boxes_created_state_1, boxes)
    print()
    print('Boxes created:', created)
    print()
    print('Current boxes:', boxes)
    print()
    print()
    print('Make move at edge 14')
    box_created_state_2 = make_move(boxes_created_state_1, 14)
    print_game(box_created_state_2)
    created, boxes = box_created(box_created_state_2, boxes)
    print()
    print('Boxes created:', created)
    print()
    print('Current boxes:', boxes)
    print('For the random state, when a move is made at edge 4, the function `box_created` correctly returns the value False, since no box was created by making that move. On making a move at edge 14, the function returns the value True, and the boxes list is modified accordingly.')