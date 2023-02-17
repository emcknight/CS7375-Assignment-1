import numpy as np
from copy import deepcopy
from collections import deque


# Creates a random beginning board for the 8 Puzzle Game
def random_begin_state():
    tmp = np.arange(9)
    np.random.shuffle(tmp)
    arr = np.reshape(tmp, [3, 3])
    arr = arr.tolist()
    return arr


def get_moves(puzzle, zero_index):
    i, j = zero_index
    moves = []
    # Up move
    if i > 0:
        up = deepcopy(puzzle)
        up[i][j], up[i - 1][j] = up[i - 1][j], up[i][j]
        moves.append((up, (i - 1, j)))
    # Down move
    if i < 2:
        down = deepcopy(puzzle)
        down[i][j], down[i + 1][j] = down[i + 1][j], down[i][j]
        moves.append((down, (i + 1, j)))
    # Left move
    if j > 0:
        left = deepcopy(puzzle)
        left[i][j], left[i][j - 1] = left[i][j - 1], left[i][j]
        moves.append((left, (i, j - 1)))
    # Right move
    if j < 2:
        right = deepcopy(puzzle)
        right[i][j], right[i][j + 1] = right[i][j + 1], right[i][j]
        moves.append((right, (i, j + 1)))
    return moves


# Breadth-First Search Function
def bfs(start, end):
    print("Solving 8 Puzzle Game using Breadth-First Search")
    # Initialize deque to emulate a queue and visited array
    queue = deque([(start, [])])  # State, Path
    visited = [start]
    limit = 0  # Limit the number of states to search
    # While loop to complete the search
    while len(queue) > 0:
        limit += 1  # Increase the number of moves checked
        # If 10,000 moves are checked, the program terminates.
        if limit > 10000:
            print("Path could not be found in under 10,000 moves. Terminating Breadth-First Search...")
            return []
        puzzle, path = queue.pop()  # Take last puzzle from the queue aka dequeue
        # If puzzle matches end, return the path the search took
        if puzzle == end:
            return path
        # Find the index of the 0 in the puzzle
        zero_index = None
        for i in range(3):
            for j in range(3):
                if puzzle[i][j] == 0:
                    zero_index = (i, j)
        # Get the potential moves given the current puzzle
        moves = get_moves(puzzle, zero_index)
        for move, move_index in moves:
            # Add the move to the queue if it hasn't been visited before
            if move not in visited:
                queue.appendleft((move, path + [move_index]))
                visited.append(move)
    return []

# Depth-First Search Function
def dfs(start, end):
    print("Solving 8 Puzzle Game using Depth-First Search")
    # Initialize deque to emulate a stack and visited array
    stack = deque([(start, [])])  # State, Path
    visited = [start]
    limit = 0  # Limit the number of states to search
    # While loop to complete the search
    while len(stack) > 0:
        limit += 1  # Increase the number of moves checked
        # If 10,000 moves are checked, the program terminates.
        if limit > 10000:
            print("Path could not be found in under 10,000 moves. Terminating Depth-First Search...")
            return []
        puzzle, path = stack.popleft()  # Take first puzzle from the stack aka pop
        # If puzzle matches end, return the path the search took
        if puzzle == end:
            return path
        # Find the index of the 0 in the puzzle
        zero_index = None
        for i in range(3):
            for j in range(3):
                if puzzle[i][j] == 0:
                    zero_index = (i, j)
        # Get the potential moves given the current puzzle
        moves = get_moves(puzzle, zero_index)
        for move, move_index in moves:
            # Add the move to the queue if it hasn't been visited before
            if move not in visited:
                stack.appendleft((move, path + [move_index]))
                visited.append(move)
    return []


def ucs(start, end):
    print("Solving 8 Puzzle Game using Uniform Cost Search")
    queue = deque([(start, [], 0)])  # State, Path, Depth
    visited = [start]
    limit = 0  # Limit the number of states to search
    while len(queue) > 0:
        limit += 1  # Increase the number of moves checked
        # If 10,000 moves are checked, the program terminates.
        if limit > 10000:
            print("Path could not be found in under 10,000 moves. Terminating Uniform Cost Search...")
            return []
        puzzle, path, depth = queue.pop()  # Take last puzzle from the queue aka dequeue
        # If puzzle matches end, return the path the search took
        if puzzle == end:
            return path
        # Find the index of the 0 in the puzzle
        zero_index = None
        for i in range(3):
            for j in range(3):
                if puzzle[i][j] == 0:
                    zero_index = (i, j)
        # Get the potential moves given the current puzzle
        moves = get_moves(puzzle, zero_index)
        for move, move_index in moves:
            # Add the move to the queue if it hasn't been visited before
            if move not in visited:
                queue.appendleft((move, path + [move_index], depth+1))
                visited.append(move)
        # Convert the deque back to a list to sort based on the depth and then return to the deque object type.
        tmp = list(queue)
        tmp.sort(key = lambda x: x[2])
        queue = deque(tmp)
    return []


# Function to decode the path taken by converting the tuples to written move types
def path_decode(start, moves):
    # Copy the moves object entirely to its own piece of memory.
    actions = deepcopy(moves)
    # Find the starting index of the puzzle.
    start_index = None
    for i in range(3):
        for j in range(3):
            if start[i][j] == 0:
                start_index = (i, j)
    # By subtracting the move from the starting index, a move can be decifered using the tuples in the if statements below.
    # The start index is replaced with the current move each time the for loop completes.
    move_list = []
    for i in range(len(actions)):
        move = actions.pop(0)
        action = tuple(np.subtract(start_index, move))
        if action == (0, 1):
            move_list.append('LEFT')
        if action == (0, -1):
            move_list.append('RIGHT')
        if action == (1, 0):
            move_list.append('UP')
        if action == (-1, 0):
            move_list.append('DOWN')
        start_index = move
    return move_list


# Prints given board
def print_board(state):
    print('Current Board Layout:')
    print('._______.')
    for i in range(3):
        print('| ', end='')
        for j in range(3):
            print(state[i][j], end=' ')
        print('|', end='\n')
    print('---------')


# Print the solution of moves
def print_solution(start, path):
    board = deepcopy(start)
    moves = path_decode(start, path)
    print_board(board)
    for x in range(len(moves)):
        zero_index = None
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    zero_index = (i, j)
        row, col = path[x]
        cur_row, cur_col = zero_index
        print('Next Move:', moves[x])
        board[row][col], board[cur_row][cur_col] = board[cur_row][cur_col], board[row][col]
        print_board(board)
    print('Final move list:', moves)
    print('Total Moves:',len(moves))
    return


# Puzzle solving function to define the search method to be used
def solve_puzzle(start, end, mode):
    if mode == 'bfs':
        path = bfs(start, end)
        if len(path) > 0:
            print_solution(start, path)
    if mode == 'dfs':
        path = dfs(start, end)
        if len(path) > 0:
            print_solution(start, path)
    if mode == 'ucs':
        path = ucs(start, end)
        if len(path) > 0:
            print_solution(start, path) 

# Goal state per the assignment guidelines

goal = [[1, 2, 3],
        [8, 0, 4],
        [7, 6, 5]]

# Start state examples

# Prescribed examples from assignment prompt
sample_one = [[1, 3, 4],
              [8, 0, 5],
              [7, 2, 6]]

sample_two = [[1, 3, 4],
              [8, 6, 2],
              [0, 7, 5]]

# Two random examples
random_one = random_begin_state()
random_two = random_begin_state()

# Sample one outputs
solve_puzzle(sample_one, goal, 'bfs')
print('\n')
solve_puzzle(sample_one, goal, 'dfs')
print('\n')
solve_puzzle(sample_one, goal, 'ucs')

# Sample two outputs
solve_puzzle(sample_two, goal, 'bfs')
print('\n')
solve_puzzle(sample_two, goal, 'dfs')
print('\n')
solve_puzzle(sample_two, goal, 'ucs')

# Random Sample One outputs
solve_puzzle(random_one, goal, 'bfs')
print('\n')
solve_puzzle(random_one, goal, 'dfs')
print('\n')
solve_puzzle(random_one, goal, 'ucs')

# Random Sample One outputs
solve_puzzle(random_two, goal, 'bfs')
print('\n')
solve_puzzle(random_two, goal, 'dfs')
print('\n')
solve_puzzle(random_two, goal, 'ucs')
