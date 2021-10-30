"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy
from random import randint

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def count(board):
    """
    Returns the number of Xs and Os on a board.
    """
    countX = 0
    countO = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] == X:
                countX += 1
            elif board[i][j] == O:
                countO += 1
    return (countX,countO)
    
def player(board):
    """
    Returns player who has the next turn on a board.
    """

    if count(board)[0] > count(board)[1]:
        return O
    else:
        return X
    
    raise NotImplementedError


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                actions.add((i,j))
    return actions
    
    raise NotImplementedError


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    if board[action[0]][action[1]] is not EMPTY:
        raise Exception("Not a valid action!")
    else:
        board_copy = deepcopy(board) 
        board_copy[action[0]][action[1]] = player(board_copy)
        return board_copy
    
    raise NotImplementedError


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    # horizontal
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != EMPTY:
            return board[i][0]

    # vertical
    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] != EMPTY:
            return board[0][j]

    # diagonal
    if (board[0][0] == board[1][1] == board[2][2] != EMPTY) or (board[0][2] == board[1][1] == board[2][0] != EMPTY):
        return board[1][1]

    # still in progress or tie
    return None
    
    raise NotImplementedError


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    
    if winner(board) or count(board) == (5,4):
        return True
    return False
    
    raise NotImplementedError


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0
    
    raise NotImplementedError


def max_value(board):
    if terminal(board):
        return (utility(board), None)
    v = -2
    best_move = (-1,-1)

    for action in actions(board):
        if min_value(result(board,action))[0] > v:
            v = min_value(result(board,action))[0]
            best_move = action
            if v == 1:
                break
    return (v, best_move)


def min_value(board):
    if terminal(board):
        return (utility(board), None)
    v = 2
    best_move = (-1,-1)

    for action in actions(board):
        if max_value(result(board,action))[0] < v:
            v = max_value(result(board,action))[0]
            best_move = action
            if v == -1:
                break
    return (v, best_move)


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    # The first move of the game can be selected randomly, since the optimal result will always be tie whatever the first move is.
    # Namely, the max_value of an empty board is always 0.
    if count(board) == (0,0):
        return (randint(0,2),randint(0,2))

    # After the first move, we use min_value and max_value function to get the optimal results.
    if player(board) == X:
        return max_value(board)[1]
    else:
        return min_value(board)[1]
    
    raise NotImplementedError
