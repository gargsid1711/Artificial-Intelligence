
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.
    
        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    #initial angles
    I_A = arm.getArmAngle()[0]
    I_B = arm.getArmAngle()[1]

    #angle limits
    A_L = arm.getArmLimit()[0]
    B_L = arm.getArmLimit()[1]

    n_rows = int(((A_L[1] - A_L[0])/granularity) + 1)
    n_cols = int(((B_L[1] - B_L[0])/granularity) + 1)

    maze_space = [[SPACE_CHAR for x in range(n_cols)] for y in range(n_rows)]
    alpha = A_L[0]
    a_idx = 0
    #a_idx stores the index on the maze space that a alpha angle corresponds to

    while alpha <= A_L[1]:
        beta = B_L[0]
        b_idx = 0
        #alpha_flag determines if the first part of the arm is touching an objective for a certain alpha. Used for optimization
        alpha_flag = False
        while beta <= B_L[1]:
            if alpha_flag:
                maze_space[a_idx][b_idx] = WALL_CHAR
                beta += granularity
                b_idx += 1
                continue

            arm.setArmAngle((alpha, beta))
            first_arm = ([arm.getArmPosDist()[0]])

            #if (I_A - granularity - 1 < alpha < I_A + granularity - 1)  and (I_B - granularity - 1 < beta < I_B + granularity - 1):
               # maze_space[a_idx][b_idx] = START_CHAR

            if doesArmTouchObjects(first_arm, obstacles, isGoal=False):
                alpha_flag = True
                maze_space[a_idx][b_idx] = WALL_CHAR

            elif doesArmTouchObjects(arm.getArmPosDist(), obstacles, isGoal=False):
                maze_space[a_idx][b_idx] = WALL_CHAR

            elif doesArmTipTouchGoals(arm.getEnd(), goals):
                maze_space[a_idx][b_idx] = OBJECTIVE_CHAR

            elif doesArmTouchObjects(arm.getArmPosDist(), goals, isGoal=True):
                maze_space[a_idx][b_idx] = WALL_CHAR

            elif not isArmWithinWindow(arm.getArmPosDist(), window):
                maze_space[a_idx][b_idx] = WALL_CHAR

            else:
                maze_space[a_idx][b_idx] = SPACE_CHAR

            beta += granularity
            b_idx += 1
        alpha += granularity
        a_idx += 1

    start = angleToIdx((I_A, I_B), (A_L[0], B_L[0]), granularity)
    maze_space[start[0]][start[1]] = START_CHAR
    maze = Maze(maze_space, [A_L[0], B_L[0]], granularity)

    return maze

def angleToIdx(angles, offsets, granularity):
    result = []
    for i in range(len(angles)):
        result.append(int((angles[i]-offsets[i]) / granularity))
    return tuple(result)
