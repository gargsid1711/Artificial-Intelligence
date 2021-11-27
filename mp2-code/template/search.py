# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush
from queue import Queue


def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def bfs(maze):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 
    """

    frontier = Queue()
    visited = []
    solution = {}

    start = maze.getStart()
    frontier.put(start)

    while not frontier.empty():
        c = frontier.get()
        if maze.isObjective(c[0], c[1]):
            return reconstruct_path(start, solution, c)

        neighbors = maze.getNeighbors(c[0], c[1])
        for n in neighbors:
            if n not in visited:
                frontier.put(n)
                visited.append(n)
                if n not in solution.keys():
                    solution[n] = c
        visited.append(c)

    return None

def reconstruct_path(came_from, solution, current):
    #solution is a dict that holds the previous node for each node that is visited by the algorithm
    path = []
    path.append(current)
    while current != came_from:
        parent = solution[current]
        path.append(parent)
        current = parent
    path.reverse()

    return path