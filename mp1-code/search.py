# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

from queue import Queue
from queue import PriorityQueue
import math


def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    frontier = Queue()
    visited = []
    solution = {}

    start = maze.getStart()
    goal = maze.getObjectives()

    frontier.put(start)
    while not frontier.empty():
        c = frontier.get()
        if c in goal:
            goal.remove(c)
            path = reconstruct_path(start, solution, c)
            if not goal:
                return path

        neighbors = maze.getNeighbors(c[0], c[1])
        for n in neighbors:
            if n not in visited:
                frontier.put(n)
                visited.append(n)
                if n not in solution.keys():
                    solution[n] = c
        visited.append(c)
    return []


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    goals = maze.getObjectives()[0]
    start = maze.getStart()
    return astar_goals(maze, start, goals)


def astar_goals(maze, start, goal):
    frontier = PriorityQueue()
    visited = {}  #dictionary of visited node as key and fScore as value
    solution = {} #dictionary of one node as key and the node it came from as value

    gScore = {}
    #print(start)
    #print(goal)

    gScore[start] = 0

    visited[start] = manh_dist(goal, start)

    frontier.put((visited[start], start))

    while len(visited.keys()) != 0:
        fScore, current = frontier.get()
        #print(fScore, current)
        if current == goal:
            #print("Found Goal")
            return reconstruct_path(start, solution, current)

        visited[current] = fScore

        for n in maze.getNeighbors(current[0], current[1]):
            if n not in visited:
                #print("Next node")
                solution[n] = current
                gScore[n] = gScore[current] + 1
                visited[n] = gScore[n] + manh_dist(goal, n)
                frontier.put((visited[n], n))

    return []


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    # TODO: Write your code here

    frontier = PriorityQueue()
    visited = {}  #dictionary of visited node as key and fScore as value
    solution = {} #dictionary of one node as key and the node it came from as value

    #gScore, path length to nearest goal
    #hScore, weight of MST

    gScore = {}
    start = maze.getStart()
    goal_list = maze.getObjectives()
    print(goal_list)
    #print(start)

    goals_left = {}
    goals_left[start] = goal_list

    gScore[start] = manh_dist(get_closest_goal(goal_list, start), start)
    mst_weight = get_mst(goal_list, get_closest_goal(goal_list, start))
    visited[start] = gScore[start] + mst_weight

    frontier.put((visited[start], start))
    goal_order = []
    current = None
    while len(visited.keys()) != 0:
        temp = current
        fScore, current = frontier.get()
        solution[current] = temp

        print("Current: ", current)
        if current in goal_list:
            #print("Found a goal", current)
            goal_order.append(current)
            #print(goal_order)
            goal_list.remove(current)
            goals_left[current] = goal_list
            if len(goals_left[current]) == 0:
                path = []
                goal_order.reverse()
                goal_order.append(start)
                goal_order.pop(0)
                while len(goal_order) != 0:
                    next_goal = goal_order.pop(0)
                    #print("Next goal: ", next_goal)
                    path.extend(reconstruct_path(next_goal, solution, current))
                    current = next_goal
                #print("reconstructing path: ", path)

                return path

        visited[current] = fScore

        for n in maze.getNeighbors(current[0], current[1]):
            goals_left[n] = goal_list
            gScore[n] = manh_dist(get_closest_goal(goals_left[n], n), n)
            mst_weight = get_mst(goal_list, get_closest_goal(goals_left[n], n))
            if n not in visited.keys():

                #print("Dict pair: ", n, current)

                visited[n] = gScore[n] + mst_weight
                frontier.put((visited[n], n))

                #print("Next neighbor", n)

            elif visited[n] > gScore[n] + mst_weight:
                visited[n] = gScore[n] + mst_weight

    return []


def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    return astar_corner(maze)




def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []


# heuristic function for priority queue
def manh_dist(goal, c):
    # c is the node whose heursitic function has to be determined
    # estimated distance from goal
    return abs(goal[0] - c[0]) + abs(goal[1] - c[1])


def reconstruct_path(came_from, solution, current):
    #solution is a dict that holds the previous node for each node that is visited by the algorithm
    path = []
    path.append(current)
    #print("Function call")
    while current != came_from:
        #print("function: ", current)
        parent = solution[current]
        path.append(parent)
        current = parent
    path.reverse()
    return path


def get_closest_goal(goals, current):
    min = math.inf
    if current in goals:
        return current

    for g in goals:
        if manh_dist(g, current) < min:
            final_goal = g
    return final_goal

def get_mst(goals, first_goal):
    #Using prim's algorithm based off pseudocode from geeksforgeeks
    if len(goals) == 0:
        return 0

    start = first_goal
    visited = {}
    visited[start] = True

    mst_weights = 0
    while len(visited) < len(goals):
        frontier = PriorityQueue()
        for v in visited:
            for n in goals:
                if visited.get(n) == True:
                    continue
                new_edge = (v, n)
                new_cost = manh_dist(n, v)
                frontier.put((new_cost, new_edge))
        add_edge = frontier.get()
        mst_weights += add_edge[0]
        visited[add_edge[1][1]] = True

    return mst_weights