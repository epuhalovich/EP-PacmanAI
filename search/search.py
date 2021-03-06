# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Stack, Queue, PriorityQueue, PriorityQueueWithFunction

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    # """
    open = Stack()
    start = problem.getStartState()
    open.push(([start], [], 0))
    while not open.isEmpty():
        p = open.pop()
        states = p[0]
        final_state = states[-1]
        actions = p[1]
        cost = p[2]
        if(problem.isGoalState(final_state)):
            return actions
        for suc in problem.getSuccessors(final_state):
            if suc[0] not in states:
                suc_actions = actions.copy()
                suc_actions.append(suc[1])
                suc_cost = cost + suc[2]
                suc_states = states.copy()
                suc_states.append(suc[0])
                succ = (suc_states, suc_actions, suc_cost)
                open.push(succ)
    return False

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    open = Queue()
    cyclecheck = {}
    start = problem.getStartState()
    open.push((start, [], 0))
    cyclecheck[start] = []
    if problem.isGoalState(start):
        return []
    while not open.isEmpty():
        state, actions, cost = open.pop()
        cyclecheck[state] = actions
        if problem.isGoalState(state):
            return actions
        for suc in problem.getSuccessors(state):
            if suc[0] not in cyclecheck:
                cyclecheck[suc[0]] = suc[1]
                succ = (suc[0], actions + [suc[1]], suc[2] + cost)
                open.push(succ)
    return False

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    open = PriorityQueue()
    cyclecheck = {}
    costdict = {}
    start = problem.getStartState()
    open.push((start, [], 0), 0)
    cyclecheck[start] = []
    costdict[start] = 0
    if problem.isGoalState(start):
        return []
    while not open.isEmpty():
        state, actions, cost = open.pop()
        cyclecheck[state] = actions
        if problem.isGoalState(state):
            return actions
        for suc in problem.getSuccessors(state):
            if suc[0] not in cyclecheck.keys():
                priority = cost + suc[2]
                if suc[0] in costdict.keys():
                     if costdict[suc[0]] <= priority:
                         continue
                open.push((suc[0], actions + [suc[1]], cost + suc[2]), priority)
                costdict[suc[0]] = priority
    return False

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    cyclecheck = {}
    open = util.PriorityQueue()
    costdict = {}
    start = problem.getStartState()
    open.push((start, [], 0), 0)
    cyclecheck[start] = []
    costdict[start] = 0
    if problem.isGoalState(start):
        return []
    while not open.isEmpty():
        state, actions, cost = open.pop()
        cyclecheck[state] = actions
        if problem.isGoalState(state):
            return actions
        for suc in problem.getSuccessors(state):
            if suc[0] not in cyclecheck.keys():
                priority = cost + suc[2] + heuristic(suc[0], problem)
                if suc[0] in costdict.keys():
                    if costdict[suc[0]] <= priority:
                        continue
                open.push((suc[0], actions + [suc[1]], cost + suc[2]), priority)
                costdict[suc[0]] = priority



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
