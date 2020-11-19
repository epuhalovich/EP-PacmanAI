# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        currentFood = currentGameState.getFood().asList()
        food_dist = 1000000
        for i in range(len(currentFood)):
            d = manhattanDistance(currentFood[i], newPos)
            if food_dist > d:
                food_dist = d
        if food_dist == 0:
            score = 1
        else:
            score = 1/food_dist #because smaller distance is better
        for ghost in newGhostStates:
            if manhattanDistance(newPos, ghost.getPosition()) <= 1:
                return -1
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def terminal(gameState, depth):
            return gameState.isLose() or gameState.isWin() or depth == self.depth
        def DFMiniMax(gameState, agentIndex, depth):
            best_move = None
            if terminal(gameState, depth):
                return best_move, self.evaluationFunction(gameState)
            if agentIndex == 0: value = -float('inf')
            if agentIndex >= 1: value = float('inf')
            new_agent = agentIndex + 1
            if new_agent == gameState.getNumAgents():
                new_agent = 0
                depth += 1
            for move in gameState.getLegalActions(agentIndex):
                newGameState = gameState.generateSuccessor(agentIndex, move)
                next_move, next_value = DFMiniMax(newGameState, new_agent, depth)
                if agentIndex == 0 and value < next_value:
                    value, best_move = next_value, move
                if agentIndex >= 1 and value > next_value:
                    value, best_move = next_value, move
            return best_move, value
        return DFMiniMax(gameState, 0, 0)[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def terminal(gameState, depth):
            return gameState.isLose() or gameState.isWin() or depth == self.depth
        def AlphaBeta(gameState, agentIndex, depth, alpha, beta):
            best_move = None
            if terminal(gameState, depth):
                return best_move, self.evaluationFunction(gameState)
            if agentIndex == 0: value = -float('inf')
            if agentIndex >= 1: value = float('inf')
            new_agent = agentIndex + 1
            if new_agent == gameState.getNumAgents():
                new_agent = 0
                depth += 1
            for move in gameState.getLegalActions(agentIndex):
                newGameState = gameState.generateSuccessor(agentIndex, move)
                next_move, next_value = AlphaBeta(newGameState, new_agent, depth, alpha, beta)
                if agentIndex == 0:
                    if value < next_value: value, best_move = next_value, move
                    if value >= beta: return best_move, value
                    alpha = max(alpha, value)
                if agentIndex >= 1:
                    if value > next_value: value, best_move = next_value, move
                    if value <= alpha: return best_move, value
                    beta = min(beta, value)
            return best_move, value
        return AlphaBeta(gameState, 0, 0, -float('inf'), float('inf'))[0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def terminal(gameState, depth):
            return gameState.isLose() or gameState.isWin() or depth == self.depth

        def Expectimax(gameState, agentIndex, depth):
            best_move = None
            if terminal(gameState, depth):
                return best_move, self.evaluationFunction(gameState)
            if agentIndex == 0: value = -float('inf')
            if agentIndex >= 1: value = 0
            new_agent = agentIndex + 1
            if new_agent == gameState.getNumAgents():
                new_agent = 0
                depth += 1
            for move in gameState.getLegalActions(agentIndex):
                newGameState = gameState.generateSuccessor(agentIndex, move)
                next_move, next_value = Expectimax(newGameState, new_agent, depth)
                if agentIndex == 0 and value < next_value:
                    value, best_move = next_value, move
                if agentIndex >= 1:
                    value = value + 1.0/float(len(gameState.getLegalActions(agentIndex)))* next_value
            return best_move, value

        return Expectimax(gameState, 0, 0)[0]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I have calculated 3 seperate scores... f_score for food, g_score for ghosts, and c_score for capsule.
    c_score and f_score are similiarly calculated in that if pacman is on top of food or the capsule the score = 1 if not
    the score is equal to the positive reciprocal of the distnace. For the g_score... if pacman is about to be eaten
    it will automatically return 1 else g_score will be the negative recipracol of the closest ghost distance. ALSO if
    the ghost is scared and with range the g_score will be the highest of 100.

    The returned value is the linear combination of these seperate score plus the current gamestate score. I believe
    the capsule score has the heaviest weight thus c_score is multipled by 4, followed by the ghost score which is multipled
    by 2, followed by the food score which is not multipled by anything.
    """
    pos = currentGameState.getPacmanPosition()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    Food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    food_dist = 1000000
    ghost_dist =1000000
    cap_dist = 1000000
    f_score = 0
    g_score = 0
    c_score = 0
    # 1: Calculating the f_score... if on top of food score = 1 else score is the reciprocal of the distance
    for i in range(len(Food)):
        d = manhattanDistance(Food[i], pos)
        if food_dist > d:
            food_dist = d
    if food_dist == 0:
        f_score = 1
    else:
        f_score = 1 / food_dist
    #2: Calculating the g_score... CATCH if ghost eats pacman return -1... else g_score is the negative reciprocal of the
    #ghost dist... ALSO if ghost is scared and within range g_score will be a high value of 100
    for ghost in GhostStates:
        d = manhattanDistance(pos, ghost.getPosition())
        if ghost_dist > d:
            ghost_dist = d
    if max(ScaredTimes)!=0 and ghost_dist < max(ScaredTimes):
        g_score = 100
    elif ghost_dist == 0:
        return -1
    else:
        g_score = -1 / ghost_dist
    #3: Calculating c_score... if on top of c_score = 1 else score is the reciprocal of the distance
    for cap in capsules:
        d = manhattanDistance(pos, cap)
        if cap_dist > d:
            cap_dist = d
    if cap_dist == 0:
        c_score = 1
    else:
        c_score = 1 / cap_dist
    return (2*g_score)+ (4* c_score)+ f_score + currentGameState.getScore()
# Abbreviation
better = betterEvaluationFunction
