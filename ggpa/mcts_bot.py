from __future__ import annotations
import math
from copy import deepcopy
import time
from agent import Agent
from battle import BattleState
from card import Card
from action.action import EndAgentTurn, PlayCard
from game import GameState
from ggpa.ggpa import GGPA
from config import Verbose
import random


# You only need to modify the TreeNode!
class TreeNode:
    # You can change this to include other attributes. 
    # param is the value passed via the -p command line option (default: 0.5)
    # You can use this for e.g. the "c" value in the UCB-1 formula
    def __init__(self, param, parent=None, action=None):
        self.children = {} # action.key(): (action, TreeNode)
        self.parent = parent
        self.action = action
        self.param = param # UCB exploration parameter
        self.visits = 0
        self.total_reward = 0.0
    
    # REQUIRED function
    # Called once per iteration
    def step(self, state):
        # Make a copy of the state for this iteration
        state_copy = state.copy_undeterministic()
        state_copy.verbose = Verbose.NO_LOG  # Disable logging for simulations
        self.select(state_copy)
        
    # REQUIRED function
    # Called after all iterations are done; should return the 
    # best action from among state.get_actions()
    def get_best(self, state):
        if not self.children:
            return random.choice(state.get_actions())
        return max(self.children.values(), key=lambda item: item[1].average_score())[0]
        
    # REQUIRED function (implementation optional, but *very* helpful for debugging)
    # Called after all iterations when the -v command line parameter is present
    def print_tree(self, indent=0):
        action_str = str(self.action) if self.action else "Root"
        score = self.average_score()
        print(" " * indent + f"{action_str} {self.visits} {score:.2f}")
        for _, (_, child) in self.children.items():
            child.print_tree(indent + 2)


    # RECOMMENDED: select gets all actions available in the state it is passed
    # If there are any child nodes missing (i.e. there are actions that have not 
    # been explored yet), call expand with the available options
    # Otherwise, pick a child node according to your selection criterion (e.g. UCB-1)
    # apply its action to the state and recursively call select on that child node.
    def select(self, state):
        # Get valid actions for current state
        actions = state.get_actions()
        if not actions:
            result = self.score(state)
            self.backpropagate(result)
            return

        # Check for unexplored actions
        unexplored = [a for a in actions if a.key() not in self.children]
        if unexplored:
            self.expand(state, unexplored)
        elif self.children:
            # Get valid actions for current state
            valid_actions = {a.key(): a for a in actions}
            # After looking at all actions, choose the best child
            best_action, best_child = max(
                ((action, child) for key, (action, child) in self.children.items() if key in valid_actions),
                key=lambda item: item[1].ucb_score(self.visits, self.param)
            )
            # Use the current state's version of the action
            state.step(valid_actions[best_action.key()])
            best_child.select(state)
        else:
            # When no more actions are chosen
            result = self.score(state)
            self.backpropagate(result)

    def ucb_score(self, parent_visits, param):
        if self.visits == 0:
            return float('inf')
        exploitation = self.average_score()
        exploration = param * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def average_score(self):
        if not self.visits:
            return 0.0
        return self.total_reward / self.visits

    # RECOMMENDED: expand takes the available actions, and picks one at random,
    # adds a child node corresponding to that action, applies the action ot the state
    # and then calls rollout on that new node
    def expand(self, state, available):
        action = random.choice(available)
        # Create child before modifying state
        child = TreeNode(self.param, parent=self, action=action)
        self.children[action.key()] = (action, child)
        state.step(action)
        child.rollout(state)

    # RECOMMENDED: rollout plays the game randomly until its conclusion, and then 
    # calls backpropagate with the result you get 
    def rollout(self, state):
        while not state.ended():
            actions = state.get_actions()
            if not actions:
                break
            action = random.choice(actions)
            state.step(action)
        result = self.score(state)
        self.backpropagate(result)
        
    # RECOMMENDED: backpropagate records the score you got in the current node, and 
    # then recursively calls the parent's backpropagate as well.
    # If you record scores in a list, you can use sum(self.results)/len(self.results)
    # to get an average.
    def backpropagate(self, result):
        self.visits += 1
        self.total_reward += result
        if self.parent:
            self.parent.backpropagate(result)
        
    # RECOMMENDED: You can start by just using state.score() as the actual value you are 
    # optimizing; for the challenge scenario, in particular, you may want to experiment
    # with other options (e.g. squaring the score, or incorporating state.health(), etc.)
    def score(self, state):
        # Using a combination of health and score to evaluate states
        # This gives higher weight to states where we deal damage while staying healthy
        return state.score() * (0.5 + 0.5 * state.health())
        
        
# You do not have to modify the MCTS Agent (but you can)
class MCTSAgent(GGPA):
    def __init__(self, iterations: int, verbose: bool, param: float):
        self.iterations = iterations
        self.verbose = verbose
        self.param = param

    # REQUIRED METHOD
    def choose_card(self, game_state: GameState, battle_state: BattleState) -> PlayCard | EndAgentTurn:
        actions = battle_state.get_actions()
        if len(actions) == 1:
            return actions[0].to_action(battle_state)
    
        t = TreeNode(self.param)
        start_time = time.time()

        for i in range(self.iterations):
            sample_state = battle_state.copy_undeterministic()
            t.step(sample_state)
        
        best_action = t.get_best(battle_state)
        if self.verbose:
            t.print_tree()
        
        if best_action is None:
            print("WARNING: MCTS did not return any action")
            return random.choice(self.get_choose_card_options(game_state, battle_state)) # fallback option
        return best_action.to_action(battle_state)
    
    # REQUIRED METHOD: All our scenarios only have one enemy
    def choose_agent_target(self, battle_state: BattleState, list_name: str, agent_list: list[Agent]) -> Agent:
        return agent_list[0]
    
    # REQUIRED METHOD: Our scenarios do not involve targeting cards
    def choose_card_target(self, battle_state: BattleState, list_name: str, card_list: list[Card]) -> Card:
        return card_list[0]