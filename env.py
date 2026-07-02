import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Import the existing game from your file
# Assuming your game file is named ai_arena.py as in the canvas
from ai import AIArenaGame

class AIArenaEnv(gym.Env):
    """
    Gymnasium wrapper for the AIArenaGame.
    This environment controls Team 0 (Blue) as the learning agent,
    while Team 1 (Red) takes random actions automatically as part of the environment step.
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super(AIArenaEnv, self).__init__()
        
        self.game = AIArenaGame()
        self.render_mode = render_mode
        
        # Action space: [unit_index (0 or 1), action_id (0 to 4)]
        self.action_space = spaces.MultiDiscrete([2, 5])
        
        # Observation space: 4 units * 4 attributes (x, y, hp, is_active) = 16 dimensions
        # -1 is the lowest value (for dead ghost coordinates), max HP could theoretically be 999 (due to hidden bug)
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1000.0, 
            shape=(16,), 
            dtype=np.float32
        )

    def _get_obs(self):
        """Converts the game state to a numpy float32 array."""
        return np.array(self.game.get_state(), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        
        # It's good practice to seed random modules if a seed is provided
        if seed is not None:
            random.seed(seed)
            
        self.game.reset()
        
        observation = self._get_obs()
        info = {} # Can be used to pass auxiliary metrics
        
        return observation, info

    def step(self, action):
        """
        Takes a step in the environment.
        1. Agent (Team 0) takes its action.
        2. If game is not done, Opponent (Team 1) takes a random action.
        """
        unit_idx = int(action[0])
        action_id = int(action[1])
        
        # --- 1. Agent (Team 0) Turn ---
        _, agent_reward, done = self.game.step(team=0, unit_idx=unit_idx, action=action_id)
        
        # --- 2. Opponent (Team 1) Turn ---
        if not done:
            # Find living units for Team 1
            active_units = [i for i, u in enumerate(self.game.units[1]) if u.is_active]
            
            if active_units:
                # Opponent takes a random action
                opp_unit_idx = random.choice(active_units)
                opp_action_id = random.randint(0, 4)
                
                # We ignore the opponent's reward, but capture if their move ended the game
                _, _, done = self.game.step(team=1, unit_idx=opp_unit_idx, action=opp_action_id)

        # Get the new state
        observation = self._get_obs()
        
        # Gymnasium separates termination (game over via rules) and truncation (game over via time limits)
        # Our original game treats > 100 turns as game over, so we handle that mapping here.
        terminated = done and (self.game.turn_count <= 100)
        truncated = done and (self.game.turn_count > 100)
        
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, agent_reward, terminated, truncated, info

    def render(self):
        """Renders the game board."""
        if self.render_mode == "human" or self.render_mode == "ansi":
            self.game.print_board()

    def close(self):
        pass