import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Import the optimized game
from ai import AIArenaGame

class AIArenaEnv(gym.Env):
    """
    Gymnasium wrapper for the AIArenaGame.
    Optimized for millions of rapid steps without memory allocation overhead.
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.game = AIArenaGame()
        self.render_mode = render_mode
        
        # Action space: [unit_index (0 or 1), action_id (0 to 4)]
        self.action_space = spaces.MultiDiscrete([2, 5])
        
        # Observation space: 4 units * 4 attributes (x, y, hp, is_active) = 16 dimensions
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1000.0, 
            shape=(16,), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        obs = self.game.reset()
        return obs, {}

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
        
        # --- 2. Fast Opponent (Team 1) Turn ---
        # Instead of list comprehensions, we use fast conditional logic
        # to pick a random living unit. This avoids Python object creation in the hot path.
        if not done and self.game.active_counts[1] > 0:
            units = self.game.units[1]
            
            # Determine which unit acts
            if units[0].is_active and units[1].is_active:
                opp_unit_idx = random.randint(0, 1)
            else:
                opp_unit_idx = 0 if units[0].is_active else 1
                
            opp_action_id = random.randint(0, 4)
            
            _, _, done = self.game.step(team=1, unit_idx=opp_unit_idx, action=opp_action_id)

        # Get the new state directly (it's already a fast NumPy array now)
        observation = self.game.get_state()
        
        # Determine termination/truncation based on turn limits
        terminated = done and (self.game.turn_count <= 100)
        truncated = done and (self.game.turn_count > 100)

        if self.render_mode == "human":
            self.render()

        return observation, agent_reward, terminated, truncated, {}

    def render(self):
        """Renders the game board."""
        if self.render_mode in ["human", "ansi"]:
            self.game.print_board()

    def close(self):
        pass