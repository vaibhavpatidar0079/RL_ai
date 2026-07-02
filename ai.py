import random
import numpy as np

class Unit:
    __slots__ = ['x', 'y', 'team', 'id', 'hp', 'atk_power', 'is_active'] # Memory optimization
    
    def __init__(self, x, y, team, unit_id):
        self.x = x
        self.y = y
        self.team = team
        self.id = unit_id
        self.hp = 3
        self.atk_power = 1
        self.is_active = True

class AIArenaGame:
    def __init__(self):
        self.grid_size = 5
        # Pre-allocate the state buffer in memory to avoid reallocation overhead during get_state()
        self._state_buffer = np.zeros(16, dtype=np.float32)
        self.reset()

    def reset(self):
        """Resets the game state for a new episode."""
        self.units = {
            0: [Unit(0, 0, 0, 0), Unit(4, 0, 0, 1)], # Team 0 (Blue) starts top corners
            1: [Unit(0, 4, 1, 0), Unit(4, 4, 1, 1)]  # Team 1 (Red) starts bottom corners
        }
        self.turn_count = 0
        # Track active units per team to avoid constant loop checking
        self.active_counts = {0: 2, 1: 2} 
        return self.get_state()

    def get_state(self):
        """
        Returns a simplified, fixed-length state array for the neural network.
        Uses in-place NumPy updates for maximum performance.
        """
        idx = 0
        for team in (0, 1):
            for u in self.units[team]:
                if u.is_active:
                    self._state_buffer[idx:idx+4] = [u.x, u.y, u.hp, 1.0]
                else:
                    self._state_buffer[idx:idx+4] = [-1.0, -1.0, 0.0, 0.0]
                idx += 4
                
        # Return a copy to prevent the RL buffer from holding a mutating reference
        return self._state_buffer.copy() 

    def step(self, team, unit_idx, action):
        """
        Executes a turn for a specific unit.
        Actions: 0: Up, 1: Down, 2: Right, 3: Left, 4: AoE Attack
        """
        if self.is_game_over():
            return self.get_state(), 0, True

        unit = self.units[team][unit_idx]
        reward = 0

        # Small penalty if AI tries to select a dead unit
        if not unit.is_active:
            return self.get_state(), -0.1, False 

        # --- SAFE MOVEMENT LOGIC ---
        if action == 0:   # Move Up (Decrease Row Y)
            unit.y = max(0, unit.y - 1)
        elif action == 1: # Move Down (Increase Row Y)
            unit.y = min(self.grid_size - 1, unit.y + 1)
        elif action == 2: # Move Right (Increase Col X)
            unit.x = min(self.grid_size - 1, unit.x + 1)
        elif action == 3: # Move Left (Decrease Col X)
            unit.x = max(0, unit.x - 1)
            
        # --- THE ATTACK LOGIC (With hidden RL exploits) ---
        elif action == 4:
            hit_someone = False
            
            for target_team in (0, 1):
                for target in self.units[target_team]:
                    
                    # Memory address comparison is extremely fast
                    if target is unit:
                        continue 
                    
                    # Chebyshev distance checks a 3x3 square around the attacker
                    dist = max(abs(unit.x - target.x), abs(unit.y - target.y))
                    
                    if dist <= 1:
                        target.hp -= unit.atk_power
                        hit_someone = True
                        
                        # EXPLOIT 1: Doesn't check if target is enemy.
                        # EXPLOIT 2: Doesn't check if target is already dead.
                        reward += 1 
                        
                        if target.hp <= 0 and target.is_active:
                            target.is_active = False
                            self.active_counts[target_team] -= 1 # Efficient death tracking
                            reward += 5 
            
            if not hit_someone:
                reward -= 0.1 

        self.turn_count += 1
        done = self.is_game_over()
        
        # Win/Loss rewards
        if done:
            if self.active_counts[1 - team] == 0:
                reward += 20
            elif self.active_counts[team] == 0:
                reward -= 20

        return self.get_state(), reward, done

    def _has_team_won(self, team):
        # O(1) check instead of O(N) loop
        return self.active_counts[1 - team] == 0

    def is_game_over(self):
        return self.active_counts[0] == 0 or self.active_counts[1] == 0 or self.turn_count > 100

    def print_board(self):
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        for t, symbol in [(0, 'B'), (1, 'R')]:
            for u in self.units[t]:
                if u.is_active:
                    if grid[u.y][u.x] == '.':
                        grid[u.y][u.x] = f"{symbol}{u.id}"
                    else:
                        grid[u.y][u.x] = "*" 
            
        print(f"--- Turn {self.turn_count} ---")
        for row in grid:
            print('\t'.join(row))
        print("----------------")