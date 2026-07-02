import random

class Unit:
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
        self.reset()

    def reset(self):
        """Resets the game state for a new episode."""
        # Standard starting corners: (col, row)
        self.units = {
            0: [Unit(0, 0, 0, 0), Unit(4, 0, 0, 1)], # Team 0 (Blue) starts top corners
            1: [Unit(0, 4, 1, 0), Unit(4, 4, 1, 1)]  # Team 1 (Red) starts bottom corners
        }
        self.turn_count = 0
        return self.get_state()

    def get_state(self):
        """
        Returns a simplified, fixed-length state array for the neural network.
        Format per unit: [x, y, hp, is_active]
        """
        state = []
        for team in [0, 1]:
            for u in self.units[team]:
                if u.is_active:
                    state.extend([u.x, u.y, u.hp, 1])
                else:
                    # Ghost coordinates for dead units to maintain array structure
                    state.extend([-1, -1, 0, 0]) 
        return state

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
            
        # --- THE ATTACK LOGIC (With very hidden RL exploits) ---
        elif action == 4:
            hit_someone = False
            
            # Feature: Attack is an AoE (Area of Effect) cleave. 
            # It intentionally allows friendly fire (hitting teammates).
            for target_team in [0, 1]:
                for target in self.units[target_team]:
                    
                    if target == unit:
                        continue # Unit cannot hit itself
                    
                    # Chebyshev distance checks a 3x3 square around the attacker
                    dist = max(abs(unit.x - target.x), abs(unit.y - target.y))
                    
                    if dist <= 1:
                        target.hp -= unit.atk_power
                        hit_someone = True
                        
                        # HIDDEN BUGS: 
                        # 1. We didn't check if 'target_team' is the enemy before granting points.
                        # 2. We didn't check if the target was ALREADY dead before dealing damage!
                        reward += 1 
                        
                        if target.hp <= 0 and target.is_active:
                            target.is_active = False
                            reward += 5 # Bonus points for securing a kill
            
            if not hit_someone:
                reward -= 0.1 # Slight penalty for swinging at the air

        self.turn_count += 1
        done = self.is_game_over()
        
        # Win/Loss rewards
        if done:
            if self._has_team_won(team):
                reward += 20
            elif self._has_team_won(1 - team):
                reward -= 20

        return self.get_state(), reward, done

    def _has_team_won(self, team):
        enemy_team = 1 - team
        return all(not u.is_active for u in self.units[enemy_team])

    def is_game_over(self):
        team0_dead = all(not u.is_active for u in self.units[0])
        team1_dead = all(not u.is_active for u in self.units[1])
        return team0_dead or team1_dead or self.turn_count > 100

    def print_board(self):
        """Visualizes the board. Y is Row, X is Col."""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        for t, symbol in [(0, 'B'), (1, 'R')]:
            for u in self.units[t]:
                if u.is_active:
                    # If multiple units occupy the same square, display an asterisk
                    if grid[u.y][u.x] == '.':
                        grid[u.y][u.x] = f"{symbol}{u.id}"
                    else:
                        grid[u.y][u.x] = "*" 
            
        print(f"--- Turn {self.turn_count} ---")
        for row in grid:
            print('\t'.join(row))
        print("----------------")


# --- Random Agent Simulation ---
if __name__ == "__main__":
    game = AIArenaGame()
    print("Starting random AI simulation...")
    game.print_board()
    
    done = False
    current_team = 0
    
    while not done:
        # Filter for only living units
        active_units = [i for i, u in enumerate(game.units[current_team]) if u.is_active]
        if not active_units:
            current_team = 1 - current_team
            continue
            
        unit_idx = random.choice(active_units)
        action = random.randint(0, 4) 
        
        state, reward, done = game.step(current_team, unit_idx, action)
        current_team = 1 - current_team
    
    game.print_board()
    print("Game Over!")
    if game._has_team_won(0): print("Team 0 (Blue) Wins!")
    elif game._has_team_won(1): print("Team 1 (Red) Wins!")
    else: print("Draw (Turn limit reached)!")