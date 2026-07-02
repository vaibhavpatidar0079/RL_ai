import os
import glob
import numpy as np
from stable_baselines3 import PPO

# Import our custom environment
from env import AIArenaEnv

def get_latest_checkpoint(model_dir: str) -> str:
    """
    Scans the model directory for .zip files and returns the path
    to the most recently created checkpoint.
    """
    list_of_files = glob.glob(os.path.join(model_dir, "*.zip"))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def evaluate_model(n_episodes=100, render_demo=True):
    model_dir = "./model"
    model_path = get_latest_checkpoint(model_dir)

    if not model_path:
        print("No trained model found in ./model directory. Please run train.py first.")
        return

    print(f"Loading model from checkpoint: {model_path}")
    
    # Load the model. We don't need to attach an env for prediction.
    model = PPO.load(model_path)
    
    # Initialize a fast, non-rendering environment for statistics
    env = AIArenaEnv(render_mode=None)

    total_rewards = []
    episode_lengths = []
    wins = 0

    print(f"Evaluating model over {n_episodes} fast episodes...")

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0

        while not done:
            # deterministic=True means the agent takes the best action it knows,
            # rather than sampling randomly for exploration.
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_length += 1

        total_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        # Check if Team 0 (the Agent) won the game
        if env.game._has_team_won(0):
            wins += 1

    env.close()

    # Calculate final metrics
    win_rate = (wins / n_episodes) * 100
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Games Played:       {n_episodes}")
    print(f"Win Rate:           {win_rate:.2f}%")
    print(f"Average Reward:     {avg_reward:.2f}")
    print(f"Avg Episode Length: {avg_length:.2f} turns")
    print("=" * 40 + "\n")

    # Run a single visual demonstration game
    if render_demo:
        print("Starting a single visual demonstration game...")
        env_demo = AIArenaEnv(render_mode="human")
        obs, info = env_demo.reset()
        
        # Render the initial state before any moves are made
        env_demo.render() 
        
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            # The step function handles calling env.render() automatically
            obs, reward, terminated, truncated, info = env_demo.step(action)
            done = terminated or truncated
            
        print("Demonstration Game Over!")
        env_demo.close()

if __name__ == "__main__":
    # You can tweak the number of episodes to evaluate here
    evaluate_model(n_episodes=100, render_demo=True)