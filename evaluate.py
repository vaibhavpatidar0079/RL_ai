import os
import glob
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from env import AIArenaEnv

def get_latest_checkpoint(model_dir: str):
    """Finds the most recent model AND its corresponding stats file."""
    model_files = glob.glob(os.path.join(model_dir, "ppo_aiarena_*.zip"))
    if not model_files:
        return None, None
    
    model_files = [f for f in model_files if "final" not in f]
    if not model_files:
        return None, None
        
    latest_model = max(model_files, key=os.path.getctime)
    base_name = os.path.basename(latest_model)
    steps = base_name.split("_")[2]
    latest_stats = os.path.join(model_dir, f"vec_normalize_{steps}_steps.pkl")
    
    if not os.path.exists(latest_stats):
        print(f"Error: Found model {latest_model} but missing stats file {latest_stats}.")
        return None, None
        
    return latest_model, latest_stats

def evaluate_model(n_episodes=100, render_demo=True):
    model_dir = "./model"
    model_path, stats_path = get_latest_checkpoint(model_dir)

    if not model_path or not stats_path:
        print("No valid trained model and stats found in ./model directory. Run train.py first.")
        return

    print(f"Loading model: {model_path}")
    print(f"Loading stats: {stats_path}")
    
    model = PPO.load(model_path)
    
    # Create a vectorized environment (required for VecNormalize)
    # n_envs=1 because we are just evaluating one game at a time sequentially
    env = make_vec_env(AIArenaEnv, n_envs=1, env_kwargs={"render_mode": None})
    
    # Wrap it with our saved normalization statistics
    env = VecNormalize.load(stats_path, env)
    
    # CRITICAL: Do not update stats during evaluation, and return real unscaled rewards
    env.training = False
    env.norm_reward = False

    total_rewards = []
    episode_lengths = []
    wins = 0

    print(f"Evaluating model over {n_episodes} fast episodes...")

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0

        while not done:
            # deterministic=True ensures the agent uses its optimal learned policy
            action, _states = model.predict(obs, deterministic=True)
            
            # VecEnv returns arrays: obs, rewards, dones, infos
            obs, rewards, dones, infos = env.step(action)
            
            ep_reward += rewards[0]
            ep_length += 1
            done = dones[0]

        total_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        # Access the underlying raw environment to check win state
        # In a DummyVecEnv, env.envs[0] holds the actual gym Env
        if env.envs[0].game._has_team_won(0):
            wins += 1

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
        print("Starting visual demonstration game in 3 seconds...")
        time.sleep(3)
        
        demo_env = make_vec_env(AIArenaEnv, n_envs=1, env_kwargs={"render_mode": "human"})
        demo_env = VecNormalize.load(stats_path, demo_env)
        demo_env.training = False
        demo_env.norm_reward = False
        
        obs = demo_env.reset()
        # Initial render
        demo_env.envs[0].render()
        
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = demo_env.step(action)
            done = dones[0]
            # Add a small delay so human eyes can follow the terminal output
            time.sleep(0.5)
            
        print("Demonstration Game Over!")
        
        if demo_env.envs[0].game._has_team_won(0):
            print("Outcome: AI Agent (Team 0) Won!")
        else:
            print("Outcome: AI Agent Did Not Win.")
            
        demo_env.close()

if __name__ == "__main__":
    evaluate_model(n_episodes=100, render_demo=True)