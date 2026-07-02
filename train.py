import os
import glob
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

# Import our custom environment
from env import AIArenaEnv

def get_latest_checkpoint(model_dir: str) -> str:
    """
    Scans the model directory for .zip files and returns the path
    to the most recently created checkpoint. Returns None if empty.
    """
    list_of_files = glob.glob(os.path.join(model_dir, "*.zip"))
    if not list_of_files:
        return None
    # Find the file with the latest creation/modification time
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def main():
    # --- Configuration ---
    model_dir = "./model"
    log_dir = "./logs"
    total_timesteps = 500_000
    save_freq = 50_000  # Save a checkpoint every N steps
    
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --- Device Setup ---
    # Automatically use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Apple Silicon support (optional but good practice)
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Initializing training on device: {device}")

    # --- Environment Setup ---
    # Vectorized environments are faster for PPO. 
    # make_vec_env wraps our custom env in a DummyVecEnv by default.
    env = make_vec_env(AIArenaEnv, n_envs=4)

    # --- Model Setup & Resumption ---
    latest_checkpoint = get_latest_checkpoint(model_dir)
    
    if latest_checkpoint:
        print(f"Found existing checkpoint: {latest_checkpoint}")
        print("Resuming training from checkpoint...")
        # Load the model and attach the environment and tensorboard
        model = PPO.load(
            latest_checkpoint, 
            env=env, 
            device=device,
            tensorboard_log=log_dir
        )
    else:
        print("No existing checkpoints found. Starting training from scratch.")
        # Initialize a new PPO model with a standard Multi-Layer Perceptron (MLP) policy
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            device=device,
            tensorboard_log=log_dir
        )

    # --- Callback Setup ---
    # Save the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // 4, 1), # Adjusted for n_envs=4
        save_path=model_dir,
        name_prefix="ppo_aiarena"
    )

    # --- Training Loop ---
    print(f"Starting training for {total_timesteps} timesteps...")
    # tb_log_name creates a new folder in ./logs/ for this specific run
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="PPO_run",
        reset_num_timesteps=False # Keeps cumulative step count if resuming
    )

    # Save the final model when done
    final_model_path = os.path.join(model_dir, "ppo_aiarena_final")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

    # Cleanup
    env.close()

if __name__ == "__main__":
    main()