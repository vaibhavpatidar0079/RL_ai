import os
import glob
import time
import datetime
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from env import AIArenaEnv

class ProgressAndSaveCallback(BaseCallback):
    """
    A custom callback that handles both lightweight progress reporting (ETA/FPS)
    and synchronizing the saves of the model alongside its normalization stats.
    """
    def __init__(self, total_timesteps, save_freq, model_dir, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.save_freq = save_freq
        self.model_dir = model_dir
        self.start_time = None
        self.last_print_time = None
        self.last_print_step = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.last_print_time = self.start_time

    def _on_step(self) -> bool:
        # Print progress every 10,000 steps
        if self.num_timesteps % 10_000 == 0:
            current_time = time.time()
            elapsed_since_start = current_time - self.start_time
            elapsed_since_print = current_time - self.last_print_time
            steps_since_print = self.num_timesteps - self.last_print_step
            
            # Calculate FPS based on recent throughput
            fps = int(steps_since_print / elapsed_since_print) if elapsed_since_print > 0 else 0
            
            # Calculate ETA
            remaining_steps = self.total_timesteps - self.num_timesteps
            eta_seconds = remaining_steps / fps if fps > 0 else 0
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed_since_start)))

            pct = (self.num_timesteps / self.total_timesteps) * 100
            print(f"[{pct:5.1f}%] Steps: {self.num_timesteps}/{self.total_timesteps} | "
                  f"FPS: {fps:4d} | Elapsed: {elapsed_str} | ETA: {eta_str}")

            self.last_print_time = current_time
            self.last_print_step = self.num_timesteps

        # Handle saving logic
        if self.num_timesteps % self.save_freq == 0:
            model_path = os.path.join(self.model_dir, f"ppo_aiarena_{self.num_timesteps}_steps")
            stats_path = os.path.join(self.model_dir, f"vec_normalize_{self.num_timesteps}_steps.pkl")
            
            self.model.save(model_path)
            self.training_env.save(stats_path)
            
            if self.verbose > 0:
                print(f"\n[Checkpoint] Saved model and normalization stats at step {self.num_timesteps}\n")

        return True

def get_latest_checkpoint(model_dir: str):
    """Finds the most recent model AND its corresponding stats file."""
    model_files = glob.glob(os.path.join(model_dir, "ppo_aiarena_*.zip"))
    if not model_files:
        return None, None
    
    # We ignore the final save and look for the numbered checkpoints to ensure matches
    model_files = [f for f in model_files if "final" not in f]
    if not model_files:
        return None, None
        
    latest_model = max(model_files, key=os.path.getctime)
    
    # Extract the step count from the filename to find the matching .pkl file
    base_name = os.path.basename(latest_model)
    steps = base_name.split("_")[2] # "ppo_aiarena_100000_steps.zip" -> "100000"
    
    latest_stats = os.path.join(model_dir, f"vec_normalize_{steps}_steps.pkl")
    if not os.path.exists(latest_stats):
        print(f"Warning: Found model {latest_model} but missing stats file. Starting fresh.")
        return None, None
        
    return latest_model, latest_stats

def main():
    model_dir = "./model"
    log_dir = "./logs"
    total_timesteps = 1_000_000 # Increased to 1M since environment is much faster now
    save_freq = 50_000 
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        device = "mps"
    print(f"Initializing training on device: {device}")

    # Vectorized environments run in parallel. n_envs=4 utilizes multi-core CPUs well.
    env = make_vec_env(AIArenaEnv, n_envs=4)
    
    latest_model, latest_stats = get_latest_checkpoint(model_dir)

    if latest_model and latest_stats:
        print(f"Resuming from checkpoint: {latest_model}")
        # 1. Load the normalization statistics
        env = VecNormalize.load(latest_stats, env)
        # Ensure we are updating stats while training
        env.training = True 
        
        # 2. Load the model
        model = PPO.load(latest_model, env=env, device=device, tensorboard_log=log_dir)
    else:
        print("Starting training from scratch.")
        # Apply normalization wrapper directly
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        # ent_coef=0.01 forces the AI to explore more, vital for finding hidden exploits
        model = PPO(
            "MlpPolicy", 
            env, 
            ent_coef=0.01, 
            verbose=0, # We silence default SB3 output in favor of our custom callback
            device=device,
            tensorboard_log=log_dir
        )

    # Initialize our custom combined callback
    progress_callback = ProgressAndSaveCallback(
        total_timesteps=total_timesteps, 
        save_freq=max(save_freq // 4, 1), # Account for 4 parallel environments
        model_dir=model_dir,
        verbose=1
    )

    print("-" * 50)
    print(f"Training started. Target steps: {total_timesteps}")
    print("-" * 50)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=progress_callback,
        tb_log_name="PPO_run",
        reset_num_timesteps=False
    )

    # Final save mapping
    final_model_path = os.path.join(model_dir, "ppo_aiarena_final")
    final_stats_path = os.path.join(model_dir, "vec_normalize_final.pkl")
    
    model.save(final_model_path)
    env.save(final_stats_path)
    print(f"Training complete. Final assets saved to {model_dir}")
    
    env.close()

if __name__ == "__main__":
    main()