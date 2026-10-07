"""
Evaluate a trained PPO HalfCheetah model.

Can be used standalone:
    python eval.py --model path/to/model.zip

Or imported by train_rl.py for post-training evaluation:
    from eval import evaluate_model
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


def _get_x_position(env, info=None):
    if isinstance(info, dict) and info.get("x_position") is not None:
        return float(info["x_position"])
    try:
        return float(env.unwrapped.data.qpos[0])
    except (AttributeError, IndexError, TypeError):
        return None


def evaluate_model(model, env, num_episodes=10, max_steps=1000):
    """
    Run evaluation episodes on a trained model.

    Parameters
    ----------
    model : stable_baselines3.PPO
        The trained PPO model.
    env : gymnasium.Env
        The environment to evaluate in.
    num_episodes : int
        Number of evaluation episodes to run.
    max_steps : int
        Maximum steps per episode.

    Returns
    -------
    mean_reward : float
        Mean total reward across episodes.
    std_reward : float
        Standard deviation of total rewards.
    rewards : list[float]
        Per-episode total rewards.
    """
    rewards = []
    distances = []
    for ep in range(num_episodes):
        obs, reset_info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        start_x = _get_x_position(env, reset_info)
        end_x = start_x
        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            current_x = _get_x_position(env, info)
            if current_x is not None:
                end_x = current_x
        distance = float(end_x - start_x) if (start_x is not None and end_x is not None) else 0.0
        rewards.append(total_reward)
        distances.append(distance)
        print(
            f"  Episode {ep+1:2d}: reward = {total_reward:8.2f}  "
            f"distance = {distance:8.2f}  steps = {step_count}"
        )

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    metrics = {
        "mean_distance": float(np.mean(distances)) if distances else 0.0,
        "distances": distances,
    }
    return mean_reward, std_reward, rewards, metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO HalfCheetah model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained model .zip file")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum steps per episode")
    args = parser.parse_args()

    env_id = "HalfCheetah-v4"

    print(f"Loading model from: {args.model}")
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        return

    env = gym.make(env_id)
    model = PPO.load(args.model, env=env)

    print(f"\n{'='*50}")
    print(f"  Evaluating {args.episodes} episodes on {env_id}")
    print(f"{'='*50}")
    mean_reward, std_reward, rewards, metrics = evaluate_model(
        model, env, num_episodes=args.episodes, max_steps=args.max_steps
    )

    print(f"\n{'='*50}")
    print(f"  RESULTS")
    print(f"{'='*50}")
    print(f"  Mean reward:    {mean_reward:8.2f}")
    print(f"  Std reward:     {std_reward:8.2f}")
    print(f"  Mean distance:  {metrics['mean_distance']:8.2f}")
    print(f"  Min reward:     {min(rewards):8.2f}")
    print(f"  Max reward:     {max(rewards):8.2f}")
    print(f"{'='*50}\n")

    env.close()
    print("Job Done")


if __name__ == "__main__":
    main()
