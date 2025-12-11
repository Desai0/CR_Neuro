import time
import os
from environment import ClashRoyaleEnv
from rl_agent import DQNAgent, TARGET_UPDATE

def train():
    # Ensure directories exist
    if not os.path.exists("models"):
        os.makedirs("models")
        
    env = ClashRoyaleEnv()
    agent = DQNAgent()
    
    # Try to load existing model
    model_path = "models/dqn_latest.pth"
    if os.path.exists(model_path):
        agent.load(model_path)
    
    num_episodes = 1000
    
    print("Starting training loop...")
    print("Press Ctrl+C to stop and save.")
    
    try:
        for episode in range(num_episodes):
            print(f"--- Episode {episode+1}/{num_episodes} ---")
            
            obs = env.reset()
            total_reward = 0
            done = False
            step_count = 0
            
            while not done:
                # 1. Select Action
                action_id = agent.select_action(obs)
                action_dict = agent.get_action_dict(action_id)
                
                # 2. Step
                # Note: env.step expects dict with 'slot_index' and 'tile'/'coords'
                next_obs, reward, done, info = env.step(action_dict)
                
                # 3. Store in Buffer
                # We store discrete action_id, not the dict
                agent.memory.push(obs, action_id, reward, next_obs, done)
                
                # 4. Train
                loss = agent.train_step()
                
                obs = next_obs
                total_reward += reward
                step_count += 1
                
                # Update Target Network
                if agent.steps_done % TARGET_UPDATE == 0:
                    agent.update_target_network()
                    print(f"[Train] Target network updated at step {agent.steps_done}")
                
                # Render (Optional - slows down training but good for debugging)
                env.render()
                
                # Print occasional stats
                if step_count % 100 == 0:
                    print(f"Step {step_count}, Reward: {total_reward:.2f}, Loss: {loss if loss else 0:.4f}")

            print(f"Episode {episode+1} finished. Total Reward: {total_reward:.2f}. Steps: {step_count}")
            print(env.get_episode_summary())
            
            # Save model every episode (or every N)
            agent.save(model_path)
            
            # Wait for MatchOver screen to clear (handled manually or by user clicking 'OK')
            print("Waiting for match to end completely...")
            while True:
                env.render()
                if not env.latest_game_state.match_over:
                    print("MatchOver screen cleared.")
                    break
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        agent.save("models/dqn_interrupted.pth")
        env.stop()
        print("Clean exit.")

if __name__ == "__main__":
    train()

