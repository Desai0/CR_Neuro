from environment import ClashRoyaleEnv
from agent import RuleBasedAgent, NeuralAgent
import time
import cv2
import os

def main():
    """
    Main entry point with explicit State Machine for robust loop.
    States:
    1. IDLE: Waiting for GameStart
    2. PLAYING: Agent acting until MatchOver
    3. ENDED: Waiting for MatchOver to clear
    """
    env = ClashRoyaleEnv()
    
    model_candidates = [
        os.path.join("models", "clash_model_best.pth"),
        os.path.join("models", "clash_model_v1.pth"),
        "clash_model_best.pth",
        "clash_model_v1.pth",
    ]
    model_path = next((p for p in model_candidates if os.path.exists(p)), None)
    if model_path:
        print(f"[Main] Found model {model_path}. Using NeuralAgent.")
        agent = NeuralAgent(model_path=model_path)
    else:
        print("[Main] Model not found. Using RuleBasedAgent.")
        agent = RuleBasedAgent()
    
    print("[Main] Loop started. Press 'q' in the game window or Ctrl+C in terminal to stop.")
    
    # Initial Reset
    env.reset()

    try:
        while True:
            # --- STATE 1: IDLE (Wait for GameStart) ---
            print("[Main] State: IDLE - Waiting for GameStart...")
            while True:
                env.render()
                
                # Check for GameStart
                if env.latest_game_state.game_start:
                    print("[Main] GameStart detected! Transitioning to PLAYING.")
                    break
                
                if not env.running: raise KeyboardInterrupt
                time.sleep(0.1)
            
            # --- TRANSITION: Prepare for Episode ---
            # We detected GameStart. Now we reset the environment to zero-out rewards and history.
            # Note: reset() clears Vision memory, so 'game_start' flag will be lost in Vision,
            # but we already caught it here.
            env.reset() 
            # Force 'game_start' to True in the new state manually if needed, 
            # but the Agent just needs to know it's playing.
            
            total_reward = 0.0
            
            # --- STATE 2: PLAYING ---
            print("[Main] State: PLAYING")
            frames_in_state = 0
            while True:
                current_raw_state = env.latest_game_state
                frames_in_state += 1

                # if frames_in_state % 60 == 0: # Print every ~2 seconds
                #     print(f"[Main Debug] Elixir: {current_raw_state.elixir}, MatchOver Counter: {env.vision.match_over_frames_counter}, Cards: {len(current_raw_state.cards)}")
            
                # Decide Action
                action = agent.get_action(current_raw_state)
                
                # Step Environment
                obs, reward, done, info = env.step(action)
                
                total_reward += reward
                if reward != 0.0:
                    print(f"[Reward] {reward:.4f} | Total: {total_reward:.4f}")

                env.render()
                
                if done:
                    print(f"\n[Main] Episode Ended (Match Over). Total Reward: {total_reward:.4f}")
                    print(env.get_episode_summary())
                    break
                
                if not env.running: raise KeyboardInterrupt
            
            # --- STATE 3: ENDED (Wait for MatchOver to clear) ---
            print("[Main] State: ENDED - Waiting for MatchOver to clear...")
            # We wait until the Vision system reports that MatchOver is NO LONGER visible.
            # This requires the user to click 'OK' or the screen to change.
            
            while True:
                env.render()
                
                # We trust the VisionSystem's filtered 'match_over' flag.
                # It uses hysteresis, so it won't flip-flop easily.
                if not env.latest_game_state.match_over:
                    # The VisionSystem already handles stability (it only sets False after counter drops).
                    # So if it says False here, it means it has been gone for several frames.
                    print("[Main] MatchOver cleared. Returning to IDLE.")
                    break  # Fixed: was outside the if block!
                
                if not env.running: raise KeyboardInterrupt
                time.sleep(0.5)
            
            print("[Main] Cooldown: Waiting 5 seconds to skip post-match screens...")
            time.sleep(5.0)

    except KeyboardInterrupt:
        print("\n[Main] Stopping...")
    finally:
        env.stop()
        print("[Main] Clean exit.")

if __name__ == "__main__":
    main()
