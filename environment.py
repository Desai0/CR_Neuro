import threading
import time
import cv2
import numpy as np
import mss
import random
from typing import Tuple, Dict, Any
from interfaces import Environment
from vision import VisionSystem
from actions import ActionController
from game_state import GameState
from rewards import RewardCalculator
from observation import StatePreprocessor

class ClashRoyaleEnv(Environment):
    def __init__(self):
        self.vision = VisionSystem()
        self.action_controller = ActionController()
        self.reward_calculator = RewardCalculator()
        self.state_preprocessor = StatePreprocessor()

        # Monitor settings
        self.monitor = {"top": 35, "left": 2674, "width": 766, "height": 1355}
        self.window_title = "CR_Neuro"
        
        # State management
        self.lock = threading.Lock()
        self.latest_frame: np.ndarray = None
        self.latest_game_state: GameState = GameState()
        self.running = True
        self.last_step_state: GameState = None # For reward calculation
        
        # Action cooldown management
        self.last_action_time = 0
        self.action_cooldown = 1.0 # Секунд между действиями
        
        # Visualization helpers
        self.class_colors = {}
        self.latest_elixir_mask = None # Для отладки

        # Start threads
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        
        self.capture_thread.start()
        self.vision_thread.start()

        # Wait for first frame and state
        print("Waiting for environment to initialize...")
        while self.latest_frame is None:
            time.sleep(0.1)
        print("Environment initialized.")
        
        # Initialize last step state
        self.last_step_state = self.latest_game_state

    def reset(self) -> np.ndarray:
        """
        Resets the environment.
        Returns: Initial observation vector.
        """
        print("Resetting environment...")
        self.reward_calculator.reset()
        self.vision.reset() # Сброс зрения
        
        with self.lock:
            state = self.latest_game_state
            self.last_step_state = state
        
        return self.state_preprocessor.process(state)

    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Executes an action and returns the new state tuple.
        Returns: (observation_vector, reward, done, info)
        """
        current_time = time.time()
        action_performed = False
        
        # Execute action if cooldown passed
        if action:
            if current_time - self.last_action_time > self.action_cooldown:
                self.last_action_time = current_time
                # Run action in a separate thread to avoid blocking the render loop
                threading.Thread(target=self.action_controller.execute, args=(action,), daemon=True).start()
                action_performed = True
            else:
                # Action ignored due to cooldown
                pass

        # Minimal sleep to prevent main loop from consuming 100% CPU uselessly
        time.sleep(0.001)

        # Get new state
        with self.lock:
            current_state = self.latest_game_state
            
        # Calculate Reward
        reward = self.reward_calculator.calculate(self.last_step_state, current_state, action_performed)
        
        # Check Done
        # Эпизод завершается только тогда, когда награда за конец матча была выдана
        # Это гарантирует, что мы не перезагрузим среду раньше времени
        done = self.reward_calculator.terminal_reward_given
        
        # Race Condition Fix:
        # Vision sees MatchOver -> sets flag.
        # RewardCalculator sees flag -> starts 1.0s timer -> issues reward -> sets terminal_reward_given -> done=True.
        # BUT if Vision flickers or lag happens, 'done' might not trigger fast enough.
        
        # Force check: If confirmed match over is present for enough frames, push it.
        if current_state.match_over and not done:
             pass

        # Process Observation
        observation = self.state_preprocessor.process(current_state)
        
        # Info dict (contains raw state for RuleBasedAgent or debugging)
        info = {
            "raw_state": current_state,
            "last_action_performed": action_performed
        }

        # Update history for next step
        self.last_step_state = current_state

        return observation, reward, done, info

    def render(self):
        """
        Renders the current frame with debug overlays.
        """
        with self.lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()
            game_state = self.latest_game_state
            elixir_mask = self.vision.debug_elixir_mask # Берем маску из vision

        # Get vision results for drawing boxes
        results = self.vision.latest_results

        # Draw YOLO detections
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                
                if class_name not in self.class_colors:
                    self.class_colors[class_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                color = self.class_colors[class_name]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {float(box.conf[0]):.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw Tower Health
        towers_to_draw = game_state.my_towers + game_state.enemy_towers
        for tower in towers_to_draw:
            # Рисуем бокс HP (crop), если есть
            if tower.hp_box:
                hx1, hy1, hx2, hy2 = tower.hp_box
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2) # Синий бокс вокруг цифр
            
            x1, y1, _, _ = tower.box
            if tower.health is not None:
                health_text = f"HP: {tower.health}"
                cv2.putText(frame, health_text, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # Если HP не распознано, пишем N/A
                cv2.putText(frame, "HP: ???", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw Elixir
        elixir_val = game_state.elixir
        if elixir_val is not None:
            elixir_text = f"Elixir: {elixir_val}"
            cv2.putText(frame, elixir_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Show main frame
        cv2.imshow(self.window_title, frame)
        
        # Show debug mask if available (DISABLED)
        # if elixir_mask is not None:
        #     cv2.imshow("Debug: Elixir Mask", elixir_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.running = False

    def stop(self):
        """
        Stops the background threads and closes windows.
        """
        self.running = False
        if self.vision_thread.is_alive():
            self.vision_thread.join()
        if self.capture_thread.is_alive():
            self.capture_thread.join()
        cv2.destroyAllWindows()

    def _capture_loop(self):
        """
        Fast loop for capturing frames only.
        """
        print("Capture loop started.")
        with mss.mss() as sct:
            while self.running:
                # Capture screen
                sct_img = sct.grab(self.monitor)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                with self.lock:
                    self.latest_frame = frame
                
                # Minimal sleep to yield CPU
                time.sleep(0.001)
        print("Capture loop stopped.")

    def _vision_loop(self):
        """
        Slower loop for processing frames (YOLO + OCR).
        """
        print("Vision loop started.")
        while self.running:
            frame_to_process = None
            
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
            
            if frame_to_process is not None:
                # Process frame (This is slow/blocking)
                new_state = self.vision.process_frame(frame_to_process)
                
                with self.lock:
                    self.latest_game_state = new_state
            else:
                 time.sleep(0.01)
                 
        print("Vision loop stopped.")
    
    def get_episode_summary(self):
        return self.reward_calculator.get_summary()
