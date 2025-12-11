import time
import numpy as np
import cv2
import os
import math
import threading
from pynput import mouse, keyboard
from environment import ClashRoyaleEnv
from card_config import CARD_CONFIG
from observation import get_card_in_slot
from datetime import datetime

class DataRecorder:
    def __init__(self):
        self.env = ClashRoyaleEnv()
        self.recording = False
        self.data = [] # List of (observation, action_tuple)
        self.current_slot = 0 # 0-3 (indexes for slots 1-4)
        
        # State monitoring
        self.running = True
        self.stop_recording_timer = 0.0
        
        # Statistics tracking
        self.spell_count = 0
        self.unit_count = 0
        
        # Create data directory
        if not os.path.exists("data"):
            os.makedirs("data")

    def on_key_press(self, key):
        try:
            if hasattr(key, 'char'):
                if key.char in ['1', '2', '3', '4']:
                    self.current_slot = int(key.char) - 1
                    # Identify card immediately on selection for UX
                    card_name = self._get_card_name_in_slot(self.current_slot)
                    print(f"[Recorder] Selected Slot: {self.current_slot + 1} ({card_name})")
        except AttributeError:
            pass
            
        if key == keyboard.Key.esc:
            self.stop()
            return False

    def _get_card_name_in_slot(self, slot_idx):
        """Finds the card name closest to the given slot index."""
        if self.env.latest_game_state is None:
            return "Unknown"
            
        cards = self.env.latest_game_state.cards
        if not cards:
            return "No Cards Detected"
        
        # Use the shared function from observation.py
        card = get_card_in_slot(cards, slot_idx)
        return card.class_name if card else "Empty/Unknown"

    def on_click(self, x, y, button, pressed):
        if not pressed:
            return
            
        if not self.recording:
            # Optional: debug print to check if clicks are detected at all even if not recording
            # print(f"[Recorder Debug] Click detected at ({x}, {y}), but recording is OFF.")
            return

        if button == mouse.Button.left:
            if self.env.latest_game_state is None:
                return

            raw_state = self.env.latest_game_state
            obs = self.env.state_preprocessor.process(raw_state)
            
            # Identify the card we are playing
            card_name = self._get_card_name_in_slot(self.current_slot)
            
            # --- Calculate Grid Cell & Snapping ---
            CAPTURE_LEFT = 2674
            CAPTURE_TOP = 35
            PLAYABLE_LOCAL_Y_MIN = 601
            PLAYABLE_LOCAL_Y_MAX = 1023
            PLAYABLE_LOCAL_X_MIN = 57
            PLAYABLE_W = 654
            PLAYABLE_H = 422
            
            # Grid params matches observation.py logic
            playable_rows = 15
            playable_cols = 18
            cell_w = PLAYABLE_W / playable_cols
            cell_h = PLAYABLE_H / playable_rows
            total_rows = 36
            total_cols = 18

            local_x = x - CAPTURE_LEFT
            local_y = y - CAPTURE_TOP

            # --- Snapping Logic ---
            # Determine clamping bounds based on card type
            card_info = CARD_CONFIG.get(card_name, {})
            card_type = card_info.get('type', 'unit') # Default to unit (safe snapping)
            
            clamp_y_min = PLAYABLE_LOCAL_Y_MIN
            if card_type == 'spell':
                 # Spells can use the whole arena (approx top ~11)
                 clamp_y_min = 11

            clamp_y_max = PLAYABLE_LOCAL_Y_MAX
            clamp_x_min = PLAYABLE_LOCAL_X_MIN
            clamp_x_max = PLAYABLE_LOCAL_X_MIN + PLAYABLE_W
            
            # Check raw validity before snapping
            is_out_of_bounds = not (
                (clamp_y_min <= local_y <= clamp_y_max) and 
                (clamp_x_min <= local_x <= clamp_x_max)
            )
            
            # Snap to bounds
            snapped_local_x = max(clamp_x_min, min(local_x, clamp_x_max))
            snapped_local_y = max(clamp_y_min, min(local_y, clamp_y_max))
            
            # Update global coordinates to snapped values
            snapped_global_x = int(snapped_local_x + CAPTURE_LEFT)
            snapped_global_y = int(snapped_local_y + CAPTURE_TOP)
            
            # Use snapped coords for action recording
            action = (self.current_slot, snapped_global_x, snapped_global_y)
            
            # Recalculate grid indices using snapped coordinates
            rel_x = snapped_local_x - PLAYABLE_LOCAL_X_MIN
            dist_from_bottom = PLAYABLE_LOCAL_Y_MAX - snapped_local_y
            rows_from_bottom = int(dist_from_bottom / cell_h)
            
            gx = int(rel_x / cell_w)
            gy = (total_rows - 1) - rows_from_bottom
            
            # Clamp grid indices for display safety
            gx_safe = max(0, min(gx, total_cols - 1))
            gy_safe = max(0, min(gy, total_rows - 1))
            
            status_str = "VALID"
            if is_out_of_bounds:
                status_str = "SNAPPED"
            
            # Highlight spells in logging and track statistics
            type_marker = ""
            if card_type == 'spell':
                type_marker = " [SPELL]"
                self.spell_count += 1
            elif card_type == 'unit':
                type_marker = " [UNIT]"
                self.unit_count += 1
            
            print(f"[Recorder] ACTION: Place {card_name} (Slot {self.current_slot+1}){type_marker}")
            print(f"           Coords: Global({x}, {y}) -> Snapped({snapped_global_x}, {snapped_global_y})")
            print(f"           Grid: ({gx_safe}, {gy_safe}) [{status_str}]")
            print(f"           Elixir: {raw_state.elixir}")

            self.data.append((obs, action))

    def start(self):
        print("--- CLASH ROYALE DATA RECORDER ---")
        print("1. Focus on the game window.")
        print("2. Play naturally using keys 1-4 and Mouse.")
        print("3. Press 'ESC' to save and exit.")
        print("Waiting for 'GameStart' to begin recording...")

        k_listener = keyboard.Listener(on_press=self.on_key_press)
        m_listener = mouse.Listener(on_click=self.on_click)
        
        k_listener.start()
        m_listener.start()

        self.env.reset()
        
        try:
            while self.running:
                self.env.render()
                state = self.env.latest_game_state
                
                # Hysteresis logic
                if state.game_start and not state.match_over:
                    if not self.recording:
                        print("[Recorder] *** RECORDING STARTED (Match Begin) ***")
                        self.recording = True
                        self.stop_recording_timer = 0.0
                    else:
                        self.stop_recording_timer = 0.0
                else:
                    if self.recording:
                         self.stop_recording_timer += 0.01
                         if self.stop_recording_timer > 3.0:
                             print("[Recorder] ... Match Ended. Saving buffer ...")
                             self.recording = False
                             self.save_buffer()
                             self.stop_recording_timer = 0.0
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            pass
        finally:
            if self.data:
                print("[Recorder] Saving remaining data before exit...")
                self.save_buffer()
            
            self.env.stop()
            k_listener.stop()
            m_listener.stop()

    def stop(self):
        self.running = False

    def save_buffer(self):
        if not self.data:
            print("[Recorder] Buffer empty, nothing to save.")
            return
        
        # Calculate percentages
        total = self.spell_count + self.unit_count
        spell_pct = (100 * self.spell_count / total) if total > 0 else 0
        unit_pct = (100 * self.unit_count / total) if total > 0 else 0
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/match_{timestamp}.npy"
        try:
            np.save(filename, np.array(self.data, dtype=object))
            print(f"[Recorder] Saved {len(self.data)} actions to {filename}")
            print(f"[Recorder] Composition: {self.spell_count} Spells ({spell_pct:.1f}%), {self.unit_count} Units ({unit_pct:.1f}%)")
            if spell_pct < 15:
                print(f"[Recorder] TIP: Try using more spells! Current: {spell_pct:.1f}%, Recommended: 15-25%")
        except Exception as e:
            print(f"[Recorder] ERROR saving file: {e}")
            
        # Reset counters
        self.data = []
        self.spell_count = 0
        self.unit_count = 0

if __name__ == "__main__":
    recorder = DataRecorder()
    recorder.start()
