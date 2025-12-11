import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
from game_state import GameState, Tower, Unit, Card
from typing import Tuple, Optional, List, Any, Dict
import traceback
import pytesseract

class VisionSystem:
    def __init__(self, model_path: str = "runs/detect/train6/weights/best.pt"):
        # --- OCR Configuration ---
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # --- YOLO Configuration ---
        self.model = YOLO(model_path)
        
        # --- Class Definitions ---
        self.ally_tower_classes = ['MyPrincessTower', 'MyKingTower']
        self.enemy_tower_classes = ['PrincessTower', 'KingTower', 'EnemyTower']
        self.tower_classes = self.ally_tower_classes + self.enemy_tower_classes

        # Добавляем классы уничтоженных башен (предполагаемые имена)
        self.ally_broken_tower_classes = ['MyPrincessTowerBrocken', 'MyKingTowerBrocken'] 
        self.enemy_broken_tower_classes = ['PrincessTowerBrocken', 'KingTowerBrocken', 'EnemyTowerBrocken']

        self.hp_classes = ['TowerPrincessHP', 'MyPrincessTowerHP', 'MyKingHP', 'KingTowerHP']
        self.ally_hp_classes = ['MyPrincessTowerHP', 'MyKingHP']
        self.enemy_hp_classes = ['TowerPrincessHP', 'KingTowerHP']

        # These depend on model names, so we initialize them here
        self.card_classes = [cls for cls in self.model.names.values() if 'Deck' in cls or 'Next' in cls or cls == 'Empty']
        
        self.my_unit_classes = [
            'MyBandit', 'MyBarbarian', 'MyBattleRam', 'MyElectroSpirit', 
            'MyMinion', 'MyPekka', 'MyRoyaleGhost'
        ]
        
        self.spell_classes = ['Rage', 'Arrows', 'FireBall']
        
        # Список исключений для юнитов
        excluded_from_units = (
            self.tower_classes + 
            self.ally_broken_tower_classes + 
            self.enemy_broken_tower_classes + 
            self.hp_classes + 
            self.card_classes + 
            self.spell_classes + 
            ['Elixir', 'GameStart', 'MatchOver']
        )

        # Обновляем unit_classes
        self.unit_classes = [
            cls for cls in self.model.names.values() 
            if cls not in excluded_from_units
        ]
        
        self.hp_to_tower_map = {
            'MyPrincessTowerHP': 'MyPrincessTower',
            'MyKingHP': 'MyKingTower',
            'KingTowerHP': 'KingTower',
            'TowerPrincessHP': 'PrincessTower'
        }

        # --- State Management ---
        self.last_game_state: Optional[GameState] = None
        self.frame_counter = 0
        self.slow_analysis_rate = 5
        self.latest_results = None
        self.debug_elixir_mask = None
        
        # --- Stability Filters (Hysteresis) ---
        self.match_over_frames_counter = 0
        self.MATCH_OVER_THRESHOLD = 5 # Reduced from 10 to 5 for faster detection

        self.game_start_frames_counter = 0
        self.GAME_START_THRESHOLD = 10 # Increased for stability
        self.GAME_START_MIN_CONF = 0.55 # Require higher confidence to avoid false positives
        self.DECK_MISSING_TIMEOUT = 3.5 # Require longer deck absence to declare MatchOver fallback
        
        # --- Match Flow Control (Fallback) ---
        self.is_in_game = False
        self.last_full_deck_time = 0.0
        
        # --- Elixir Settings ---
        self.MAX_ELIXIR_WIDTH_PX = 555 

    def reset(self):
        """Сбрасывает состояние зрения (забывает предыдущие кадры)."""
        self.last_game_state = None
        self.frame_counter = 0
        self.latest_results = None
        self.match_over_frames_counter = 0
        self.game_start_frames_counter = 0
        self.is_in_game = False
        self.last_full_deck_time = time.time()
        print("[Vision] State reset.")

    def process_frame(self, frame: np.ndarray) -> GameState:
        """
        Main perception function.
        """
        try:
            results = self.model.predict(source=frame, conf=0.3, verbose=False, tracker='bytetrack.yaml')
            self.latest_results = results 
            
            current_game_state = GameState()
            
            # --- 1. Inherit States (Optional) ---
            if self.last_game_state and self.last_game_state.elixir is not None:
                current_game_state.elixir = self.last_game_state.elixir
            else:
                 current_game_state.elixir = 5 

            # --- 2. Slow Analysis (OCR) ---
            analyze_slow = (self.frame_counter % self.slow_analysis_rate == 0)

            if analyze_slow:
                all_towers = self._get_tower_health(results, frame)
                current_game_state.my_towers = [t for t in all_towers if t.class_name in self.ally_tower_classes]
                current_game_state.enemy_towers = [t for t in all_towers if t.class_name in self.enemy_tower_classes]
            elif self.last_game_state is not None:
                current_game_state.my_towers = self.last_game_state.my_towers
                current_game_state.enemy_towers = self.last_game_state.enemy_towers

            # --- 3. Fast Analysis (YOLO Classes) ---
            if results[0].boxes is not None:
                self._process_detections(results, current_game_state, frame)

            # --- 4. Apply Stability Filters ---
            self._update_match_over_filter(current_game_state)
            self._update_game_start_filter(current_game_state)

            # --- 5. Fallback MatchOver Logic (Missing Deck) ---
            current_time = time.time()
            
            # Track deck visibility
            # We count all card classes (Deck + Next + Empty)
            card_count = len(current_game_state.cards)
            if card_count >= 3:  # consider 3+ as "deck visible" to be more forgiving
                self.last_full_deck_time = current_time
            
            # Update internal 'in_game' state based on GameStart banner
            if current_game_state.game_start:
                self.is_in_game = True
                # Reset deck timer when game starts to avoid immediate timeout if deck loads slowly
                if not self.last_game_state or not self.last_game_state.game_start:
                    self.last_full_deck_time = current_time
            
            # Fallback: If we see a full deck, we are definitely in game
            if card_count >= 4:
                self.is_in_game = True
                
            # Check for Match End
            if current_game_state.match_over:
                self.is_in_game = False
            elif self.is_in_game:
                # Fallback only if deck fully missing (<=1 card) for a sustained period
                deck_missing_duration = current_time - self.last_full_deck_time
                deck_missing = card_count <= 1
                if deck_missing and deck_missing_duration > self.DECK_MISSING_TIMEOUT:
                    print(f"[Vision] MatchOver fallback: Deck missing for {deck_missing_duration:.1f}s (cards={card_count})")
                    current_game_state.match_over = True
                    current_game_state.game_start = False
                    self.is_in_game = False
                    # Boost counter to avoid immediate clearing on the next frame
                    self.match_over_frames_counter = max(self.match_over_frames_counter, self.MATCH_OVER_THRESHOLD)

            self.last_game_state = current_game_state
            self.frame_counter += 1
            
            return current_game_state
            
        except Exception as e:
            print(f"[Vision Error] {e}")
            traceback.print_exc()
            return self.last_game_state if self.last_game_state else GameState()

    def _update_match_over_filter(self, state: GameState):
        """Updates match_over status based on counter history."""
        if state.match_over: # Raw detection from current frame
             self.match_over_frames_counter += 1
             self.match_over_frames_counter = min(self.match_over_frames_counter, self.MATCH_OVER_THRESHOLD + 20)
        else:
             self.match_over_frames_counter -= 1
             self.match_over_frames_counter = max(0, self.match_over_frames_counter)
        
        # Determine final state
        if self.match_over_frames_counter >= self.MATCH_OVER_THRESHOLD:
            if not (self.last_game_state and self.last_game_state.match_over):
                 print(f"[Vision] Confirmed MatchOver (Counter: {self.match_over_frames_counter})")
            state.match_over = True
        else:
            if self.last_game_state and self.last_game_state.match_over:
                 print(f"[Vision] MatchOver cleared (Counter: {self.match_over_frames_counter})")
            state.match_over = False

    def _update_game_start_filter(self, state: GameState):
        """Updates game_start status based on counter history."""
        if state.game_start: # Raw detection
             self.game_start_frames_counter += 1
             self.game_start_frames_counter = min(self.game_start_frames_counter, self.GAME_START_THRESHOLD + 20)
        else:
             # If fallback 'is_in_game' is active, we don't let the counter drop to zero immediately
             # unless MatchOver is detected. But here we just manage the visual detection counter.
             self.game_start_frames_counter -= 1
             self.game_start_frames_counter = max(0, self.game_start_frames_counter)
             
        if self.game_start_frames_counter >= self.GAME_START_THRESHOLD:
            if not (self.last_game_state and self.last_game_state.game_start):
                 print(f"[Vision] Confirmed GameStart (Counter: {self.game_start_frames_counter})")
            state.game_start = True
            self.is_in_game = True # LATCH: Once confirmed, we are IN GAME.
        else:
            # Only reset game_start if we are NOT in the latched state
            # This allows the 'GameStart' flag to remain True even if the banner disappears,
            # as long as we haven't seen MatchOver.
            if self.is_in_game:
                state.game_start = True # Keep it True
            else:
                state.game_start = False

    def _process_detections(self, results, current_game_state: GameState, frame: np.ndarray):
        for box in results[0].boxes:
            class_name = self.model.names[int(box.cls[0])]
            coords = tuple(int(i) for i in box.xyxy[0])
            
            if class_name == 'GameStart':
                # Class-specific confidence gate to suppress false positives
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                if conf < self.GAME_START_MIN_CONF:
                    continue
                # Filter out GameStart if it's detected in the bottom 20% of the screen (likely "Battle" button)
                # Frame height is ~1355. 
                h, w, _ = frame.shape
                y_center = (coords[1] + coords[3]) / 2
                if y_center > h * 0.8:
                    # print(f"[Vision] Ignored GameStart at bottom: {coords}")
                    continue
                
                # Fix for flickering:
                # If we are already in a confirmed GameStart state (counter high), 
                # do not reset it unless we are sure match is over.
                # But here we just set the RAW flag.
                current_game_state.game_start = True 
                
                if current_game_state.elixir is None:
                     current_game_state.elixir = 7
                continue
            elif class_name == 'MatchOver':
                # 1. Geometric Check (Sanity)
                h, w, _ = frame.shape
                
                # 2. OCR Verification
                # Extract ROI
                x1, y1, x2, y2 = coords
                # Clamp
                x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
                roi = frame[y1:y2, x1:x2]
                
                if roi.size == 0: continue

                # Preprocess for text (White text)
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                
                # OCR
                try:
                    # psm 7 = Single line
                    text = pytesseract.image_to_string(thresh, config='--psm 7').strip().upper()
                    
                    if "MATCH" in text or "OVER" in text:
                        current_game_state.match_over = True
                        current_game_state.game_start = False
                        # print(f"[Vision] Verified MatchOver: {text}")
                    else:
                        # print(f"[Vision] Ignored MatchOver (Text mismatch): '{text}'")
                        pass
                except Exception as e:
                     print(f"[Vision OCR Error] {e}")
                
                continue
            
            # Elixir detection (COLOR BASED)
            elif class_name == 'Elixir':
                x1, y1, x2, y2 = coords
                h, w, _ = frame.shape
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
                if x2 <= x1 or y2 <= y1: continue

                h_box = y2 - y1
                new_y1 = int(y1 + h_box * 0.63)
                new_y2 = int(y1 + h_box * 0.65)
                if new_y2 <= new_y1: new_y2 = new_y1 + 1
                new_y1 = max(0, new_y1); new_y2 = min(h, new_y2)

                elixir_roi = frame[new_y1:new_y2, x1:x2]
                
                try:
                    if elixir_roi.size > 0:
                        hsv = cv2.cvtColor(elixir_roi, cv2.COLOR_BGR2HSV)
                        lower_pink = np.array([125, 50, 50]) 
                        upper_pink = np.array([179, 255, 255])
                        mask = cv2.inRange(hsv, lower_pink, upper_pink)
                        self.debug_elixir_mask = mask
                        
                        col_sums = np.sum(mask, axis=0)
                        pink_indices = np.where(col_sums > 0)[0]
                        
                        if len(pink_indices) > 0:
                            rightmost_pink_x = pink_indices[-1]
                            roi_width = elixir_roi.shape[1]
                            if roi_width > 0:
                                percentage = rightmost_pink_x / roi_width
                                adjusted_percentage = min(1.0, percentage * 1.05)
                                elixir_val = int(round(adjusted_percentage * 10))
                                elixir_val = max(0, min(10, elixir_val))
                                current_game_state.elixir = elixir_val
                except Exception:
                    pass

            elif class_name in self.ally_broken_tower_classes:
                current_game_state.my_broken_towers.append(Unit(class_name=class_name, box=coords))
            elif class_name in self.enemy_broken_tower_classes:
                current_game_state.enemy_broken_towers.append(Unit(class_name=class_name, box=coords))
            elif class_name in self.unit_classes:
                if class_name in self.my_unit_classes:
                    current_game_state.my_units.append(Unit(class_name=class_name, box=coords))
                else:
                    current_game_state.enemy_units.append(Unit(class_name=class_name, box=coords))
            elif class_name in self.card_classes:
                is_next = 'Next' in class_name
                current_game_state.cards.append(Card(class_name=class_name, box=coords, is_next=is_next))

    def _preprocess_for_ocr(self, image, is_ally=False, is_elixir=False):
        if image.size == 0: return None
        if is_elixir:
            height, width = image.shape[:2]
            image = cv2.resize(image, (width*3, height*3), interpolation=cv2.INTER_CUBIC)
        else:
            height, width = image.shape[:2]
            scale_factor = max(2, 30 / height)
            image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold_value = 200 if is_elixir or not is_ally else 170
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        return thresh

    def _get_tower_health(self, yolo_results, frame):
        towers = {cls: [] for cls in self.tower_classes}
        hp_regions = {cls: [] for cls in self.hp_classes}

        if yolo_results[0].boxes is not None:
            for box in yolo_results[0].boxes:
                class_id = int(box.cls[0])
                class_name = yolo_results[0].names[class_id]
                coords = [int(i) for i in box.xyxy[0]]
                if class_name in self.tower_classes:
                    towers[class_name].append(coords)
                elif class_name in self.hp_classes:
                    hp_regions[class_name].append(coords)

        towers_with_health = []
        for hp_class, tower_class in self.hp_to_tower_map.items():
            for hp_box in hp_regions[hp_class]:
                closest_tower_box = None
                min_dist = float('inf')
                possible_towers = towers.get(tower_class, [])
                if tower_class == 'PrincessTower':
                    possible_towers.extend(towers.get('EnemyTower', []))

                hp_center = ((hp_box[0] + hp_box[2]) / 2, (hp_box[1] + hp_box[3]) / 2)
                for tower_box in possible_towers:
                    tower_center = ((tower_box[0] + tower_box[2]) / 2, (tower_box[1] + tower_box[3]) / 2)
                    dist = math.dist(hp_center, tower_center)
                    if dist < min_dist:
                        min_dist = dist
                        closest_tower_box = tower_box
                
                if closest_tower_box and min_dist < 250:
                    x1_hp, y1_hp, x2_hp, y2_hp = hp_box
                    hp_crop = None
                    
                    if hp_class in ['MyKingHP', 'KingTowerHP']: crop_right = 32
                    elif hp_class == 'MyPrincessTowerHP': crop_right = 27
                    else: crop_right = 20
                    new_x2_hp = x2_hp - crop_right

                    if hp_class == 'MyKingHP': crop_left = 55
                    elif hp_class == 'KingTowerHP': crop_left = 35 + 25 - 3 
                    elif hp_class == 'MyPrincessTowerHP': crop_left = 30
                    else: crop_left = 20 + 13 - 3

                    new_x1 = x1_hp + crop_left
                    final_crop_coords = None
                    
                    if hp_class in self.ally_hp_classes:
                        crop_top = 12
                        new_y1 = y1_hp + crop_top
                        if new_x1 < new_x2_hp and new_y1 < y2_hp:
                            hp_crop = frame[new_y1:y2_hp, new_x1:new_x2_hp]
                            final_crop_coords = (new_x1, new_y1, new_x2_hp, y2_hp)
                    else:
                        base_crop_bottom = 10
                        crop_bottom = base_crop_bottom + 10 if hp_class == 'KingTowerHP' else base_crop_bottom + 5
                        new_y2 = y2_hp - crop_bottom
                        if new_x1 < new_x2_hp and y1_hp < new_y2:
                            hp_crop = frame[y1_hp:new_y2, new_x1:new_x2_hp]
                            final_crop_coords = (new_x1, y1_hp, new_x2_hp, new_y2)

                    health_value = None
                    if hp_crop is not None:
                        is_ally_hp = hp_class in self.ally_hp_classes
                        custom_config = r'--psm 7 -c tessedit_char_whitelist=0123456789'
                        try:
                            processed_image = self._preprocess_for_ocr(hp_crop, is_ally=is_ally_hp)
                            text = pytesseract.image_to_string(processed_image, config=custom_config)
                            health_value = int(text.strip())
                        except (ValueError, TypeError):
                            health_value = None
                    
                    towers_with_health.append(Tower(class_name=tower_class, box=closest_tower_box, health=health_value, hp_box=final_crop_coords))

        return towers_with_health
