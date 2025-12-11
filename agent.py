import math
import time
import os
import torch
import numpy as np
import ctypes
from typing import Any, Optional, Dict, Tuple
from interfaces import Agent
from game_state import GameState
from model import ClashRoyaleAgent
from observation import StatePreprocessor, get_card_in_slot
from card_config import get_card_cost, get_card_type
from unit_channels import CHANNEL_COUNT

class ClusteringWrapper:
    def __init__(self):
        try:
            dll_name = "clash_utils_v2.dll" # Updated to v2 to bypass lock
            dll_abspath = os.path.abspath(dll_name)
            dll_dir = os.path.dirname(dll_abspath)
            
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(dll_dir)

            self.lib = ctypes.CDLL(dll_abspath)
            
            self.lib.create_kmeans.argtypes = [ctypes.c_int, ctypes.c_int]
            self.lib.create_kmeans.restype = ctypes.c_void_p
            
            self.lib.destroy_kmeans.argtypes = [ctypes.c_void_p]
            
            self.lib.calculate_best_position.argtypes = [
                ctypes.c_void_p, 
                ctypes.POINTER(ctypes.c_int), 
                ctypes.POINTER(ctypes.c_int), 
                ctypes.c_int, 
                ctypes.POINTER(ctypes.c_int), 
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int)
            ]
            
            # Knapsack bindings
            self.lib.solve_knapsack.argtypes = [
                ctypes.POINTER(ctypes.c_int), # weights
                ctypes.POINTER(ctypes.c_int), # values
                ctypes.c_int,                 # n
                ctypes.c_int,                 # capacity
                ctypes.POINTER(ctypes.c_int), # out_indices
                ctypes.POINTER(ctypes.c_int)  # out_count
            ]
            
            # Initialize with K=3, Iters=10
            self.obj = self.lib.create_kmeans(3, 10)
            print("[C++] Clash Utils (KMeans + Knapsack) initialized.")
        except Exception as e:
            print(f"[C++] Clash Utils loading failed: {e}")
            self.lib = None

    def __del__(self):
        if hasattr(self, 'lib') and self.lib:
            self.lib.destroy_kmeans(self.obj)

    def get_best_cluster_center(self, points: list[tuple[int, int]]) -> tuple[int, int, int]:
        if not self.lib or not points:
            return (0, 0, 0)
        
        n = len(points)
        x_arr = (ctypes.c_int * n)(*[p[0] for p in points])
        y_arr = (ctypes.c_int * n)(*[p[1] for p in points])
        
        out_x = ctypes.c_int()
        out_y = ctypes.c_int()
        out_size = ctypes.c_int()
        
        self.lib.calculate_best_position(self.obj, x_arr, y_arr, n, ctypes.byref(out_x), ctypes.byref(out_y), ctypes.byref(out_size))
        
        return (out_x.value, out_y.value, out_size.value)

    def solve_knapsack(self, cards: list[dict], capacity: int) -> list[int]:
        """
        Solves 0/1 Knapsack problem for card selection.
        cards: list of {'index': i, 'cost': c, 'value': v}
        Returns list of selected INDICES (from the input list).
        """
        if not self.lib or not cards:
            return []
            
        n = len(cards)
        weights = (ctypes.c_int * n)(*[c['cost'] for c in cards])
        values = (ctypes.c_int * n)(*[c['value'] for c in cards])
        
        out_indices = (ctypes.c_int * n)()
        out_count = ctypes.c_int()
        
        max_val = self.lib.solve_knapsack(weights, values, n, capacity, out_indices, ctypes.byref(out_count))
        
        selected = []
        for i in range(out_count.value):
            selected.append(out_indices[i])
            
        print(f"[C++] Knapsack: Capacity {capacity} -> Value {max_val}. Selected {len(selected)} cards.")
        return selected

class NeuralAgent(Agent):
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = StatePreprocessor()
        # Dropout is automatically disabled during eval mode (must match training architecture)
        self.model = ClashRoyaleAgent(scalars_size=self.preprocessor.SCALARS_SIZE, dropout_rate=0.3).to(self.device)
        self.clusterer = ClusteringWrapper() # Handles both KMeans and Knapsack now
        
        # Resolve model path (prefer models/ directory)
        if model_path is None:
            candidates = [
                os.path.join("models", "clash_model_best.pth"),
                os.path.join("models", "clash_model_v1.pth"),
                "clash_model_best.pth",
                "clash_model_v1.pth",
            ]
            model_path = next((p for p in candidates if os.path.exists(p)), None)
        
        if model_path is None:
            raise FileNotFoundError("No model file found (looked for models/clash_model_best.pth, models/clash_model_v1.pth, clash_model_best.pth, clash_model_v1.pth)")
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"[NeuralAgent] Loaded model from {model_path}")
        except Exception as e:
            print(f"[NeuralAgent] ERROR loading model: {e}")
            raise e
        self.last_action_time = 0.0
        self.ACTION_COOLDOWN = 2.0 # Increased to allow elixir accumulation
        
        # Elixir management constants
        self.ELIXIR_RESERVE_THRESHOLD = 5.5  # Keep reserve for defense
        self.ELIXIR_SAFE_PLAY = 7.0  # Safe to play expensive cards
        self.DEFENSE_RADIUS = 250  # Distance to consider enemy as threat
        # Spell block log throttle
        self.last_spell_block_log_time = 0.0
        self.SPELL_BLOCK_LOG_COOLDOWN = 1.5

    def get_action(self, state: GameState) -> Optional[Dict[str, Any]]:
        if state.match_over:
            return None
            
        current_time = time.time()
        if current_time - self.last_action_time < self.ACTION_COOLDOWN:
            return None

        # 1. Preprocess state
        obs_vector = self.preprocessor.process(state)
        
        # 2. Prepare tensors
        SCALARS_SIZE = self.preprocessor.SCALARS_SIZE
        scalars = obs_vector[:SCALARS_SIZE]
        grid_flat = obs_vector[SCALARS_SIZE:]
        
        # Reshape to (Batch=1, Channels, Rows, Cols) for coarse grid 18x9
        grid_tensor = torch.tensor(grid_flat.reshape(1, CHANNEL_COUNT, 18, 9), dtype=torch.float32).to(self.device)
        scalars_tensor = torch.tensor(scalars.reshape(1, -1), dtype=torch.float32).to(self.device)
        
        # 3. Inference
        with torch.no_grad():
            slot_logits, pos_logits = self.model(grid_tensor, scalars_tensor)
            
            # 4. Decode outputs
            slot_idx = torch.argmax(slot_logits, dim=1).item() # 0-3
            pos_flat = torch.argmax(pos_logits, dim=1).item() # 0-647
            
            # Convert flat pos to (gx, gy)
            # Grid is 18 rows, 9 cols
            # pos = gy * 9 + gx
            total_cols = 9
            gy = pos_flat // total_cols
            gx = pos_flat % total_cols
            
        # --- Identify Card & Check Elixir ---
        # Ensure we have cards detected
        if not state.cards:
             # No cards detected (start/end of game)
             return None
             
        current_elixir = state.elixir if state.elixir is not None else 0
        
        # --- KNAPSACK LOGIC (Task 3) ---
        # If we have high elixir, use Knapsack to pick the best "Combo"
        if current_elixir >= 8 and len(state.cards) >= 1:
            # Prepare card data
            knapsack_items = []
            # Iterate through SLOTS (0-3) to ensure we map correctly
            for slot_i in range(4):
                card = get_card_in_slot(state.cards, slot_i)
                if not card or card.is_next:
                     continue
                 
                name = card.class_name
                i = slot_i # Use slot index as identifier
                cost = get_card_cost(name)
                
                # --- Deck-Specific Heuristics (User Deck) ---
                # Base Value = Cost * 10
                value = cost * 10
                
                # Role-based Bonuses
                is_tank = name in ['PekkaDeck', 'BattleRamDeck', 'MyPekka', 'MyBattleRam']
                is_support = name in ['BanditDeck', 'RoyaleGhostDeck', 'MinionsDeck', 'ElectroSpiritDeck']
                is_spell = name in ['ArrowsDeck', 'RageDeck']
                
                # Context 1: Empty Field (Start Push)
                if not state.enemy_units:
                    if name == 'PekkaDeck': value += 30      # Win Con
                    elif name == 'BattleRamDeck': value += 25 # Win Con
                    elif name == 'BanditDeck': value += 15    # Bridge Spam
                    elif name == 'RoyaleGhostDeck': value += 15 
                
                # Context 2: Enemy Presence (Defense)
                else:
                    if name == 'PekkaDeck': value += 20       # Good Tank killer
                    elif name == 'ArrowsDeck': value += 40    # High value defense
                    elif name == 'ElectroSpiritDeck': value += 20 # Stun/Reset
                    elif name == 'MinionsDeck': value += 25   # DPS
                    elif name == 'RageDeck': value += 10      # Boost defense

                knapsack_items.append({'index': i, 'cost': cost, 'value': int(value), 'name': name})
                print(f"   > [Knapsack Calc] {name} (Cost {cost}) -> Value {int(value)}")
            
            # Solve
            selected_indices = self.clusterer.solve_knapsack(knapsack_items, current_elixir)
            
            if selected_indices:
                # Log the optimal set
                clean_names = [knapsack_items[k]['name'].replace('Deck', '') for k in selected_indices]
                total_val = sum(knapsack_items[k]['value'] for k in selected_indices)
                print(f"[Knapsack] Optimal Combo (Cp={current_elixir}): {clean_names} | Total Score: {total_val}")

                # Pick the first one from the optimal set if the Model's choice is NOT in the set
                # Or just prioritize the Knapsack suggestion?
                # Let's see if the Model's predicted 'slot_idx' is in the set.
                if slot_idx not in [knapsack_items[k]['index'] for k in selected_indices]:
                    # Model wants to play a card that is NOT optimal for elixir/value efficiency.
                    # Override with the highest value card from the set.
                    best_k_idx = max(selected_indices, key=lambda k: knapsack_items[k]['value'])
                    best_slot = knapsack_items[best_k_idx]['index']
                    
                    model_card = get_card_in_slot(state.cards, slot_idx)
                    model_card_name = model_card.class_name if model_card else "Unknown"
                    override_card_name = knapsack_items[best_k_idx]['name']
                    
                    print(f"   >>> [Knapsack OVERRIDE] Model chose {model_card_name} (Slot {slot_idx}) -> FORCE {override_card_name} (Slot {best_slot})")
                    slot_idx = best_slot
                else:
                     print(f"   [Knapsack] Model choice (Slot {slot_idx}) agrees with optimal set.")
        
        
        # Find card in the predicted (or overriden) slot
        selected_card = get_card_in_slot(state.cards, slot_idx)
        if selected_card is None:
             # No card found in this slot
             return None
             
        card_name = selected_card.class_name
        
        # Calculate cost
        cost = get_card_cost(card_name)
        current_elixir = state.elixir if state.elixir is not None else 0
        
        # Elixir Check - Basic
        if current_elixir < cost:
             return None
        
        # --- INTELLIGENT ELIXIR MANAGEMENT ---
        # Check if there's an immediate threat
        has_threat = self._has_immediate_threat(state)
        card_type = get_card_type(card_name)
        
        # Don't waste elixir on non-urgent plays if we need to save for defense/expensive cards
        if not has_threat:
            # If elixir is low and this is not a cheap cycle card, skip
            if current_elixir < self.ELIXIR_RESERVE_THRESHOLD and cost >= 3:
                # print(f"[NeuralAgent] Saving elixir ({current_elixir:.1f}) - no threat. Skipping {card_name}.")
                return None
            
            # If we have a high-cost card (like Pekka), save elixir for it
            if self._has_expensive_card_in_hand(state) and current_elixir < self.ELIXIR_SAFE_PLAY and cost >= 3:
                # print(f"[NeuralAgent] Saving for expensive card. Current: {current_elixir:.1f}. Skipping {card_name}.")
                return None
        
        # --- SPELL HEURISTIC CHECKS ---
        if card_type == 'spell':
            # Extract swarm channel (channel 1) from observation grid
            grid_swarm = grid_flat.reshape(CHANNEL_COUNT, 18, 9)[1]  # Channel 1 (coarse grid)
            swarm_count = int(np.sum(grid_swarm))
            
            # Check if there's a swarm cluster (3+ units in any 3x3 area)
            if not self._has_swarm_cluster(grid_swarm, min_count=3):
                # Log sparingly and only if there is at least 1 swarm unit
                if swarm_count > 0:
                    now = time.time()
                    if now - self.last_spell_block_log_time > self.SPELL_BLOCK_LOG_COOLDOWN:
                        print(f"[NeuralAgent] SPELL BLOCKED: {card_name} - Swarm units detected: {swarm_count}, but no cluster of 3+")
                        self.last_spell_block_log_time = now
                return None
            
            # Convert grid prediction to rough pixels first
            pred_px, pred_py = self._grid_to_pixels(gx, gy)
            
            best_px, best_py = self._correct_spell_position_pixels(pred_px, pred_py, state)
            
            print(f"[NeuralAgent] SPELL APPROVED: {card_name}")
            print(f"              Grid: ({gx}, {gy}) -> Pixels: ({pred_px}, {pred_py}) -> Final: ({best_px}, {best_py})")
            
            self.last_action_time = current_time
            return {
                'slot_index': slot_idx,
                'coords': (best_px, best_py) # Explicit pixel coordinates
            }
             
        self.last_action_time = current_time
        
        # Log context for debugging
        total_units_on_grid = int(np.sum(grid_flat.reshape(CHANNEL_COUNT, 18, 9)))
        enemy_unit_count = len(state.enemy_units) if state.enemy_units else 0
        
        print(f"[NeuralAgent] Action: Place {card_name} (Slot {slot_idx+1}) at Grid({gx}, {gy}) | Cost: {cost} | Elixir: {current_elixir:.1f}")
        
        return {
            'slot_index': slot_idx,
            'tile': (gx, gy)
        }
    
    def _has_swarm_cluster(self, grid_channel: np.ndarray, min_count: int = 3) -> bool:
        """
        Check if there's a cluster of min_count+ units in any 3x3 area of the grid.
        grid_channel: 2D array (36, 18) representing unit presence in a single channel.
        """
        rows, cols = grid_channel.shape
        for cy in range(rows):
            for cx in range(cols):
                # Count units in 3x3 window centered at (cy, cx)
                count = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < rows and 0 <= nx < cols:
                            count += int(grid_channel[ny, nx])
                
                if count >= min_count:
                    return True
        return False
    
    def _has_immediate_threat(self, state: GameState) -> bool:
        """Check if enemy units are near our towers."""
        if not state.enemy_units or not state.my_towers:
            return False
        
        for enemy in state.enemy_units:
            enemy_center = self._get_box_center(enemy.box)
            for tower in state.my_towers:
                tower_center = self._get_box_center(tower.box)
                dist = self._get_distance(enemy_center, tower_center)
                if dist < self.DEFENSE_RADIUS:
                    return True
        return False
    
    def _has_expensive_card_in_hand(self, state: GameState) -> bool:
        """Check if we have a 6+ cost card in hand."""
        if not state.cards:
            return False
        for card in state.cards:
            if card.is_next:
                continue
            cost = get_card_cost(card.class_name)
            if cost >= 6:
                return True
        return False
    
    def _grid_to_pixels(self, gx: int, gy: int) -> Tuple[int, int]:
        """Convert grid coordinates back to screen pixels (approx center of tile).
        
        IMPORTANT: The grid is 18x9, but only 9 rows fit in the PLAYABLE_H (422px).
        The cell_h MUST match the encoding in observation.py (422 / 9 = 46.9px).
        """
        PLAYABLE_LOCAL_Y_MAX = 1023
        PLAYABLE_LOCAL_X_MIN = 57
        PLAYABLE_W = 654
        PLAYABLE_H = 422
        
        # Grid dimensions
        GRID_W = 9
        GRID_H = 18
        PLAYABLE_ROWS = 9  # Only 9 rows fit in the playable area height
        
        cell_w = PLAYABLE_W / GRID_W
        cell_h = PLAYABLE_H / PLAYABLE_ROWS  # CRITICAL: Must match observation.py encoding!

        # gx is 0-8, gy is 0-17 
        # gy 0 is TOP (enemy territory), 17 is BOTTOM (our side)
        
        # Calculate offset from bottom
        # If gy=17 (Bottom), rows_from_bottom=0
        # If gy=0 (Top), rows_from_bottom=17
        rows_from_bottom = (GRID_H - 1) - gy
        
        rel_x = (gx + 0.5) * cell_w
        rel_y_from_bottom = (rows_from_bottom + 0.5) * cell_h
        
        screen_x = int(PLAYABLE_LOCAL_X_MIN + rel_x)
        screen_y = int(PLAYABLE_LOCAL_Y_MAX - rel_y_from_bottom)
        
        return (screen_x, screen_y)

    def _correct_spell_position_pixels(self, px: int, py: int, state: GameState) -> Tuple[int, int]:
        """
        Smart spell targeting using C++ K-Means with PIXEL coordinates.
        But restricted to the area predicted by the Model to avoid cross-lane targeting errors.
        INCLUDES MANUAL OFFSET (User request: -50px).
        """
        OFFSET_Y = -71
        
        final_x, final_y = px, py

        if state.enemy_units:
            # Filter enemies: Only consider those close to the model's prediction
            # Radius ~350px covers a good chunk of a lane/bridge area
            SEARCH_RADIUS = 350
            
            enemy_pts = []
            for enemy in state.enemy_units:
                ex, ey = self._get_box_center(enemy.box)
                dist = math.sqrt((ex - px)**2 + (ey - py)**2)
                if dist <= SEARCH_RADIUS:
                    enemy_pts.append((int(ex), int(ey)))

            # Use C++ to find best cluster center among RELEVANT units
            if len(enemy_pts) >= 2:
                best_x, best_y, size = self.clusterer.get_best_cluster_center(enemy_pts)
                if size >= 2:
                    print(f"[C++ KMeans] Refined target: ({px},{py}) -> ({best_x}, {best_y}) | Units in range: {len(enemy_pts)}")
                    final_x, final_y = best_x, best_y
        
        # Apply manual offset
        final_y += OFFSET_Y
        
        return (final_x, final_y)

    
    def _get_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _get_box_center(self, box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

class RuleBasedAgent(Agent):
    def __init__(self):
        # Constants for decision making
        self.DEFENSE_RADIUS = 300
        self.SAFE_PLAY_COORDS = (515, 650)
        self.BRIDGE_ATTACK_COORDS = (515, 610)
        self.TANK_CLASSES = ['MyPekka', 'MyBarbarian']
        
        # Logging throttle
        self.last_log_time = {}
        self.LOG_COOLDOWN = 1.5 # Seconds to suppress duplicate logs

    def get_action(self, state: GameState) -> Optional[Dict[str, Any]]:
        """
        Decides on an action based on simple rules.
        Returns None if no action is to be taken.
        """
        if state.match_over:
            return None
        
        # Note: We removed the 'state.game_start' check here because the Environment 
        # clears this flag on reset(), but main.py handles the IDLE state. 
        # If we are calling get_action, we assume we are in PLAYING state.

        # --- Priority 1: DEFENSE ---
        if state.enemy_units:
            for enemy in state.enemy_units:
                enemy_center = self._get_box_center(enemy.box)
                for tower in state.my_towers:
                    tower_center = self._get_box_center(tower.box)
                    if self._get_distance(enemy_center, tower_center) < self.DEFENSE_RADIUS:
                        if state.cards:
                            self._log_throttled(f"[Agent] DEFENSE! Enemy {enemy.class_name} near. Playing card 0.")
                            return {'slot_index': 0, 'coords': (int(enemy_center[0]), int(enemy_center[1]))}

        # --- Priority 2: ATTACK SUPPORT ---
        if state.my_units and state.elixir and state.elixir >= 4:
            front_unit = max(state.my_units, key=lambda u: self._get_box_center(u.box)[1])
            front_unit_coords = self._get_box_center(front_unit.box)
            if front_unit_coords[1] > 480: # Bridge Y coordinate
                if state.cards:
                    self._log_throttled(f"[Agent] ATTACK SUPPORT! Adding unit to {front_unit.class_name}.")
                    return {'slot_index': 1, 'coords': (int(front_unit_coords[0]), int(front_unit_coords[1] + 50))}

        # --- Priority 3: START ATTACK ---
        if state.elixir and state.elixir >= 8:
            if state.cards:
                # Look for tank
                tank_card_index = -1
                for i, card in enumerate(state.cards):
                    if not card.is_next and card.class_name.replace('Deck', '') in self.TANK_CLASSES:
                        tank_card_index = i
                        break
                
                if tank_card_index != -1:
                    self._log_throttled(f"[Agent] START ATTACK (Tank)! Elixir {state.elixir}. Playing tank safely.")
                    return {'slot_index': tank_card_index, 'coords': self.SAFE_PLAY_COORDS}
                else:
                    self._log_throttled(f"[Agent] START ATTACK (Cycle)! Elixir {state.elixir}. Playing card 2 at bridge.")
                    return {'slot_index': 2, 'coords': self.BRIDGE_ATTACK_COORDS}

        return None

    def _log_throttled(self, message: str):
        """Prints message only if LOG_COOLDOWN seconds have passed since last same message."""
        current_time = time.time()
        if message not in self.last_log_time or (current_time - self.last_log_time[message] > self.LOG_COOLDOWN):
            print(message)
            self.last_log_time[message] = current_time

    def _get_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def _get_box_center(self, box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
