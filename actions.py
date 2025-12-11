import pyautogui
import time
import random
from typing import Dict, Any, Tuple

class ActionController:
    def __init__(self):
        self.card_slots = [
            (2900, 1241), # Slot 1
            (3050, 1241), # Slot 2
            (3200, 1241), # Slot 3
            (3350, 1241)  # Slot 4
        ]
        self.playable_area = (2731, 636, 3385, 1058)
        self.vision_capture_area = {"top": 35, "left": 2674, "width": 766, "height": 1355}
        
        # Safety settings
        pyautogui.PAUSE = 0.1

    def execute(self, action: Dict[str, Any]) -> bool:
        """
        Executes the given action.
        Expected action format: 
            - {'slot_index': int, 'coords': (x, y)}  (Exact coordinates)
            - {'slot_index': int, 'tile': (tx, ty)}  (Grid coordinates 15x18)
        """
        if action is None:
            return False

        slot_index = action.get('slot_index')
        
        if slot_index is None:
            print(f"[ActionController] Invalid action format (no slot): {action}")
            return False

        final_coords = None
        
        # Case 1: Grid Tile
        if 'tile' in action:
            tile_x, tile_y = action['tile']
            final_coords = self.grid_to_global_coords(tile_x, tile_y)
            
        # Case 2: Exact Coords (Local Vision Coords)
        elif 'coords' in action:
            local_coords = action.get('coords')
            global_coords = self._convert_vision_to_global_coords(local_coords)
            final_coords = global_coords

        if final_coords is None:
             print(f"[ActionController] Invalid action format (no coords/tile): {action}")
             return False

        # Clamp coordinates to playable area (ONLY if not a spell - but here we don't know card type)
        # TODO: Pass card type to execute() to allow spells outside playable area.
        # For now, we trust the grid_to_global_coords produced valid coords, 
        # but we might want to relax clamping for the top of the screen.
        
        # clamped_coords = self._clamp_coords_to_playable_area(final_coords)
        # UPDATE: We disable clamping here because 'grid_to_global_coords' now supports full screen.
        # We rely on the Agent/Model to predict valid locations.
        clamped_coords = final_coords 
        
        if final_coords != clamped_coords:
            # print(f"[ActionController] Coords clamped: {final_coords} -> {clamped_coords}")
            pass

        return self._play_card(slot_index, clamped_coords)

    def grid_to_global_coords(self, gx: int, gy: int) -> Tuple[int, int]:
        """
        Converts grid tile (gx: 0..8, gy: 0..17) to global screen coordinates.
        Matches the 18x9 grid defined in observation.py.
        gy=0 is TOP of screen. gy=17 is BOTTOM of screen.
        """
        PLAYABLE_W = 654
        PLAYABLE_H = 422
        PLAYABLE_LOCAL_X_MIN = 57
        PLAYABLE_LOCAL_Y_MAX = 1023  # Bottom of playable area
        
        GRID_W = 9
        GRID_H = 18
        PLAYABLE_ROWS = 9
        
        cell_w = PLAYABLE_W / GRID_W
        cell_h = PLAYABLE_H / PLAYABLE_ROWS
        
        rows_from_bottom = (GRID_H - 1) - gy
        
        rel_x = (gx + 0.5) * cell_w
        dist_from_bottom = (rows_from_bottom + 0.5) * cell_h
        
        local_x = int(PLAYABLE_LOCAL_X_MIN + rel_x)
        local_y = int(PLAYABLE_LOCAL_Y_MAX - dist_from_bottom)
        
        return self._convert_vision_to_global_coords((local_x, local_y))

    def _convert_vision_to_global_coords(self, local_coords: Tuple[int, int]) -> Tuple[int, int]:
        local_x, local_y = local_coords
        global_x = local_x + self.vision_capture_area["left"]
        global_y = local_y + self.vision_capture_area["top"]
        return (global_x, global_y)

    def _clamp_coords_to_playable_area(self, coords: Tuple[int, int]) -> Tuple[int, int]:
        x, y = coords
        x_min, y_min, x_max, y_max = self.playable_area
        
        clamped_x = max(x_min, min(x, x_max))
        clamped_y = max(y_min, min(y, y_max))
        
        return (clamped_x, clamped_y)

    def _is_within_playable_area(self, coords: Tuple[int, int]) -> bool:
        x, y = coords
        x_min, y_min, x_max, y_max = self.playable_area
        return x_min <= x <= x_max and y_min <= y <= y_max

    def _play_card(self, card_slot_index: int, placement_coords: Tuple[int, int]) -> bool:
        # Validate inputs
        if not 0 <= card_slot_index < len(self.card_slots):
            print(f"[ActionController] Invalid card slot index: {card_slot_index}")
            return False

        # We removed the strict _is_within_playable_area check because the Model now can predict
        # actions anywhere on the screen (e.g. spells).
        # The game itself will handle invalid placements (red zone).
        # if not self._is_within_playable_area(placement_coords):
        #    print(f"[ActionController] Target coordinates {placement_coords} out of bounds.")
        #    return False

        # Execute click sequence
        card_coords = self.card_slots[card_slot_index]
        
        print(f"[Action] Playing card from slot {card_slot_index+1} at {placement_coords}")
        try:
            pyautogui.click(card_coords)
            pyautogui.click(placement_coords)
            return True
        except Exception as e:
            print(f"[ActionController] Error simulating input: {e}")
            return False

