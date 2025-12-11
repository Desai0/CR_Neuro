import numpy as np
from game_state import GameState
import math
from unit_channels import get_unit_channel, CHANNEL_COUNT
from card_config import CARD_CONFIG

# Card vocabulary derived from card_config to keep a stable ordering
CARD_LIST = sorted(CARD_CONFIG.keys())
CARD_TO_IDX = {name: idx for idx, name in enumerate(CARD_LIST)}
CARD_VOCAB_SIZE = len(CARD_LIST)

# Slot centers in LOCAL vision coordinates (must match recorder.py)
SLOT_CENTERS_LOCAL = [
    (226, 1206),  # Slot 1
    (376, 1206),  # Slot 2
    (526, 1206),  # Slot 3
    (676, 1206),  # Slot 4
]

def get_card_in_slot(cards, slot_idx: int):
    """
    Finds the card closest to the given slot position.
    Returns the card object or None if not found.
    """
    if not cards or slot_idx < 0 or slot_idx >= 4:
        return None
    
    target_center = SLOT_CENTERS_LOCAL[slot_idx]
    min_dist = float('inf')
    best_card = None
    
    for card in cards:
        if card.is_next:  # Skip "next" card
            continue
        # Card box center
        cx = (card.box[0] + card.box[2]) / 2
        cy = (card.box[1] + card.box[3]) / 2
        
        dist = math.hypot(cx - target_center[0], cy - target_center[1])
        
        # Threshold: Card must be reasonably close (within 100px)
        if dist < 100 and dist < min_dist:
            min_dist = dist
            best_card = card
    
    return best_card

class StatePreprocessor:
    def __init__(self):
        # --- Настройки вектора ---
        # numpy shape: (Channels, Rows, Cols)
        self.GRID_CHANNELS = CHANNEL_COUNT
        self.GRID_ROWS = 18  # Coarse grid rows (was 36)
        self.GRID_COLS = 9   # Coarse grid cols (was 18)
        self.GRID_SHAPE = (self.GRID_CHANNELS, self.GRID_ROWS, self.GRID_COLS)
        
        self.N_CARD_SLOTS = 4
        self.N_CARD_CLASSES = CARD_VOCAB_SIZE
        
        # Размер grid_flat будет: CHANNEL_COUNT * 36 * 18
        self.GRID_FLAT_SIZE = self.GRID_CHANNELS * self.GRID_ROWS * self.GRID_COLS
        # Scalars: elixir (1) + my towers (3) + enemy towers (3) + one-hot cards (4 * vocab)
        self.SCALARS_SIZE = 1 + 3 + 3 + self.N_CARD_SLOTS * self.N_CARD_CLASSES

    def process(self, state: GameState) -> np.ndarray:
        """
        Превращает GameState в вектор чисел (numpy array).
        """
        # 1. Глобальные параметры
        elixir_val = state.elixir if state.elixir is not None else 5
        norm_elixir = elixir_val / 10.0
        
        global_features = [norm_elixir]

        # 2. Башни (HP нормализованное)
        def encode_towers(towers):
            sorted_towers = sorted(towers, key=lambda t: t.box[0])
            features = []
            for i in range(3):
                if i < len(sorted_towers):
                    health = sorted_towers[i].health
                    if health is not None:
                        features.append(health / 4000.0)
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)
            return features

        my_tower_feats = encode_towers(state.my_towers)
        enemy_tower_feats = encode_towers(state.enemy_towers)

        # 3. Многоканальная сетка юнитов (Grid)
        grid = np.zeros(self.GRID_SHAPE, dtype=np.float32)
        
        # Параметры для расчета координат (coarse grid 18x9)
        PLAYABLE_LOCAL_Y_MAX = 1023
        PLAYABLE_LOCAL_X_MIN = 57
        PLAYABLE_W = 654
        PLAYABLE_H = 422
        
        playable_rows = 9   # coarse playable rows
        playable_cols = 9   # coarse playable cols
        
        cell_w = PLAYABLE_W / playable_cols
        cell_h = PLAYABLE_H / playable_rows
        
        total_rows = self.GRID_ROWS
        total_cols = self.GRID_COLS
        
        def fill_grid_channels(units):
            for unit in units:
                cx = (unit.box[0] + unit.box[2]) / 2
                cy = (unit.box[1] + unit.box[3]) / 2
                
                # Координаты
                rel_x = cx - PLAYABLE_LOCAL_X_MIN
                dist_from_bottom = PLAYABLE_LOCAL_Y_MAX - cy
                rows_from_bottom = int(dist_from_bottom / cell_h)
                
                gx = int(rel_x / cell_w)
                gy = (total_rows - 1) - rows_from_bottom
                
                # Clamp
                gx = max(0, min(gx, total_cols - 1))
                gy = max(0, min(gy, total_rows - 1))
                
                # Определяем канал
                channel_idx = get_unit_channel(unit.class_name)
                
                # Пишем 1.0 в нужный канал
                grid[channel_idx, gy, gx] += 1.0

        fill_grid_channels(state.my_units)
        fill_grid_channels(state.enemy_units)
        # Также можно добавить сломанные башни, если нужно, но пока пропустим
        
        # Debug: Log grid stats (uncomment to verify grid is being filled)
        # total_units = np.sum(grid)
        # if total_units > 0:
        #     print(f"[StatePreprocessor] Grid filled with {total_units:.0f} units across {CHANNEL_COUNT} channels")
        #     for ch in range(CHANNEL_COUNT):
        #         ch_sum = np.sum(grid[ch])
        #         if ch_sum > 0:
        #             print(f"  Channel {ch}: {ch_sum:.0f} units")
        
        grid_flat = grid.flatten().tolist()

        # 4. Карты в руке (one-hot по каждому слоту, по РЕАЛЬНОЙ позиции)
        hand_onehot = np.zeros((self.N_CARD_SLOTS, self.N_CARD_CLASSES), dtype=np.float32)
        for slot_idx in range(self.N_CARD_SLOTS):
            card = get_card_in_slot(state.cards, slot_idx)
            if card is not None:
                idx = CARD_TO_IDX.get(card.class_name)
                if idx is not None:
                    hand_onehot[slot_idx, idx] = 1.0
        hand_features = hand_onehot.flatten().tolist()

        # Собираем все вместе
        full_vector = np.array(
            global_features + 
            my_tower_feats + 
            enemy_tower_feats + 
            hand_features + 
            grid_flat, 
            dtype=np.float32
        )
        
        return full_vector
