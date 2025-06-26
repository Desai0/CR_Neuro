# This file will contain the "brain" of the agent.
# It will take a GameState object and decide which action to take.
# Initially, this will be a simple rule-based system. 

from game_state import GameState
import math

# --- Константы для принятия решений ---
DEFENSE_RADIUS = 300
SAFE_PLAY_COORDS = (515, 650) 
# Смещаем точку атаки немного вниз, чтобы она попадала в игровую зону
BRIDGE_ATTACK_COORDS = (515, 610)
TANK_CLASSES = ['MyPekka', 'MyGiant', 'MyGolem', 'MyBarbarian'] 

def _get_distance(point1, point2):
    """Вычисляет евклидово расстояние между двумя точками."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def _get_box_center(box):
    """Находит центр рамки (bounding box)."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def choose_action_rule_based(game_state: GameState):
    """
    Принимает решение о следующем действии на основе простых правил.
    """
    
    # --- Приоритет №1: ЗАЩИТА ---
    if game_state.enemy_units:
        for enemy in game_state.enemy_units:
            enemy_center = _get_box_center(enemy.box)
            for tower in game_state.my_towers:
                tower_center = _get_box_center(tower.box)
                if _get_distance(enemy_center, tower_center) < DEFENSE_RADIUS:
                    if game_state.cards:
                        print(f"[Brain] ЗАЩИТА! Враг {enemy.class_name} близко. Играем карту 0 на него.")
                        return {'slot_index': 0, 'coords': (int(enemy_center[0]), int(enemy_center[1]))}

    # --- Приоритет №2: ПОДДЕРЖКА АТАКИ ---
    if game_state.my_units and game_state.elixir and game_state.elixir >= 4:
        front_unit = max(game_state.my_units, key=lambda u: _get_box_center(u.box)[1])
        front_unit_coords = _get_box_center(front_unit.box)
        if front_unit_coords[1] > 480: # Y-координата моста
            if game_state.cards:
                print(f"[Brain] ПОДДЕРЖКА АТАКИ! Добавляем юнита к {front_unit.class_name}.")
                return {'slot_index': 1, 'coords': (int(front_unit_coords[0]), int(front_unit_coords[1] + 50))}

    # --- Приоритет №3: НАЧАЛО АТАКИ ---
    if game_state.elixir and game_state.elixir >= 8: # Уменьшим порог для большей активности
        if game_state.cards:
            # Ищем танка в руке
            tank_card_index = -1
            for i, card in enumerate(game_state.cards):
                if not card.is_next and card.class_name.replace('Deck', '') in TANK_CLASSES:
                    tank_card_index = i
                    break
            
            if tank_card_index != -1:
                print(f"[Brain] НАЧАЛО АТАКИ (Танк)! Эликсир {game_state.elixir}. Играем танка в безопасную зону.")
                return {'slot_index': tank_card_index, 'coords': SAFE_PLAY_COORDS}
            else:
                print(f"[Brain] НАЧАЛО АТАКИ (Экономика)! Эликсир {game_state.elixir}. Играем карту 2 у моста.")
                return {'slot_index': 2, 'coords': BRIDGE_ATTACK_COORDS}

    return None 