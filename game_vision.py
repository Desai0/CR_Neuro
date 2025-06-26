# This file will contain all the computer vision logic:
# - YOLO model loading and prediction
# - Elixir detection (OCR)
# - Tower health detection (OCR)
# - The main perception function to create the GameState 

import cv2
import numpy as np
import pytesseract
import math
from ultralytics import YOLO
from game_state import GameState, Tower, Unit, Card
import time

# --- НАСТРОЙКИ И КОНСТАНТЫ ---

# OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
ELIXIR_ROI = (213, 1293, 27, 34)

# YOLO
MODEL_PATH = "runs/detect/train6/weights/best.pt"
model = YOLO(MODEL_PATH)

# Классы объектов
ALLY_TOWER_CLASSES = ['MyPrincessTower', 'MyKingTower']
ENEMY_TOWER_CLASSES = ['PrincessTower', 'KingTower', 'EnemyTower']
TOWER_CLASSES = ALLY_TOWER_CLASSES + ENEMY_TOWER_CLASSES

HP_CLASSES = ['TowerPrincessHP', 'MyPrincessTowerHP', 'MyKingHP', 'KingTowerHP']
ALLY_HP_CLASSES = ['MyPrincessTowerHP', 'MyKingHP']
ENEMY_HP_CLASSES = ['TowerPrincessHP', 'KingTowerHP']

CARD_CLASSES = [cls for cls in model.names.values() if 'Deck' in cls or 'Next' in cls]

# Явный список своих юнитов, чтобы избежать ошибочной реакции
MY_UNIT_CLASSES = [
    'MyBandit', 'MyBarbarian', 'MyBattleRam', 'MyElectroSpirit', 
    'MyMinion', 'MyPekka', 'MyRoyaleGhost', 'MyPrincessTowerBrocken', 'EnemyTowerBrocken'
]

# Классы, которые являются заклинаниями или состояниями, а не юнитами
SPELL_CLASSES = ['Rage', 'Empty', 'Arrows', 'FireBall']

UNIT_CLASSES = [cls for cls in model.names.values() if cls not in TOWER_CLASSES + HP_CLASSES + CARD_CLASSES + SPELL_CLASSES + ['Elixir']]

HP_TO_TOWER_MAP = {
    'MyPrincessTowerHP': 'MyPrincessTower',
    'MyKingHP': 'MyKingTower',
    'KingTowerHP': 'KingTower',
    'TowerPrincessHP': 'PrincessTower'
}

# --- ФУНКЦИИ-ПОМОЩНИКИ (OCR) ---

def _preprocess_for_ocr(image, is_ally=False, is_elixir=False):
    """Универсальная предобработка изображений для OCR."""
    if image.size == 0: return None
    
    if is_elixir:
        height, width = image.shape[:2]
        image = cv2.resize(image, (width*3, height*3), interpolation=cv2.INTER_CUBIC)
    else:
        height, width = image.shape[:2]
        scale_factor = max(2, 30 / height)
        image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if is_elixir:
        threshold_value = 200
    elif is_ally:
        threshold_value = 170 
    else:
        threshold_value = 200
        
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh

def _get_elixir_from_frame(frame):
    """Извлекает значение эликсира из полного кадра игры."""
    x, y, w, h = ELIXIR_ROI
    elixir_image = frame[y:y+h, x:x+w]
    processed_image = _preprocess_for_ocr(elixir_image, is_elixir=True)
    if processed_image is None: return None
    
    custom_config = r'--psm 8 -c tessedit_char_whitelist=0123456789'
    try:
        text = pytesseract.image_to_string(processed_image, config=custom_config)
        value = int(text.strip())
        return value if 0 <= value <= 10 else None
    except (ValueError, TypeError):
        return None

def _get_tower_health(yolo_results, frame):
    """Определяет здоровье башен."""
    # (Эта функция будет идентична той, что мы написали, но я ее сюда вставлю)
    # ... код определения здоровья ...
    # Вместо этого я скопирую код из game_state.py в эту область
    towers = {cls: [] for cls in TOWER_CLASSES}
    hp_regions = {cls: [] for cls in HP_CLASSES}

    if yolo_results[0].boxes is not None:
        for box in yolo_results[0].boxes:
            class_id = int(box.cls[0])
            class_name = yolo_results[0].names[class_id]
            coords = [int(i) for i in box.xyxy[0]]

            if class_name in TOWER_CLASSES:
                towers[class_name].append(coords)
            elif class_name in HP_CLASSES:
                hp_regions[class_name].append(coords)

    towers_with_health = []
    for hp_class, tower_class in HP_TO_TOWER_MAP.items():
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
                
                # Определяем параметры обрезки на основе класса
                if hp_class in ['MyKingHP', 'KingTowerHP']:
                    crop_right = 32
                elif hp_class == 'MyPrincessTowerHP':
                    crop_right = 28
                else: # TowerPrincessHP
                    crop_right = 20

                new_x2_hp = x2_hp - crop_right

                # Определяем параметры обрезки на основе класса
                if hp_class in ['MyKingHP', 'KingTowerHP']:
                    crop_left = 35
                elif hp_class == 'MyPrincessTowerHP':
                    crop_left = 30
                else: # TowerPrincessHP
                    crop_left = 20

                new_x1 = x1_hp + crop_left

                # Вертикальная обрезка
                if hp_class in ALLY_HP_CLASSES:
                    crop_top = 12
                    new_y1 = y1_hp + crop_top
                    if new_x1 < new_x2_hp and new_y1 < y2_hp:
                        hp_crop = frame[new_y1:y2_hp, new_x1:new_x2_hp]
                else:
                    crop_bottom = 10
                    new_y2 = y2_hp - crop_bottom
                    if new_x1 < new_x2_hp and y1_hp < new_y2:
                        hp_crop = frame[y1_hp:new_y2, new_x1:new_x2_hp]

                health_value = None
                if hp_crop is not None:
                    is_ally_hp = hp_class in ALLY_HP_CLASSES
                    custom_config = r'--psm 7 -c tessedit_char_whitelist=0123456789'
                    try:
                        processed_image = _preprocess_for_ocr(hp_crop, is_ally=is_ally_hp)
                        text = pytesseract.image_to_string(processed_image, config=custom_config)
                        health_value = int(text.strip())
                    except (ValueError, TypeError):
                        health_value = None
                
                # Добавляем башню, только если удалось распознать здоровье
                if health_value is not None:
                    towers_with_health.append(Tower(
                        class_name=tower_class, 
                        box=closest_tower_box, 
                        health=health_value
                    ))

    return towers_with_health

# --- ГЛАВНАЯ ФУНКЦИЯ МОДУЛЯ ---

def perceive_game_state(frame, last_game_state=None, analyze_slow_components=True):
    """
    Основная функция, которая принимает кадр и возвращает GameState.
    - Всегда выполняет быстрый анализ YOLO.
    - Выполняет медленный анализ OCR только при необходимости.
    - Использует данные из `last_game_state` если медленный анализ пропускается.
    """
    results = model.predict(source=frame, conf=0.3, verbose=False, tracker='bytetrack.yaml')
    
    current_game_state = GameState()
    # --- НАСЛЕДОВАНИЕ "ЛИПКИХ" СОСТОЯНИЙ ---
    if last_game_state:
        current_game_state.game_start = last_game_state.game_start
        current_game_state.match_over = last_game_state.match_over

    if analyze_slow_components:
        # Полный анализ: получаем новые данные OCR
        current_game_state.elixir = _get_elixir_from_frame(frame)
        all_towers = _get_tower_health(results, frame)
        # Снова разделяем башни по принадлежности
        current_game_state.my_towers = [t for t in all_towers if t.class_name in ALLY_TOWER_CLASSES]
        current_game_state.enemy_towers = [t for t in all_towers if t.class_name in ENEMY_TOWER_CLASSES]

    elif last_game_state is not None:
        # Быстрый анализ: используем старые данные для медленных компонентов
        current_game_state.elixir = last_game_state.elixir
        current_game_state.my_towers = last_game_state.my_towers
        current_game_state.enemy_towers = last_game_state.enemy_towers

    # Всегда анализируем быстрые компоненты (юниты, карты, СТАТУСЫ)
    if results[0].boxes is not None:
        # Сначала проверим статусы, они могут влиять на другие решения
        for box in results[0].boxes:
            class_name = model.names[int(box.cls[0])]
            if class_name == 'GameStart':
                if not current_game_state.game_start:
                    print("[Vision] Detected GameStart! Bot is now potentially active.")
                    # Устанавливаем начальное значение эликсира, чтобы бот не ждал OCR
                    current_game_state.elixir = 7
                current_game_state.game_start = True
                current_game_state.match_over = False # Сброс на случай новой игры
            elif class_name == 'MatchOver':
                if not current_game_state.match_over:
                    print("[Vision] Detected MatchOver! Bot will be deactivated.")
                current_game_state.match_over = True
        
        # Теперь обрабатываем остальные объекты
        for box in results[0].boxes:
            class_name = model.names[int(box.cls[0])]
            coords = tuple(int(i) for i in box.xyxy[0])
            
            if class_name in UNIT_CLASSES:
                # Определяем принадлежность по явному списку, а не по префиксу
                if class_name in MY_UNIT_CLASSES:
                    current_game_state.my_units.append(Unit(class_name=class_name, box=coords))
                else:
                    current_game_state.enemy_units.append(Unit(class_name=class_name, box=coords))

            elif class_name in CARD_CLASSES:
                is_next = 'Next' in class_name
                current_game_state.cards.append(Card(class_name=class_name, box=coords, is_next=is_next))
            
            elif class_name in ['MatchOver', 'GameStart']:
                pass # Уже обработано

    return current_game_state, results 