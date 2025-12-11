# This file will contain the "hands" of the agent.
# It will be responsible for executing actions in the game,
# such as clicking on cards and locations on the screen.
# This will likely use pyautogui or ADB. 

import pyautogui
import time
import random

# --- Координаты, полученные после калибровки ---

# Координаты центров для 4-х слотов карт в руке
CARD_SLOTS = [
    (2900, 1241), # Слот 1
    (3050, 1241), # Слот 2
    (3200, 1241), # Слот 3
    (3350, 1241)  # Слот 4
]

# Прямоугольная область игрового поля, куда можно ставить юнитов
# (левый_верхний_x, левый_верхний_y, правый_нижний_x, правый_нижний_y)
PLAYABLE_AREA = (2731, 636, 3385, 1058)

# Область захвата экрана, которая передается в модель зрения
# Это нужно для пересчета координат
VISION_CAPTURE_AREA = {"top": 35, "left": 2674, "width": 766, "height": 1355}

def convert_vision_to_global_coords(local_coords):
    """
    Преобразует локальные координаты от модуля зрения в глобальные экранные координаты.
    """
    local_x, local_y = local_coords
    global_x = local_x + VISION_CAPTURE_AREA["left"]
    global_y = local_y + VISION_CAPTURE_AREA["top"]
    return (global_x, global_y)

def clamp_coords_to_playable_area(coords):
    """
    "Примагничивает" координаты к ближайшей точке внутри PLAYABLE_AREA.
    """
    x, y = coords
    x_min, y_min, x_max, y_max = PLAYABLE_AREA
    
    clamped_x = max(x_min, min(x, x_max))
    clamped_y = max(y_min, min(y, y_max))
    
    return (clamped_x, clamped_y)

def is_within_playable_area(coords):
    """Проверяет, находятся ли координаты внутри игровой области."""
    x, y = coords
    x_min, y_min, x_max, y_max = PLAYABLE_AREA
    return x_min <= x <= x_max and y_min <= y <= y_max

def play_card(card_slot_index, placement_coords):
    """
    Разыгрывает карту из указанного слота в указанное место на поле.

    :param card_slot_index: Индекс слота карты (0, 1, 2 или 3).
    :param placement_coords: Кортеж (x, y) для размещения на поле.
    :return: True, если действие выполнено, False в случае ошибки.
    """
    # 1. Проверка входных данных
    if not 0 <= card_slot_index < len(CARD_SLOTS):
        print(f"[Automation ERROR] Неверный индекс слота карты: {card_slot_index}")
        return False

    if not is_within_playable_area(placement_coords):
        print(f"[Automation ERROR] Координаты {placement_coords} вне игровой зоны.")
        return False

    # 2. Получение координат карты
    card_coords = CARD_SLOTS[card_slot_index]

    # 3. Симуляция действий
    print(f"[Action] Играем карту из слота {card_slot_index+1} в точку {placement_coords}")
    try:
        # Устанавливаем небольшую паузу между действиями, чтобы игра успела среагировать
        pyautogui.PAUSE = 0.1
        
        # Кликаем на карту
        pyautogui.click(card_coords)
        
        # Кликаем на поле
        pyautogui.click(placement_coords)
        
        return True
    except Exception as e:
        print(f"[Automation ERROR] Произошла ошибка при симуляции клика: {e}")
        return False

# --- Пример для тестирования ---
if __name__ == '__main__':
    print("Тестирование модуля автоматизации.")
    print("Наведите курсор на окно игры.")
    print("Через 3 секунды будет разыграна ПЕРВАЯ карта (индекс 0) в случайное место на поле.")
    time.sleep(3)
    
    # Генерируем случайные координаты внутри игровой зоны
    x_min, y_min, x_max, y_max = PLAYABLE_AREA
    random_x = random.randint(x_min, x_max)
    random_y = random.randint(y_min, y_max)
    
    play_card(0, (random_x, random_y))
    
    print("\\nТест завершен. Проверьте, была ли разыграна карта в игре.") 