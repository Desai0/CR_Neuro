# yolo train model=yolo12l.pt data="C:/Users/e8351/Documents/CR_Neuro/CustomWorkflowObjectDetection.v3i.yolov12/data.yaml" imgsz=900 batch=15 epochs=90 lr0=0.01 augment=True copy_paste=0.1 device=00
from ultralytics import YOLO
import cv2
import mss
import numpy as np
import random # Для генерации случайных цветов
from game_vision import perceive_game_state, ALLY_TOWER_CLASSES, ENEMY_TOWER_CLASSES
from game_state import GameState, Tower, Unit, Card
import threading
import time
from game_logic import choose_action_rule_based
from game_automation import play_card, convert_vision_to_global_coords, clamp_coords_to_playable_area

# --- Настройки ---
MONITOR_DETAILS = {"top": 35, "left": 2674, "width": 766, "height": 1355}
WINDOW_TITLE = "CR_Neuro"
# Частота выполнения МЕДЛЕННОГО анализа (OCR). 1 раз в 5 циклов потока.
SLOW_ANALYSIS_TICK_RATE = 5 
# Пауза после выполнения действия, чтобы не кликать слишком часто
ACTION_COOLDOWN = 1.0 # 1 секунда

# --- Потокобезопасное хранилище состояния ---
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.game_state = GameState()
        self.yolo_results = None
        self.last_action_time = 0
        self.running = True

    def set_frame(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def set_analysis_results(self, game_state, yolo_results):
        with self.lock:
            self.game_state = game_state
            self.yolo_results = yolo_results

    def get_analysis_results(self):
        with self.lock:
            # Возвращаем копии, чтобы избежать гонки данных при отрисовке
            return self.game_state, self.yolo_results

    def can_perform_action(self):
        with self.lock:
            return time.time() - self.last_action_time > ACTION_COOLDOWN

    def action_performed(self):
        with self.lock:
            self.last_action_time = time.time()

    def is_running(self):
        with self.lock:
            return self.running

    def stop(self):
        with self.lock:
            self.running = False

# --- Функция для фонового потока ---
def vision_worker(state: SharedState):
    """Этот 'рабочий' постоянно анализирует последний доступный кадр."""
    print("Vision worker запущен.")
    worker_frame_counter = 0
    last_game_state = GameState()

    while state.is_running():
        frame = state.get_frame()
        if frame is not None:
            analyze_slow = (worker_frame_counter % SLOW_ANALYSIS_TICK_RATE == 0)
            
            game_state, yolo_results = perceive_game_state(
                frame, 
                last_game_state=last_game_state, 
                analyze_slow_components=analyze_slow
            )
            
            state.set_analysis_results(game_state, yolo_results)
            last_game_state = game_state # Сохраняем последнее состояние для следующего цикла
            worker_frame_counter += 1
        else:
            # Если кадров нет, делаем небольшую паузу, чтобы не грузить CPU впустую
            time.sleep(0.01)
    print("Vision worker остановлен.")

# --- Основной поток (UI и Действия) ---
def main():
    """Главный цикл приложения, отвечающий за UI и выполнение действий."""
    sct = mss.mss()
    shared_state = SharedState()
    
    # Запускаем фоновый поток
    worker_thread = threading.Thread(target=vision_worker, args=(shared_state,))
    worker_thread.start()
    
    print("UI-поток запущен. Нажмите 'q' в окне для выхода.")
    class_colors = {}

    try:
        while True:
            sct_img = sct.grab(MONITOR_DETAILS)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            shared_state.set_frame(frame)

            game_state, yolo_results = shared_state.get_analysis_results()
            
            # --- Принятие и выполнение решения ---
            if game_state.game_start and not game_state.match_over and shared_state.can_perform_action():
                action = choose_action_rule_based(game_state)
                if action:
                    global_coords = convert_vision_to_global_coords(action['coords'])
                    # Примагничиваем координаты к игровой зоне
                    clamped_coords = clamp_coords_to_playable_area(global_coords)
                    
                    if global_coords != clamped_coords:
                        print(f"[Main] Координаты скорректированы: {global_coords} -> {clamped_coords}")

                    play_card(action['slot_index'], clamped_coords)
                    shared_state.action_performed()
            
            # --- Отрисовка ---
            if yolo_results and yolo_results[0].boxes is not None:
                for box in yolo_results[0].boxes:
                    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                    class_id = int(box.cls[0])
                    class_name = yolo_results[0].names[class_id]
                    if class_name not in class_colors:
                        class_colors[class_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    color = class_colors[class_name]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {float(box.conf[0]):.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            towers_to_draw = game_state.my_towers + game_state.enemy_towers
            for tower in towers_to_draw:
                if tower.health is not None:
                    x1, y1, _, _ = tower.box
                    health_text = f"HP: {tower.health}"
                    cv2.putText(frame, health_text, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow(WINDOW_TITLE, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Чистая остановка
        print("Остановка потоков...")
        shared_state.stop()
        worker_thread.join()
        cv2.destroyAllWindows()
        print("Скрипт завершен.")

if __name__ == "__main__":
    main()