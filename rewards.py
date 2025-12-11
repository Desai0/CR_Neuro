from game_state import GameState
from typing import List, Tuple, Dict
import math
import time

class RewardCalculator:
    def __init__(self):
        # --- Настройки наград ---
        self.REWARD_WIN = 200.0
        self.REWARD_LOSS = -200.0
        self.REWARD_TOWER_HP_GAIN = 0.01  # За единицу HP (урон врагу)
        self.REWARD_TOWER_HP_LOSS = -0.01 # За единицу HP (урон нам)
        self.REWARD_ELIXIR_SPENT = 0.0    # Отключено
        self.REWARD_CARD_PLAYED = 0.5     # Награда за сыгранную карту
        self.REWARD_INVALID_ACTION = -1.0 # Наказание за ошибку
        self.REWARD_SURVIVAL_PER_SEC = 0.1 # Награда за выживание (раз в секунду)
        
        # Награды за уничтожение башен
        self.REWARD_TOWER_DESTROYED_ENEMY = 100.0 # Мы снесли башню
        self.REWARD_TOWER_DESTROYED_MY = -100.0   # Мы потеряли башню

        self.survival_accumulator = 0.0
        self.match_over_confirmation_timer = 0.0 # Таймер для подтверждения конца матча
        self.reset()

    def reset(self):
        """Сбрасывает состояние калькулятора наград для нового эпизода."""
        self.event_log: List[Tuple[str, float]] = []
        self.last_time = time.time()
        self.episode_start_time = time.time()
        self.terminal_reward_given = False # Флаг, что награда за конец матча уже выдана
        self.survival_accumulator = 0.0
        self.match_over_confirmation_timer = 0.0
        
        # Единый кэш уничтоженных башен (координаты центров), за которые УЖЕ выплачена награда
        self.destroyed_towers_cache: List[Tuple[int, int]] = []
        
        # Кандидаты на уничтожение: {(x, y): timestamp_first_seen}
        # Храним объекты, которые мы видим прямо сейчас, но они еще не продержались 5 секунд
        self.broken_tower_candidates: Dict[Tuple[int, int], float] = {}

    def calculate(self, prev_state: GameState, current_state: GameState, action_performed: bool = False) -> float:
        """
        Вычисляет награду на основе изменения состояния игры.
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # 1. Задержка старта (Warm-up)
        # Если с начала эпизода прошло меньше 3 секунд, награды не считаем (даем системе прогрузиться)
        if current_time - self.episode_start_time < 3.0:
            return 0.0
        
        reward = 0.0
        
        if prev_state is None:
            return 0.0

        # 2. Победа / Поражение (ТЕРМИНАЛЬНЫЕ СОСТОЯНИЯ)
        # Проверяем MatchOver с фильтрацией: класс должен быть виден стабильно
        if current_state.match_over:
             self.match_over_confirmation_timer += dt
        else:
             # self.match_over_confirmation_timer = 0.0 # НЕ СБРАСЫВАЕМ ТАЙМЕР!
             # Это позволяет накапливать время даже при мерцании (например, 1 кадр пропуск, потом снова видно)
             # Но если он пропадет надолго (в reset), он сбросится.
             pass
             
        # Считаем матч оконченным только если MatchOver висит более 1.0 секунды (защита от случайного мерцания)
        # Или если это реальный конец (но мы не знаем без таймера). 
        # Пусть будет 1 секунда - это надежно.
        # Уменьшаем время до 0.5, раз VisionSystem уже фильтрует
        is_confirmed_match_over = self.match_over_confirmation_timer >= 0.5

        if is_confirmed_match_over:
            if not self.terminal_reward_given:
                self.terminal_reward_given = True
                if len(current_state.my_towers) > len(current_state.enemy_towers):
                    self.log_event("WIN", self.REWARD_WIN)
                    return self.REWARD_WIN
                elif len(current_state.my_towers) < len(current_state.enemy_towers):
                    self.log_event("LOSS", self.REWARD_LOSS)
                    return self.REWARD_LOSS
                else:
                    # Ничья или ошибка детекции в конце
                    return 0.0
            else:
                # Награда уже выдана, возвращаем 0, чтобы не дублировать
                return 0.0

        # 3. Уничтожение башен (С задержкой 5 секунд и проверкой стабильности)
        
        # Собираем все текущие сломанные башни
        all_broken_current = []
        for t in current_state.my_broken_towers:
            all_broken_current.append(t)
        for t in current_state.enemy_broken_towers:
            all_broken_current.append(t)

        # Множество кандидатов, которые подтверждены в ЭТОМ кадре
        active_candidates_this_frame = set()

        for tower_obj in all_broken_current:
            center = self._get_center(tower_obj.box)
            
            # A. Если эта башня УЖЕ в кэше выплаченных, пропускаем
            if self._is_already_logged(center, self.destroyed_towers_cache, threshold=150):
                continue

            # B. Ищем, есть ли эта башня в кандидатах (с порогом дистанции)
            match_candidate_key = None
            for cand_center in self.broken_tower_candidates:
                dist = math.sqrt((center[0] - cand_center[0])**2 + (center[1] - cand_center[1])**2)
                if dist < 150:
                    match_candidate_key = cand_center
                    break
            
            if match_candidate_key:
                # Башня уже отслеживается. Проверяем таймер.
                first_seen = self.broken_tower_candidates[match_candidate_key]
                duration = current_time - first_seen
                
                # Отмечаем, что кандидат активен в этом кадре (чтобы не удалить)
                active_candidates_this_frame.add(match_candidate_key)
                
                if duration >= 5.0:
                    # ВЫПЛАТА НАГРАДЫ (Таймер истек, событие подтверждено)
                    y_coord = match_candidate_key[1] # Используем стабильную координату кандидата
                    is_enemy_zone = y_coord < 600
                    is_my_zone = y_coord > 700
                    
                    final_reward = 0.0
                    event_name = "UNKNOWN TOWER"
                    
                    if is_enemy_zone:
                        final_reward = self.REWARD_TOWER_DESTROYED_ENEMY
                        event_name = "ENEMY TOWER DESTROYED"
                    elif is_my_zone:
                        final_reward = self.REWARD_TOWER_DESTROYED_MY
                        event_name = "MY TOWER LOST"
                    else:
                        if "My" in tower_obj.class_name:
                            final_reward = self.REWARD_TOWER_DESTROYED_MY
                            event_name = "MY TOWER LOST"
                        else:
                            final_reward = self.REWARD_TOWER_DESTROYED_ENEMY
                            event_name = "ENEMY TOWER DESTROYED"
                    
                    # Добавляем в кэш выплаченных
                    self.destroyed_towers_cache.append(center)
                    reward += final_reward
                    self.log_event(f"{event_name} ({tower_obj.class_name}) at {center}", final_reward)
                    
                    # Удаляем из кандидатов (больше не нужно отслеживать, перенесено в destroyed_towers_cache)
                    if match_candidate_key in self.broken_tower_candidates:
                        del self.broken_tower_candidates[match_candidate_key]
                        active_candidates_this_frame.remove(match_candidate_key)

            else:
                # Новая башня (кандидат). Запускаем таймер (timestamp текущего времени).
                self.broken_tower_candidates[center] = current_time
                active_candidates_this_frame.add(center)

        # C. Очистка кандидатов
        # Оставляем только тех кандидатов, которые были видны в ЭТОМ кадре.
        # Если башня исчезла хотя бы на кадр, она удаляется из списка кандидатов.
        # При следующем появлении она будет добавлена заново с новым временем старта (таймер с нуля).
        self.broken_tower_candidates = {
            k: v for k, v in self.broken_tower_candidates.items() 
            if k in active_candidates_this_frame
        }

        # 4. Изменение здоровья башен (Постоянное)
        hp_diff_my = self._calculate_hp_diff(prev_state.my_towers, current_state.my_towers)
        hp_diff_enemy = self._calculate_hp_diff(prev_state.enemy_towers, current_state.enemy_towers)
        
        # Начисляем награды за урон (hp_diff возвращает отрицательное число при уроне)
        if hp_diff_my < 0:
            r = abs(hp_diff_my) * self.REWARD_TOWER_HP_LOSS
            reward += r
        
        if hp_diff_enemy < 0:
            r = abs(hp_diff_enemy) * self.REWARD_TOWER_HP_GAIN
            reward += r

        # 5. Награда за сыгранную карту
        prev_empty_count = sum(1 for c in prev_state.cards if c.class_name == 'Empty')
        curr_empty_count = sum(1 for c in current_state.cards if c.class_name == 'Empty')
        
        if curr_empty_count > prev_empty_count:
            diff = curr_empty_count - prev_empty_count
            reward += self.REWARD_CARD_PLAYED * diff
        
        # 6. Выживание (пока матч идет)
        if not self.terminal_reward_given:
            self.survival_accumulator += dt
            if self.survival_accumulator >= 1.0:
                reward += self.REWARD_SURVIVAL_PER_SEC
                self.survival_accumulator -= 1.0 # Сбрасываем (вычитаем 1.0, чтобы сохранить остаток)

        return reward

    def _calculate_hp_diff(self, prev_towers, current_towers):
        """
        Считает изменение HP, сопоставляя башни по координатам.
        Возвращает отрицательное число, если был нанесен урон.
        Игнорирует восстановление HP (глюки зрения) и исчезновение башен.
        """
        total_damage = 0.0
        used_prev_indices = set()
        
        for curr in current_towers:
            curr_center = self._get_center(curr.box)
            best_match = None
            best_dist = 9999
            best_idx = -1
            
            # Ищем соответствующую башню в предыдущем кадре
            for i, prev in enumerate(prev_towers):
                if i in used_prev_indices: continue
                prev_center = self._get_center(prev.box)
                dist = math.sqrt((curr_center[0]-prev_center[0])**2 + (curr_center[1]-prev_center[1])**2)
                
                # Если дистанция небольшая, считаем, что это та же башня
                if dist < 100: 
                    if dist < best_dist:
                        best_dist = dist
                        best_match = prev
                        best_idx = i
            
            if best_match:
                used_prev_indices.add(best_idx)
                if curr.health is not None and best_match.health is not None:
                    diff = curr.health - best_match.health
                    
                    # Учитываем только урон (отрицательный diff)
                    # Фильтруем слишком большие скачки (> 2000), считая их ошибкой распознавания
                    # Но позволяем проходить урону > 1000, если это не явный глюк (до 2000)
                    if -2000 < diff < 0:
                         total_damage += diff
                     
        return total_damage

    def _get_center(self, box):
        return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

    def _is_already_logged(self, center, log_list, threshold=150):
        """Проверяет, есть ли уже башня с похожими координатами в логе."""
        for logged_center in log_list:
            dist = math.sqrt((center[0] - logged_center[0])**2 + (center[1] - logged_center[1])**2)
            if dist < threshold:
                return True
        return False

    def log_event(self, name: str, value: float):
        """Записывает важное событие."""
        self.event_log.append((name, value))

    def get_summary(self) -> str:
        """Возвращает отчет о ключевых событиях."""
        if not self.event_log:
            return "No major events."
        
        summary = "\n--- Match Events Summary ---\n"
        for name, value in self.event_log:
            summary += f"{name}: {value:+.1f}\n"
        return summary
