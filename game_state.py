from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

@dataclass
class Tower:
    class_name: str
    health: int
    box: Tuple[int, int, int, int]

@dataclass
class Unit:
    class_name: str
    box: Tuple[int, int, int, int]

@dataclass
class Card:
    class_name: str
    box: Tuple[int, int, int, int]
    is_next: bool = False

@dataclass
class GameState:
    elixir: Optional[int] = None
    my_towers: List[Tower] = field(default_factory=list)
    enemy_towers: List[Tower] = field(default_factory=list)
    my_units: List[Unit] = field(default_factory=list)
    enemy_units: List[Unit] = field(default_factory=list)
    cards: List[Card] = field(default_factory=list)
    match_over: bool = False
    game_start: bool = False

    def __str__(self):
        """Красивый вывод состояния игры в консоль."""
        
        def format_towers(towers):
            return [f"{t.class_name}(HP: {t.health})" for t in towers]
        
        def format_units(units):
            return [u.class_name for u in units]

        def format_cards(cards):
            hand = [c.class_name for c in cards if not c.is_next]
            next_card = [c.class_name for c in cards if c.is_next]
            return f"Hand: {hand}, Next: {next_card[0] if next_card else 'N/A'}"

        my_towers_str = ", ".join(format_towers(self.my_towers))
        enemy_towers_str = ", ".join(format_towers(self.enemy_towers))
        my_units_str = ", ".join(format_units(self.my_units))
        enemy_units_str = ", ".join(format_units(self.enemy_units))
        cards_str = format_cards(self.cards)

        return (
            f"--- Game State ---\n"
            f"Elixir: {self.elixir if self.elixir is not None else 'N/A'}\n"
            f"My Towers: [{my_towers_str}]\n"
            f"Enemy Towers: [{enemy_towers_str}]\n"
            f"My Units: [{my_units_str}]\n"
            f"Enemy Units: [{enemy_units_str}]\n"
            f"Cards: {cards_str}\n"
            f"------------------"
        )

    def get_my_towers(self) -> List[Tower]:
        return self.my_towers
    
    def get_enemy_towers(self) -> List[Tower]:
        return self.enemy_towers 