from typing import Dict, Any

# Словарь характеристик карт
# Keys должны совпадать с именами классов в vision.py, которые отображаются В РУКЕ (...Deck)
# Также добавим маппинг для Next карт и юнитов на поле, на всякий случай.

CARD_CONFIG: Dict[str, Dict[str, Any]] = {
    # --- Карты в колоде (Deck) ---
    'ArrowDeck': {'cost': 3, 'type': 'spell', 'role': 'damage'},
    'BanditDeck': {'cost': 3, 'type': 'unit', 'role': 'dash'},
    'BattleRamDeck': {'cost': 4, 'type': 'unit', 'role': 'win_condition'},
    'ElectroSpiritDeck': {'cost': 1, 'type': 'unit', 'role': 'cycle_stun'},
    'MinionsDeck': {'cost': 3, 'type': 'unit', 'role': 'air'},
    'PekkaDeck': {'cost': 7, 'type': 'unit', 'role': 'tank'},
    'RageDeck': {'cost': 2, 'type': 'spell', 'role': 'buff'},
    'RoyaleGhostDeck': {'cost': 3, 'type': 'unit', 'role': 'aoe_stealth'},

    # --- Юниты на поле (My...) ---
    'MyBandit': {'cost': 3, 'type': 'unit', 'role': 'dash'},
    'MyBarbarian': {'cost': 5, 'type': 'unit', 'role': 'melee'}, 
    'MyBattleRam': {'cost': 4, 'type': 'unit', 'role': 'win_condition'},
    'MyElectroSpirit': {'cost': 1, 'type': 'unit', 'role': 'cycle_stun'},
    'MyMinion': {'cost': 3, 'type': 'unit', 'role': 'air'},
    'MyPekka': {'cost': 7, 'type': 'unit', 'role': 'tank'},
    'MyRoyaleGhost': {'cost': 3, 'type': 'unit', 'role': 'aoe_stealth'},

    # --- Спеллы (на поле могут не иметь класса, но для справки) ---
    'Arrows': {'cost': 3, 'type': 'spell', 'role': 'damage'},
    'Rage': {'cost': 2, 'type': 'spell', 'role': 'buff'},
    'FireBall': {'cost': 4, 'type': 'spell', 'role': 'damage'},

    # --- Special ---
    'Empty': {'cost': 0, 'type': 'empty', 'role': 'none'}
}

def get_card_cost(card_name: str) -> int:
    # 1. Прямое совпадение
    if card_name in CARD_CONFIG:
        return CARD_CONFIG[card_name]['cost']
    
    # 2. Если это Next карта (PekkaNext -> PekkaDeck)
    if 'Next' in card_name:
        deck_name = card_name.replace('Next', 'Deck')
        if deck_name in CARD_CONFIG:
            return CARD_CONFIG[deck_name]['cost']
            
    # 3. Если это просто имя (Pekka -> PekkaDeck)
    deck_name = f"{card_name}Deck"
    if deck_name in CARD_CONFIG:
         return CARD_CONFIG[deck_name]['cost']
    
    # 4. Обратная совместимость с My...
    if card_name.startswith('My'):
        short_name = card_name[2:] # Remove My
        deck_name = f"{short_name}Deck"
        if deck_name in CARD_CONFIG:
            return CARD_CONFIG[deck_name]['cost']

    # Если не нашли
    # print(f"[CardConfig] Warning: Unknown card cost for {card_name}, assuming 3.")
    return 3

def get_card_type(card_name: str) -> str:
    """Returns 'spell', 'unit', 'building', or 'unknown'"""
    # Try direct match
    if card_name in CARD_CONFIG:
        return CARD_CONFIG[card_name]['type']
    
    # Try Next variant
    if 'Next' in card_name:
        deck_name = card_name.replace('Next', 'Deck')
        if deck_name in CARD_CONFIG:
            return CARD_CONFIG[deck_name]['type']
    
    # Try Deck suffix
    deck_name = f"{card_name}Deck"
    if deck_name in CARD_CONFIG:
        return CARD_CONFIG[deck_name]['type']
    
    # Try My prefix removal
    if card_name.startswith('My'):
        short_name = card_name[2:]
        deck_name = f"{short_name}Deck"
        if deck_name in CARD_CONFIG:
            return CARD_CONFIG[deck_name]['type']
    
    return 'unknown'
