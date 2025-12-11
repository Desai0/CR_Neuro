
# Mapping from Unit Class Name to Channel Index
# Channels:
# 0: My Units (All classes starting with 'My')
# 1: Enemy Swarm/Cycle (Skeletons, Goblins, Spirits, Archers...)
# 2: Enemy Air (Minions, Dragons, Balloon, SkeletonBarrel...)
# 3: Enemy Heavy Tank (Giant, Golem, Pekka, Mk, GoblinGiant...)
# 4: Enemy Ranged/Support (Musketeer, Wizard, Witch...)
# 5: Enemy WinCond/Siege (Hog, Ram, GoblinMachine, Demolisher...)
# 6: Enemy Buildings (Tesla, Cannon, Inferno...)
# 7: Enemy Tower Spawners (Barrel, Graveyard, Miner, Drill...)
# 8: Enemy Spells (Log, Zap, Fireball...)
# 9: Enemy Fighter/Mini-Tank (Knight, Valkyrie, MiniPekka, Prince, Bandit...)
# 10: Enemy Unknown/Other

CHANNEL_COUNT = 11

# Explicit mapping for all known YOLO classes
UNIT_TO_CHANNEL = {
    # --- My Units (Mapped to 0 via logic, but explicit here for safety) ---
    'MyBandit': 0, 'MyBarbarian': 0, 'MyBattleRam': 0, 'MyElectroSpirit': 0,
    'MyMinion': 0, 'MyPekka': 0, 'MyRoyaleGhost': 0,

    # --- Channel 1: Enemy Swarm / Low HP / Cycle (Ground) ---
    'Archer': 1, 'Archers': 1, 'SpearGoblin': 1, 'Goblin': 1, 'Goblins': 1,
    'Skeleton': 1, 'Skeletons': 1, 'Bomber': 1, 'EvoBomber': 1,
    'EvoSkeleton': 1, 'EvoWallBreackers': 1, 'WallBreakers': 1,
    'Princess': 1, 'DartGoblin': 1, 'FireCracker': 1, 'Guards': 1,
    'RascalW': 1, 'IceSpirit': 1, 'ElectroSpirit': 1, 'FireSpirit': 1, 'HealSpirit': 1, 'Minion': 1, 'Minions': 1, 'Bat': 1, 'Bats': 1,

    # --- Channel 2: Enemy Air (All Air Units) ---
    'BabyDragon': 2, 'ElectroDragon': 2, 'SkeletonDragon': 2, 'InfernoDragon': 2,
    'MegaMinion': 2, 'Balloon': 2, 'LavaHound': 2, 'LavaHoundMini': 2,
    'Phoenix': 2, 'PhoenixEgg': 2, 'FlyingMachine': 2,
    'SkeletonBarrel': 2, # Flies then drops skeletons

    # --- Channel 3: Enemy Heavy Tank (High HP, slow, targets buildings) ---
    'Giant': 3, 'GoblinGiant': 3, 'ElectroGiant': 3, 'RoyalGiant': 3, 'EvoRoyaleGiant': 3,
    'Golem': 3, 'GolemMini': 3, 'ElixirGolem': 3, 
    'Pekka': 3, 'MegaKnight': 3, 'EvoMegaKnight': 3,
    'GiantSkeleton': 3, 'SkeletonKing': 3, 'Monk': 3,
    'Goblinstein': 3, # Tanky unit

    # --- Channel 4: Enemy Ranged / Support / Splash ---
    'Musketeer': 4, 'ThreeMusketeers': 4, 'Wizard': 4, 'EvoWizard': 4,
    'IceWizard': 4, 'ElectroWizard': 4,
    'Witch': 4, 'NightWitch': 4, 'MotherWitch': 4,
    'MagicArcher': 4, 'Sparky': 4, 'CannonCart': 4, 'Zappies': 4,
    'Healer': 4, 'Executioner': 4, 'Bowler': 4, 'Hunter': 4,
    'ArcherQueen': 4, 'LittlePrince': 4, 

    # --- Channel 5: Enemy WinCond / Siege (Building Targeters or Siege Units) ---
    'Hog': 5, 'RamRider': 5, 'BattleRam': 5, 'Pig': 5, 'RoyalePig': 5,
    'GoblinMachine': 5, 'Demolisher': 5, # Siege units

    # --- Channel 6: Enemy Buildings ---
    'Cannon': 6, 'Tesla': 6, 'TeslaHidden': 6, 'Inferno': 6, 'BombTower': 6,
    'Mortar': 6, 'EvoMortar': 6, 'XBow': 6,
    'GoblinHut': 6, 'BarbarianHut': 6, 'Furnace': 6, 'Tombstone': 6,
    'GoblinCage': 6, 'EvoGoblinCage': 6, 'ElixirCollector': 6,

    # --- Channel 7: Enemy Tower Spawners (Troops that appear anywhere/on tower) ---
    'GoblinBarrel': 7, 'Graveyard': 7, 'Miner': 7, 'MightyMiner': 7, 'GoblinDrill': 7,

    # --- Channel 8: Enemy Spells (Projectiles/Effects) ---
    'Rocket': 8, 'FireBall': 8, 'Lightning': 8, 'Zap': 8, 'Log': 8, 'Arrows': 8,
    'Freeze': 8, 'Poison': 8, 'Earthquake': 8, 'Tornado': 8, 'Clone': 8,
    'Mirror': 8, 'Void': 8, 'GoblinCurse': 8, 'GiantSnowball': 8,
    'BarbarianBarrel': 8, 

    # --- Channel 9: Enemy Fighter / Mini-Tank (Melee, Medium HP, mobile) ---
    'Knight': 9, 'EvoKnight': 9, 'Valkyrie': 9, 'EvoValkyrie': 9,
    'MiniPekka': 9, 'Prince': 9, 'DarkPrince': 9,
    'Bandit': 9, 'RoyaleGhost': 9, 'FisherMan': 9, 'Lumberjack': 9,
    'RascalM': 9, 'GoldenKnight': 9, 'LittlePrinceGuard': 9,
    'Barbarian': 9, 'EvoBarbarian': 9, 'EliteBarbarian': 9,
    'ElixirGolemMini': 9, 'ElixirGolemMicro': 9,
    'GoblinsteinScientist': 9,
}

def get_unit_channel(class_name: str) -> int:
    # 1. My Units Check
    if class_name.startswith('My'):
        return 0
        
    # 2. Exact Match
    if class_name in UNIT_TO_CHANNEL:
        return UNIT_TO_CHANNEL[class_name]
        
    # 3. Substring Fallbacks (for variations like Cloned)
    if 'Cloned' in class_name:
        base_name = class_name.replace('Cloned', '')
        if base_name in UNIT_TO_CHANNEL:
            return UNIT_TO_CHANNEL[base_name]

    # 4. Keyword Fallbacks
    name_lower = class_name.lower()
    if 'minion' in name_lower or 'dragon' in name_lower or 'bat' in name_lower or 'balloon' in name_lower:
        return 2 # Air
    if 'golem' in name_lower or 'giant' in name_lower or 'pekka' in name_lower:
        if 'mini' in name_lower: return 9 # MiniPekka
        return 3 # Tank
    if 'knight' in name_lower or 'valk' in name_lower or 'prince' in name_lower:
        return 9 # Fighter
    if 'goblin' in name_lower or 'skeleton' in name_lower:
        # Barrel/Drill handled by exact match above, so these are swarms
        return 1 # Swarm
    if 'tower' in name_lower or 'hut' in name_lower:
        return 6 # Building
    if 'spell' in name_lower or 'barrel' in name_lower:
        return 8 # Spell guess
        
    # Default Unknown -> Channel 10
    return 10
