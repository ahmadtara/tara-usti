from dataclasses import dataclass, field
@dataclass
class GameState:
    score:int=0
    level:int=1
    chain:list=field(default_factory=list)
    game_over:bool=False
