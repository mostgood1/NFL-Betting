from typing import Optional
from pydantic import BaseModel, Field

class GameRow(BaseModel):
    season: int
    week: int
    game_id: str
    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    # Optional splits
    home_q1: Optional[int] = None
    home_q2: Optional[int] = None
    home_q3: Optional[int] = None
    home_q4: Optional[int] = None
    away_q1: Optional[int] = None
    away_q2: Optional[int] = None
    away_q3: Optional[int] = None
    away_q4: Optional[int] = None

class TeamStatRow(BaseModel):
    season: int
    week: int
    team: str
    off_epa: Optional[float] = Field(default=None, description="Offensive EPA/play")
    def_epa: Optional[float] = Field(default=None, description="Defensive EPA/play")
    pace_secs_play: Optional[float] = None
    pass_rate: Optional[float] = None
    rush_rate: Optional[float] = None
    qb_adj: Optional[float] = Field(default=None, description="QB availability/grade adjustment")
    sos: Optional[float] = Field(default=None, description="Strength of schedule index")

class LineRow(BaseModel):
    season: int
    week: int
    game_id: str
    home_team: str
    away_team: str
    spread_home: Optional[float] = Field(default=None, description="Home spread (negative means favored)")
    total: Optional[float] = None
    moneyline_home: Optional[int] = None
    moneyline_away: Optional[int] = None
    close_spread_home: Optional[float] = None
    close_total: Optional[float] = None

