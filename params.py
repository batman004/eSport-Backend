from pydantic import BaseModel
# Class which describes Total Score Paramaters
class Score(BaseModel):
    venue: str 
    bat_team: str 
    bowl_team: str 
    batsman: str
    bowler: str
    runs:int 
    wickets: int 
    overs: float 
    striker: int	
    non_striker: int