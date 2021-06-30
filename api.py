import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from params import Score
import pickle

df = pd.read_csv('odi.csv')
df=df.drop(['date'],axis=1)

from typing import Optional

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [

    "http://localhost",
    "http://localhost:3000"

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[""]
)

app = FastAPI()

open_file = open('model.pkl', "rb")
loaded_list = pickle.load(open_file)
# print(loaded_list)

model=loaded_list[0]
scaler=loaded_list[1]

open_file.close()

#ML Stuff

le_bat=LabelEncoder()
le_bowl=LabelEncoder()
le_ven=LabelEncoder()
le_bats=LabelEncoder()
le_bowler=LabelEncoder()
df['bat_team']=le_bat.fit_transform(df['bat_team'])
df['venue']=le_ven.fit_transform(df['venue'])
df['bowl_team']=le_bowl.fit_transform(df['bowl_team'])
df['batsman']=le_bats.fit_transform(df['batsman'])
df['bowler']=le_bowler.fit_transform(df['bowler'])



@app.get("/")
def read_root():
    return {"Message": "Welcome to eSport"}


@app.post('/predict')
def predict_score(data:Score):
    data = data.dict()
    venue=data['venue']
    bat_team=data['bat_team']
    bowl_team=data['bowl_team']
    batsman=data['batsman']
    bowler=data['bowler']
    runs=data['runs']
    wickets=data['wickets']
    overs=data['overs']
    striker=data['striker']
    non_striker=data['non_striker']
    v=[venue,bat_team,bowl_team,batsman,bowler,runs,wickets,overs,striker,non_striker]
#     print(v)
    encoders=[le_ven,le_bat,le_bowl,le_bats,le_bowler]
    for i in range(5):
        v[i]=encoders[i].transform([v[i]])
    
    new_prediction = (model.predict(scaler.transform(np.array([v],dtype="object"))))
    score="{:.2f}".format(new_prediction[0])

    return {
        'Total_Score': score
    }

#Example Input:
# Sydney Cricket Ground	Sri Lanka	Australia	J Mubarak	NW Bracken	6	0	1.1	6	0	1	0	


#    Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
