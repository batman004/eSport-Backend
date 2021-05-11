import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from params import Score
import pickle
df = pd.read_csv('odi.csv')
df=df.drop(['date'],axis=1)


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

z = df.iloc[:,[1,2,3,4,5,6,7,8,11,12]].values
w = df.iloc[:, 13].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(z, w, test_size = 0.3, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Linear Regression 
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(x_train,y_train)
objects=[lin,sc]
pickle.dump(objects,open('model.pkl', 'wb'))

