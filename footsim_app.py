from click import option
from sqlalchemy import column
import streamlit as st
import sklearn
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
#Processamento
from sklearn.preprocessing import StandardScaler 
#Modelo
from sklearn.ensemble import RandomForestClassifier

path = ''
df = pd.read_csv('df_cup.csv',index_col=0)

df_timesh = df.groupby('home_team_cod').agg({'home_team_fifa_rank':'mean','home_team_total_fifa_points':'mean','Overal_23_home':'mean',	
                                            'Potencial_23_home':'mean',	'Overal_22_home':'mean',
                                            'Potencial_22_home':'mean','Overal_21_home':'mean', 
                                            'Potencial_21_home':'mean'}).reset_index()  
 #                                           'Potencial_20_home':'mean',	'Overal_19_home':'mean',	
#                                            'Potencial_19_home':'mean','Overal_18_home':'mean',	
#                                            'Potencial_18_home':'mean','Overal_17_home':'mean','Potencial_17_home':'mean'}).reset_index()

df_timesa = df.groupby('away_team_cod').agg({'away_team_fifa_rank':'mean','away_team_total_fifa_points':'mean',	
                                            'Overal_23_away':'mean','Potencial_23_away':'mean',
                                            'Overal_22_away':'mean','Potencial_22_away':'mean', 
                                            'Overal_21_away':'mean','Potencial_21_away':'mean'}).reset_index()  	
 #                                           'Overal_20_away':'mean','Potencial_20_away':'mean',	
 #                                           'Overal_19_away':'mean','Potencial_19_away':'mean',	
 #                                           'Overal_18_away':'mean','Potencial_18_away':'mean',
#                                            'Overal_17_away':'mean','Potencial_17_away':'mean'}).reset_index()


#df = df.sample(frac=0.1) # Take some records just to build a toy model
x_cup = df.drop(columns = ['shoot_out','home_team','away_team','ndate', 'date','home_team_continent','away_team_continent',
                           'tournament','city','country','home_team_score','away_team_score','home_team_result',
                           'result','neutral_location', 'Overal_20_away','Overal_19_away','Overal_18_away','Overal_17_away',
                           'Potencial_20_away', 'Potencial_19_away', 'Potencial_18_away', 'Potencial_17_away',
                           'Overal_20_home','Overal_19_home','Overal_18_home','Overal_17_home',
                           'Potencial_20_home', 'Potencial_19_home', 'Potencial_18_home', 'Potencial_17_home'])
y_cup = df['result']

x_train, x_test, y_train, y_test = train_test_split(x_cup, y_cup, test_size = 0.2, random_state = 2)

x_cup.to_excel('x_cup.xlsx')
#título
st.title("Simule suas apostas com o FOOTSIM")

#cabeçalho
st.subheader("Faça seu jogo")

#apelido do jogo
bet_input = st.sidebar.text_input("De um apelido para seu jogo")

#escrevendo o nome do usuário
#st.write("Jogo:", bet_input)


#dados dos usuários com a função

def get_user_date():
    optionc = st.sidebar.selectbox(
    'Seleção da casa',
    ('Albania','Algeria','Angola','Argentina','Australia','Austria','Belgium','Bolivia','Brazil','Bulgaria','Burkina Faso','Cameroon','Canada','Chile','China PR','Colombia','Congo','Costa Rica','Croatia','Cyprus','Czech Republic','Denmark','Ecuador','Egypt','England','Finland','France','Georgia','Germany','Ghana','Greece','Guinea','Honduras','Hungary','Iceland','Iran','Italy','Jamaica','Japan','Korea Republic','Kosovo','Mali','Mexico','Montenegro','Morocco','Netherlands','New Zealand','Nigeria','North Macedonia','Northern Ireland','Norway','Panama','Paraguay','Peru','Poland','Portugal','Republic of Ireland','Romania','Russia','Saudi Arabia','Scotland','Senegal','Serbia','Slovakia','Slovenia','South Africa','Spain','Sweden','Switzerland','Tunisia','Turkey','Ukraine','United States','Uruguay','Venezuela','Wales',
    ))
    st.write('Casa:', optionc)
    optionv = st.sidebar.selectbox(
    'Seleção visitante',
    ('Albania','Algeria','Angola','Argentina','Australia','Austria','Belgium','Bolivia','Brazil','Bulgaria','Burkina Faso','Cameroon','Canada','Chile','China PR','Colombia','Congo','Costa Rica','Croatia','Cyprus','Czech Republic','Denmark','Ecuador','Egypt','England','Finland','France','Georgia','Germany','Ghana','Greece','Guinea','Honduras','Hungary','Iceland','Iran','Italy','Jamaica','Japan','Korea Republic','Kosovo','Mali','Mexico','Montenegro','Morocco','Netherlands','New Zealand','Nigeria','North Macedonia','Northern Ireland','Norway','Panama','Paraguay','Peru','Poland','Portugal','Republic of Ireland','Romania','Russia','Saudi Arabia','Scotland','Senegal','Serbia','Slovakia','Slovenia','South Africa','Spain','Sweden','Switzerland','Tunisia','Turkey','Ukraine','United States','Uruguay','Venezuela','Wales',
    ))
    
#    home_team_fifa_rank = st.sidebar.slider("Ranking Fifa Casa",0, 15, 1)
#    away_team_fifa_rank = st.sidebar.slider("Ranking Fifa Visitante",0, 15, 1)
#    home_team_total_fifa_points = st.sidebar.slider("Pontos Fifa Casa",0, 15, 1)
#    away_team_total_fifa_points = st.sidebar.slider("Pontos Fifa Visitante",0, 15, 1)
#    Overal_23_away = st.sidebar.slider("Overal Visitante 23",0, 15, 1)
#    Potencial_23_away = st.sidebar.slider("Potencial Visitante 23",0, 15, 1)
#    Overal_22_away = st.sidebar.slider("Overal Visitante 22",0, 15, 1)
#    Potencial_22_away = st.sidebar.slider("Potencial Visitante 22",0, 15, 1)
#    Overal_21_away = st.sidebar.slider("Overal Visitante 21",0, 15, 1)
#    Potencial_21_away = st.sidebar.slider("Potencial Visitante 21",0, 15, 1)
#    Overal_23_home = st.sidebar.slider("Overal Casa 23",0, 15, 1)
#    Potencial_23_home = st.sidebar.slider("Potencial Casa 23",0, 15, 1)
#    Overal_22_home = st.sidebar.slider("Overal Casa 22",0, 15, 1)
#    Potencial_22_home = st.sidebar.slider("Potencial Casa 22",0, 15, 1)
#    Overal_21_home = st.sidebar.slider("Overal Casa 21",0, 15, 1)
#    Potencial_21_home = st.sidebar.slider("Potencial Casa 21",0, 15, 1)
    st.write('Visitante:', optionv)
    option1 = st.sidebar.selectbox(
    'O Jogo sera realizado em local neutro?',('Sim','Nao'))
    
#    user_data = {'home_team': optionc, 'Pontos Fifa Casa': home_team_fifa_rank, 'Pontos Fifa Casa': home_team_total_fifa_points,
#                 'Overal Casa 23':Overal_23_home, 'Overal Casa 22':Overal_22_home ,'Overal Casa 21':Overal_21_home,
#                 'Potencial Casa 23': Potencial_23_home, 'Potencial Casa 22':Potencial_22_home, 'Potencial Casa 21':Potencial_21_home
#                 } 
#    user_data2 = {'away_team':optionv, 'Pontos Fifa Visitante': away_team_fifa_rank, 'Pontos Fifa Visitante': away_team_total_fifa_points,
#                 'Overal Visitante 23':Overal_23_away, 'Overal Visitante 22':Overal_22_away ,'Overal Visitante 21':Overal_21_away,
#                 'Potencial Visitante 23': Potencial_23_away, 'Potencial Visitante 22':Potencial_22_away, 'Potencial Visitante 21':Potencial_21_away
#                 }
#    user_data = {'home_team': optionc, 'Ranking Fifa Casa': home_team_fifa_rank, 'Pontos Fifa Casa': home_team_total_fifa_points,
#                 'Overal Casa 23':Overal_23_home, 'Overal Casa 22':Overal_22_home ,'Overal Casa 21':Overal_21_home,
#                 'Potencial Casa 23': Potencial_23_home, 'Potencial Casa 22':Potencial_22_home, 'Potencial Casa 21':Potencial_21_home,
#                'away_team':optionv, 'Ranking Fifa Visitante': away_team_fifa_rank, 'Pontos Fifa Visitante': away_team_total_fifa_points,
#                 'Overal Visitante 23':Overal_23_away, 'Overal Visitante 22':Overal_22_away ,'Overal Visitante 21':Overal_21_away,
#                 'Potencial Visitante 23': Potencial_23_away, 'Potencial Visitante 22':Potencial_22_away, 'Potencial Visitante 21':Potencial_21_away
#                 } 
 
    user_data = {'home_team':optionc}
    user_data2 = {'away_team':optionv}
    features = pd.DataFrame(user_data, index=[0])
    features2 = pd.DataFrame(user_data2, index=[0])

    return features, features2
    
user_input_variables1, user_input_variables2 = get_user_date()

user_input_variables1 = user_input_variables1.reset_index()
user_input_variables1.rename(columns = {'index':'home_team_cod'}, inplace=True)
df_timesh_ = user_input_variables1.merge(df_timesh, how = 'inner', on = 'home_team_cod')

user_input_variables2 = user_input_variables2.reset_index()
user_input_variables2.rename(columns = {'index':'away_team_cod'}, inplace=True)
df_timesa_ = user_input_variables2.merge(df_timesa, how = 'inner', on = 'away_team_cod')

user_input_variables = pd.concat([df_timesa_, df_timesh_], axis = 1)
user_input_variables.drop(columns = ['away_team','home_team'], inplace=True)    #   remove the columns  that are already    in place    for
#print(user_input_variables.head())
user_input_variables.to_excel('user.xlsx')
#search = user_input_variables.merge(x_train)

#grafico
#graf = st.bar_chart(user_input_variables)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 1)
x_train_, y_train_ = sm.fit_resample(x_train, y_train)

#Treinar
SEED = 2
split = 10
num_trees = 200
kfold = StratifiedKFold(n_splits = split, shuffle = True)

#Validacao
from sklearn.model_selection import (KFold,
                                     cross_val_score,
                                     StratifiedKFold,
                                     GridSearchCV,
                                     train_test_split,
)

rf = RandomForestClassifier(random_state=1, n_estimators = num_trees, max_depth = 7, n_jobs = -1, min_samples_leaf=5)

results = cross_val_score(rf, x_cup, y_cup, cv = kfold)

rf.fit(x_train, y_train)

y_pred_train = rf.predict(x_train)
y_pred = rf.predict(x_test)
y_pred_proba = rf.predict_proba(x_cup)

#acurácia do modelo

#st.subheader('Acurácia do modelo')

#st.write(accuracy_score(y_test, rf.predict(x_test))*100)

#previsão do resultado

prediction = rf.predict_proba(user_input_variables)

pred = pd.DataFrame(prediction, columns = ['Time da Casa','Visitante'])
st.subheader('Previsão:')

st.write(pred)

print(pred)
