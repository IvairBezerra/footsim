import pandas as pd
import numpy as np
from sympy import acsc
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
from scipy import stats
import matplotlib.pyplot as plt

#import base de partidas
df = pd.read_csv(r'international_matches.csv')

#ajuste nome dos times
df.loc[df['home_team'] == 'IR Iran', 'home_team'] = 'Iran'
df.loc[df['home_team'] == 'USA', 'home_team'] = 'United States'
df.loc[df['home_team'] == 'Guinea-Bissau', 'home_team'] = 'Guinea Bissau'
df.loc[df['away_team'] == 'IR Iran', 'away_team'] = 'Iran'
df.loc[df['away_team'] == 'USA', 'away_team'] = 'United States'
df.loc[df['away_team'] == 'Guinea-Bissau', 'away_team'] = 'Guinea Bissau'

#import bases de stats jogadores
df_fifa_23 = pd.read_csv(r'FIFA23_official_data.csv')
df_fifa_22 = pd.read_csv(r'FIFA22_official_data.csv')
df_fifa_21 = pd.read_csv(r'FIFA21_official_data.csv')
df_fifa_20 = pd.read_csv(r'FIFA20_official_data.csv')
df_fifa_19 = pd.read_csv(r'FIFA19_official_data.csv')
df_fifa_18 = pd.read_csv(r'FIFA18_official_data.csv')
df_fifa_17 = pd.read_csv(r'FIFA17_official_data.csv')
df_fifa_16 = pd.read_csv(r'players_16.csv')
df_fifa_15 = pd.read_csv(r'players_15.csv')

#Cálculo do Overall médio e Potencial médio, por ano
dfgroup1 = df_fifa_23.groupby(by=['Nationality'], dropna=False).agg(Overal_23=('Overall','mean'),Potencial_23=('Potential','mean')).reset_index()
dfgroup2 = df_fifa_22.groupby(by=['Nationality'], dropna=False).agg(Overal_22=('Overall','mean'),Potencial_22=('Potential','mean')).reset_index()
dfgroup3 = df_fifa_21.groupby(by=['Nationality'], dropna=False).agg(Overal_21=('Overall','mean'),Potencial_21=('Potential','mean')).reset_index()
dfgroup4 = df_fifa_20.groupby(by=['Nationality'], dropna=False).agg(Overal_20=('Overall','mean'),Potencial_20=('Potential','mean')).reset_index()
dfgroup5 = df_fifa_19.groupby(by=['Nationality'], dropna=False).agg(Overal_19=('Overall','mean'),Potencial_19=('Potential','mean')).reset_index()
dfgroup6 = df_fifa_18.groupby(by=['Nationality'], dropna=False).agg(Overal_18=('Overall','mean'),Potencial_18=('Potential','mean')).reset_index()
dfgroup7 = df_fifa_17.groupby(by=['Nationality'], dropna=False).agg(Overal_17=('Overall','mean'),Potencial_17=('Potential','mean')).reset_index()

#Junção das bases
df1 = df.merge(dfgroup1, left_on='home_team', right_on='Nationality', how= 'left')
df1 = df1.merge(dfgroup1, left_on='away_team', right_on='Nationality', how= 'left')
df1.rename(columns={'Overal_23_x':'Overal_23_home','Potencial_23_x':'Potencial_23_home','Overal_23_y':'Overal_23_away','Potencial_23_y':'Potencial_23_away'},inplace = True)
df1.drop(columns=['Nationality_x','Nationality_y'],inplace = True)

df1 = df1.merge(dfgroup2, left_on='home_team', right_on='Nationality', how= 'left')
df1 = df1.merge(dfgroup2, left_on='away_team', right_on='Nationality', how= 'left')
df1.rename(columns={'Overal_22_x':'Overal_22_home','Potencial_22_x':'Potencial_22_home','Overal_22_y':'Overal_22_away','Potencial_22_y':'Potencial_22_away'},inplace = True)
df1.drop(columns=['Nationality_x','Nationality_y'],inplace = True)

df1 = df1.merge(dfgroup3, left_on='home_team', right_on='Nationality', how= 'left')
df1 = df1.merge(dfgroup3, left_on='away_team', right_on='Nationality', how= 'left')
df1.rename(columns={'Overal_21_x':'Overal_21_home','Potencial_21_x':'Potencial_21_home','Overal_21_y':'Overal_21_away','Potencial_21_y':'Potencial_21_away'},inplace = True)
df1.drop(columns=['Nationality_x','Nationality_y'],inplace = True)

df1 = df1.merge(dfgroup4, left_on='home_team', right_on='Nationality', how= 'left')
df1 = df1.merge(dfgroup4, left_on='away_team', right_on='Nationality', how= 'left')
df1.rename(columns={'Overal_20_x':'Overal_20_home','Potencial_20_x':'Potencial_20_home','Overal_20_y':'Overal_20_away','Potencial_20_y':'Potencial_20_away'},inplace = True)
df1.drop(columns=['Nationality_x','Nationality_y'],inplace = True)

df1 = df1.merge(dfgroup5, left_on='home_team', right_on='Nationality', how= 'left')
df1 = df1.merge(dfgroup5, left_on='away_team', right_on='Nationality', how= 'left')
df1.rename(columns={'Overal_19_x':'Overal_19_home','Potencial_19_x':'Potencial_19_home','Overal_19_y':'Overal_19_away','Potencial_19_y':'Potencial_19_away'},inplace = True)
df1.drop(columns=['Nationality_x','Nationality_y'],inplace = True)

df1 = df1.merge(dfgroup6, left_on='home_team', right_on='Nationality', how= 'left')
df1 = df1.merge(dfgroup6, left_on='away_team', right_on='Nationality', how= 'left')
df1.rename(columns={'Overal_18_x':'Overal_18_home','Potencial_18_x':'Potencial_18_home','Overal_18_y':'Overal_18_away','Potencial_18_y':'Potencial_18_away'},inplace = True)
df1.drop(columns=['Nationality_x','Nationality_y'],inplace = True)

df1 = df1.merge(dfgroup7, left_on='home_team', right_on='Nationality', how= 'left')
df1 = df1.merge(dfgroup7, left_on='away_team', right_on='Nationality', how= 'left')
df1.rename(columns={'Overal_17_x':'Overal_17_home','Potencial_17_x':'Potencial_17_home','Overal_17_y':'Overal_17_away','Potencial_17_y':'Potencial_17_away'},inplace = True)
df1.drop(columns=['Nationality_x','Nationality_y'],inplace = True)

lst_cup =['Brazil',
'Belgium',
'Argentina',
'France',
'England',
'Spain',
'Netherlands',
'Portugal',
'Denmark',
'Germany',
'Mexico',
'Uruguay',
'United States',
'Croatia',
'Switzerland',
'Senegal',
'Wales',
'Iran',
'Morocco',
'Japan',
'Serbia',
'Poland',
'Korea Republic',
'Tunisia',
'Costa Rica',
'Cameroon',
'Australia',
'Canada',
'Ecuador',
'Qatar',
'Saudi Arabia',
'Ghana']

mask_cup = (df1['home_team'].isin(lst_cup)) | (df1['away_team'].isin(lst_cup))
df_cup = df1[mask_cup]
df_cup

df1['result'] = np.where(df1['home_team_score'] > df1['away_team_score'], 1, 0)
df_cup['result'] = np.where(df_cup['home_team_score'] > df_cup['away_team_score'], 1, 0)

df1.dropna(inplace = True)
df1.isna().sum()/df1.shape[0]
df_cup.dropna(inplace = True)
df_cup.isna().sum()/df_cup.shape[0]

df1['ndate'] = df1['date'].map(lambda x: pd.to_datetime(x))

#dropando as colunas com info nula
df1.drop(columns = ['home_team_goalkeeper_score',
 'away_team_goalkeeper_score',
 'home_team_mean_defense_score',
 'home_team_mean_offense_score',
 'home_team_mean_midfield_score',
 'away_team_mean_defense_score',
 'away_team_mean_offense_score',
 'away_team_mean_midfield_score'], inplace = True)

df_cup['ndate'] = df_cup['date'].map(lambda x: pd.to_datetime(x))

#dropando as colunas com info nula
df_cup.drop(columns = ['home_team_goalkeeper_score',
 'away_team_goalkeeper_score',
 'home_team_mean_defense_score',
 'home_team_mean_offense_score',
 'home_team_mean_midfield_score',
 'away_team_mean_defense_score',
 'away_team_mean_offense_score',
 'away_team_mean_midfield_score'], inplace = True)

df1.sort_values(by = 'away_team', ascending=True, inplace=True)
dict = pd.factorize(df1['away_team'])
dict_cup = pd.factorize(df_cup['away_team'])

df1['away_team_cod'] = dict[0]
df_cup['away_team_cod'] = dict_cup[0]

df1.sort_values(by = 'home_team', ascending=True, inplace=True)
dict1 = pd.factorize(df1['home_team'])
dict_cup1 = pd.factorize(df_cup['home_team'])

df1['home_team_cod'] = dict1[0]
df_cup['home_team_cod'] = dict_cup1[0]

mask_year = df1['date']  >= '2017' 
df2 = df1[mask_year]

mask_year_cup = df_cup['date']  >= '2017'
df_cup_recente = df_cup[mask_year_cup]

from sklearn.preprocessing import LabelEncoder

lab = LabelEncoder()

df2['home_team_result_f'] = lab.fit_transform(df2['home_team_result'])


#Modelo
from sklearn.ensemble import RandomForestClassifier

#Validacao
from sklearn.model_selection import (KFold,
                                     cross_val_score,
                                     StratifiedKFold,
                                     GridSearchCV,
                                     train_test_split,
)

#Processamento
from sklearn.preprocessing import StandardScaler 

#Metricas
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report

x = df2.drop(columns = ['shoot_out','home_team','away_team', 'home_team_result_f','ndate', 'date','home_team_continent','away_team_continent',
'tournament','city','country','home_team_score','away_team_score','home_team_result','home_team_result_f','result'])
y = df2['result']



x_cup = df_cup_recente.drop(columns = ['shoot_out','home_team','away_team','ndate', 'date','home_team_continent','away_team_continent',
'tournament','city','country','home_team_score','away_team_score','home_team_result','result'])
y_cup = df_cup_recente['result']

df_cup_recente.to_csv("df_cup.csv")

x_train, x_test, y_train, y_test = train_test_split(x_cup, y_cup, test_size = 0.2, random_state = 2)

#!pip install imblearn
#!pip install SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 1)
x_train_, y_train_ = sm.fit_resample(x_train, y_train)

#Treinar
SEED = 2
split = 10
num_trees = 200
kfold = StratifiedKFold(n_splits = split, shuffle = True)

rf = RandomForestClassifier(random_state=1, n_estimators = num_trees, max_depth = 7, n_jobs = -1, min_samples_leaf=5)
#results = cross_val_score(rf, x, y, cv = kfold)
results = cross_val_score(rf, x_cup, y_cup, cv = kfold)

rf.fit(x_train, y_train)

y_pred_train = rf.predict(x_train)
y_pred = rf.predict(x_test)
y_pred_proba = rf.predict_proba(x_cup)

cm = confusion_matrix(y_test, y_pred)
print('Matriz de confusão:','\n', cm)

print('Report: ', '\n', classification_report(y_pred, y_test))

print('Acuracia Treino:', accuracy_score(y_pred_train, y_train)*100)
print('Acuracia Teste:', accuracy_score(y_pred, y_test)*100)
print(y_pred_proba)

#df_cup_recente['prob_ganhar'] = y_pred_proba[1]
df_proba = pd.DataFrame(y_pred_proba,columns=['prob_perder', 'prob_ganhar'])
teste = df_proba['prob_ganhar']
print(teste)

# nao entendi pq nao to conseguindo passar as prob. para o DF completo.. 
df_cup_recente['prob_ganhar'] = teste

teste.to_excel('teste.xlsx')
df_cup_recente.to_excel('df_cup_recente.xlsx')
x_cup.to_excel('x_cup.xlsx')
