import os
import pickle
import time
from math import sqrt
from multiprocessing.sharedctypes import Value
from pathlib import Path
from urllib.error import URLError
from sklearn import svm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st
from sklearn import datasets, linear_model, neighbors, preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (classification_report, mean_squared_error,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

@st.cache
def get_data():
    path = r'footballData.csv'
    return pd.read_csv(path)
    df = get_data()

st.title('Football Player Price Prediction System')

dataframe = pd.read_csv(r'C:/Users/msidh/Documents/footballData.csv')
playerNamesDataframe = pd.read_csv(r'C:/Users/msidh/Documents/footballData.csv')
dataframe.drop(['real_face','player_tags','sofifa_id','player_url','long_name','dob','league_name','international_reputation','player_tags','loaned_from','team_jersey_number','joined','nation_position','international_reputation','nation_jersey_number','nationality','body_type','player_traits','lwb','rcm','cm','rdm','lb','rwb', 'rw','lm','lw', 'lcm', 'lcb','cb', 'rcb', 'rm','st','cf','lf','rf','lam','cam', 'ram','ls','rs', 'rb', 'ldm', 'cdm','team_position', "gk_diving","gk_handling","gk_kicking","gk_reflexes","gk_positioning"], axis=1, inplace=True)
dataframe.dropna(subset = ['player_positions', 'value_eur'], inplace=True) 
dataframe['release_clause_eur'].fillna(0, inplace=True)
dataframe['league_rank'].fillna(5, inplace=True)
dataframe.drop(dataframe.index[dataframe['value_eur'] == 0], inplace=True)
dataframe.dropna(subset = ['club_name', 'contract_valid_until'], inplace=True) 
dataframe.drop(columns = ['defending_marking'], inplace=True) 
dataframe['attacking_work_rate'] = dataframe['work_rate'].map(lambda x: x.split('/')[0])
dataframe['defensive_work_rate'] = dataframe['work_rate'].map(lambda x: x.split('/')[1])
dataframe.drop(columns = ['work_rate'], inplace=True) 
cleanup_work_rates = {"attacking_work_rate":     {"Low": 1, "Medium": 2, "High": 3},
                "defensive_work_rate": {"Low": 1, "Medium": 2, "High": 3}}
dataframe = dataframe.replace(cleanup_work_rates)
new_dataframe = dataframe.loc[:, ~dataframe.columns.isin(['short_name','club_name','player_positions','wage_eur', 'preferred_foot', 'value_eur'])]
normalisation = MinMaxScaler() 
array_normalised = normalisation.fit_transform(new_dataframe) 
columns_not_being_normalised = ['short_name','club_name','player_positions','wage_eur', 'preferred_foot', 'value_eur']
dataframe_normalised = pd.DataFrame(array_normalised, columns=new_dataframe.columns,index=new_dataframe.index)
joined_dataframe = pd.concat([dataframe[columns_not_being_normalised],dataframe_normalised], axis = 1)
joined_dataframe['player_positions_one'] = joined_dataframe['player_positions'].map(lambda x: x.split(',')[0])
joined_dataframe.drop(columns = ['player_positions'], inplace=True) 
cleanup_player_positions = {"player_positions_one":     {"GK": 1, "LWB": 2, "LB": 3, "CB": 4, "RWB": 5, "RB": 6, "CDM": 7, "CM": 8, "LM": 9,"RM": 10, "CAM": 11, "CF": 12, "LW": 13, "RW": 14, "ST": 15 }}
joined_dataframe = joined_dataframe.replace(cleanup_player_positions)
list_of_positions = joined_dataframe.player_positions_one.unique().tolist()
joined_dataframe.drop(columns =['club_name', 'league_rank','release_clause_eur'], inplace=True) 
cleanup_preferred_foot = {"preferred_foot":     {"Left": 0, "Right": 1}}
joined_dataframe = joined_dataframe.replace(cleanup_preferred_foot)
newestdf = joined_dataframe

goalkeepers = ['GK']
goalkeepers_dataframe = newestdf[newestdf['player_positions_one'] == 1] 

removing_goalkeeping_stats = ["gk_speed"]
outfield_dataframe = newestdf[newestdf['player_positions_one'] != 1]

outfield_dataframe.drop(columns = removing_goalkeeping_stats, inplace=True) 

deletedRows = ["shooting", "pace","passing", "dribbling", "defending", "physic", "skill_moves"]

goalkeepers_dataframe.drop(columns = deletedRows, inplace=True) 
print(outfield_dataframe.columns)

playerNameCol = ["short_name"]

copygoalkeepers_dataframe = goalkeepers_dataframe.copy()
copygoalkeepers_dataframe.drop(columns = playerNameCol, inplace=True) 
copyoutfielddf = outfield_dataframe.copy()
copyoutfielddf.drop(columns = playerNameCol, inplace=True) 

#position in the list is the index value in the dataframe 

copygoalkeepers_dataframe.dropna(inplace=True)
print("copygoalie information from dataframe")
copygoalkeepers_dataframe.info()

st.write("Player you want to find the price for")
players = playerNamesDataframe['short_name'].tolist()
playerName = st.selectbox("Select target player for price prediction", players)

def playerIndex(playerName):
    for i in range(len(players)):
        if playerName == players[i]:
            return i
        
def playerAndDataframeRequired(playerName):
    
    if playerName in outfield_dataframe.values:
        return outfield_dataframe, copyoutfielddf
    else:
        return goalkeepers_dataframe, copygoalkeepers_dataframe


def removePlayerAndPlayerNamesFromDataframe(playerName, dataframe):
    rowForPlayer = dataframe.loc[dataframe['short_name'] == playerName].copy()
    print(rowForPlayer)
    return rowForPlayer
    
def removePlayerFromDataframe(playerName, dataframe):
    playerNameIndex = playerIndex(playerName)
    dataframe = dataframe.drop(playerNameIndex)
    
    return dataframe
    
def percentageDiff(actualValue, predictedValue):
    percentageDifference = ((predictedValue - actualValue)/actualValue)*100
    
    return percentageDiff

def playerValue(playerName, dataframe):
    valueOfPlayer = dataframe.loc[dataframe['short_name'] == playerName, 'value_eur']
    
    return valueOfPlayer

playerNameIndex = playerIndex(playerName)

@st.cache(suppress_st_warning=True)
def linearRegression(playerName):
    np.random.seed(101)
    dataframe = playerAndDataframeRequired(playerName)[0]
    dataframeForModel = playerAndDataframeRequired(playerName)[1]
    rowOfPlayer = removePlayerAndPlayerNamesFromDataframe(playerName, dataframe)
    dataframeCleaned = removePlayerFromDataframe(playerName,dataframeForModel)
    train , test = train_test_split(dataframeCleaned, test_size = 0.3, random_state = 42)

    x_train = train.drop('value_eur', axis=1)
    y_train = train['value_eur']

    x_test = test.drop('value_eur', axis = 1)
    y_test = test['value_eur']

    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    
    regressionModel = regr.predict(rowOfPlayer.drop(rowOfPlayer.columns[[0,3]], axis=1))
    return regressionModel

@st.cache(suppress_st_warning=True)
def knnBestValueGraph(x_train, y_train, x_test, y_test):
    rmseValues = []
    for k in range(1,51):
        model = neighbors.KNeighborsRegressor(n_neighbors = k)

        model.fit(x_train, y_train)  #fit the model
        pred=model.predict(x_test) #make prediction on test set
        error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
        rmseValues.append(error) #store rmse values
        print('RMSE value for k= ' , k , 'is:', error)
        
@st.cache(suppress_st_warning=True)
def knnRegression(playerName):
    np.random.seed(101)
    dataframe = playerAndDataframeRequired(playerName)[0]
    print(dataframe)
    dataframeForModel = playerAndDataframeRequired(playerName)[1]
    print(dataframeForModel)
    rowOfPlayer = removePlayerAndPlayerNamesFromDataframe(playerName, dataframe)
    print(rowOfPlayer)
    dataframeCleaned = removePlayerFromDataframe(playerName,dataframeForModel)
    dataframeCleaned.info()
    
    train , test = train_test_split(dataframeCleaned, test_size = 0.3, random_state = 42)

    x_train = train.drop('value_eur', axis=1)
    y_train = train['value_eur']
    
    print("x train shape is:",x_train.shape)
    print("x train columns are:", x_train.columns)
    x_test = test.drop('value_eur', axis=1)
    y_test = test['value_eur']
    
    
    x_dimension = np.arange(1,100)
    print(x_dimension)
    params = {'n_neighbors':x_dimension,  "weights": ["uniform", "distance"]}

    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=5)
    model.fit(x_train,y_train)

    predictedValue = model.predict(rowOfPlayer.drop(rowOfPlayer.columns[[0,3]], axis=1))
    
    knnBestValueGraph(x_train, y_train, x_test, y_test)
    return model, model.best_params_, predictedValue

        
#append every value to an array, value of k and what the model predicts

# potentially new method with bagging 

@st.cache(suppress_st_warning=True)
def decisionTreeRegression(playerName):
    np.random.seed(101)
    depthValue = st.slider('What tree depth do you want to use', 0, 50, 1)
    dataframe = playerAndDataframeRequired(playerName)[0]
    dataframeForModel = playerAndDataframeRequired(playerName)[1]
    rowOfPlayer = removePlayerAndPlayerNamesFromDataframe(playerName, dataframe)
    dataframeCleaned = removePlayerFromDataframe(playerName,dataframeForModel)
    train , test = train_test_split(dataframeCleaned, test_size = 0.3, random_state = 42)

    x_train = train.drop('value_eur', axis=1)
    y_train = train['value_eur']

    x_test = test.drop('value_eur', axis = 1)
    y_test = test['value_eur']

    tree_reg = DecisionTreeRegressor(max_depth=depthValue, random_state= 42)
    tree_reg.fit(x_train, y_train)
    valueOfPlayer = tree_reg.predict(rowOfPlayer.drop(rowOfPlayer.columns[[0,3]], axis=1))
    
    importance = tree_reg.feature_importances_
    columns = x_train.columns.values
    print('{:.3e}'.format(importance[0]))
    #print("the importance of the decision tree regression")
    
    return valueOfPlayer

@st.cache(suppress_st_warning=True)
def randomForestRegression(playerName):
    np.random.seed(101)
    treeValue = st.slider('How many trees do you want the forest to have', 0, 200, 1)
    maxFeatureVal = st.selectbox('What max feature value would you like to use?',('log2', 'auto', 'sqrt'))

    dataframe = playerAndDataframeRequired(playerName)[0]
    dataframeForModel = playerAndDataframeRequired(playerName)[1]
    rowOfPlayer = removePlayerAndPlayerNamesFromDataframe(playerName, dataframe)
    dataframeCleaned = removePlayerFromDataframe(playerName,dataframeForModel)
    train , test = train_test_split(dataframeCleaned, test_size = 0.2, random_state = 42)

    x_train = train.drop('value_eur', axis=1)
    y_train = train['value_eur']

    x_test = test.drop('value_eur', axis = 1)
    y_test = test['value_eur']
        
    regressor = RandomForestRegressor(n_estimators = treeValue, max_features=maxFeatureVal , random_state = 42)
    regressor.fit(x_train, y_train) 
    
    predictedValue = regressor.predict(rowOfPlayer.drop(rowOfPlayer.columns[[0,3]], axis=1))

    return predictedValue


@st.cache(suppress_st_warning=True)
def linearSVR(playerName):
    np.random.seed(101)
    dataframe = playerAndDataframeRequired(playerName)[0]
    dataframeForModel = playerAndDataframeRequired(playerName)[1]
    rowOfPlayer = removePlayerAndPlayerNamesFromDataframe(playerName, dataframe)
    dataframeCleaned = removePlayerFromDataframe(playerName,dataframeForModel)
    train , test = train_test_split(dataframeCleaned, test_size = 0.3, random_state = 42)

    x_train = train.drop('value_eur', axis=1)
    y_train = train['value_eur']


    x_test = test.drop('value_eur', axis = 1)
    y_test = test['value_eur']

    regr = svm.LinearSVR()
    regr.fit(x_train, y_train)
    
    predictedValue = regr.predict(rowOfPlayer.drop(rowOfPlayer.columns[[0,3]], axis=1))
    
    return predictedValue


st.write("Linear Regression Model")
lrResult = linearRegression(playerName)
st.write("The predicted price of {} is {}".format(playerName, lrResult[0]))


st.write("KNN Regression Model")
knnResult = knnRegression(playerName)
st.write("The predicted price of {} is {}".format(playerName, knnResult[2]))
kValue = st.checkbox("Best K Value")

if kValue:
    st.write(knnRegression(playerName)[1])
    

st.write("Random Forest Regression Model")
randomForestResult = randomForestRegression(playerName)
st.write("The predicted price of {} is {}".format(playerName, randomForestResult))


st.write("Decision Tree Regression Model")
decisionTreeReg = decisionTreeRegression(playerName)
st.write("The predicted price of {} is {}".format(playerName,decisionTreeReg))


st.write("Linear SVR Model")
linearSVRPrice = linearSVR(playerName)
st.write("The predicted price of {} is {}".format(playerName,linearSVRPrice))
