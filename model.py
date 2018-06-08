#!/usr/bin/python3
"""
Modelo de regresion con arboles de decision usando sklearn
#Autor: Fernando Rodrigo Aguilar Javier
#Correo: faguilar@comunidad.unam.mx
"""
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def model_nive(train, test):
	numerical_features = train.select_dtypes(exclude=['object']).columns
	column_interest = numerical_features[:-1]
	#print(column_interest)
	X_train = train[column_interest]
	X_train = X_train.dropna(axis=0)
	Y_train = train.SalePrice[:1121]
	X_test = test[column_interest]
	X_test = X_test.dropna(axis=0)[:1121]
	#print(X_test.shape, X_train.shape, Y_train.shape)
	#Creamos el modelo
	House_saleprice_model = DecisionTreeRegressor()
	#Entrenamos el modelo
	House_saleprice_model.fit(X_train, Y_train)
	predicted_home_saleprices = House_saleprice_model.predict(X_test)
	return mean_absolute_error(Y_train, predicted_home_saleprices)

def model_column_interest(train, test):
	column_interest = ['OverallQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'Fireplaces', 'GarageCars', 'GarageArea']
	X_train = train[column_interest]
	X_train = X_train.dropna(axis=0)[:1457]
	Y_train  =  train.SalePrice[:1457]
	X_test = test[column_interest]
	X_test = X_test.dropna(axis=0)
	print(X_train.shape, Y_train.shape, X_test.shape)
	House_saleprice_model = DecisionTreeRegressor()
	#Entrenamos el modelo
	House_saleprice_model.fit(X_train, Y_train)
	predicted_home_saleprices = House_saleprice_model.predict(X_test)
	print(mean_absolute_error(Y_train, predicted_home_saleprices))
	return X_train, X_test, Y_train, predicted_home_saleprices

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

if __name__ == '__main__':
	train = pd.read_csv('train.csv',  index_col=0)
	test = pd.read_csv('test.csv',  index_col=0)
	#print(model_nub(train, test))
	X_train, X_test, Y_train, predicted_home_saleprices = model_column_interest(train, test)
	arr_mae, arr_n_nodes = [], []
	for max_leaf_nodes in [2, 5, 50, 500,1000, 3000]:
		my_mae = get_mae(max_leaf_nodes, X_train, X_test, Y_train, predicted_home_saleprices)
		arr_mae.append(my_mae)
		arr_n_nodes.append(max_leaf_nodes)
		print("Maximo numero de hojas: %d  \t\t Valor absoluto medio del error:  %d" %(max_leaf_nodes, my_mae))
		
