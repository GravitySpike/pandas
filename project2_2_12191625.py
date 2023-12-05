import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

def sort_dataset(dataset_df):
	return dataset_df.sort_values(by='year')

def split_dataset(dataset_df):	
	X_train=dataset_df.iloc[:1718]
	X_test=dataset_df.iloc[1718:]

	dataset_df.insert(37, 's_label', sorted_df.salary.map(lambda x: x*0.001))

	Y_train=dataset_df.loc[:,'s_label'].iloc[:1718]
	Y_test=dataset_df.loc[:,'s_label'].iloc[1718:]

	Y_train=Y_train.astype('int')
	Y_test = Y_test.astype('int')

	return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(dataset_df):
	return dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]

def train_predict_decision_tree(X_train, Y_train, X_test):
	dt=DecisionTreeClassifier()
	dt.fit(X_train, Y_train)
	return dt.predict(X_test)

def train_predict_random_forest(X_train, Y_train, X_test):
	rf = RandomForestClassifier()
	rf.fit(X_train, Y_train)
	return rf.predict(X_test)

def train_predict_svm(X_train, Y_train, X_test):
	pipe=make_pipeline(
		StandardScaler(),
		SVC()
	)
	pipe.fit(X_train, Y_train)
	return pipe.predict(X_test)

def calculate_RMSE(labels, predictions):
	mse=mean_squared_error(labels, predictions)
	rmse=np.sqrt(mse)
	return rmse

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))