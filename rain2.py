__author__ = 'Raihan Masud'
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import linear_model
from sklearn.linear_model import Ridge
from glob import glob

"""
ToDo: Feature Engineering
See more:
http://blog.kaggle.com/2015/07/01/how-much-did-it-rain-winners-interview-1st-place-devin-anzelmo/
https://www.kaggle.com/c/how-much-did-it-rain/forums/t/14242/congratulations
http://blog.kaggle.com/2015/06/08/how-much-did-it-rain-winners-interview-2nd-place-no-rain-no-gain/
http://blog.kaggle.com/2015/05/07/profiling-top-kagglers-kazanovacurrently-2-in-the-world/
"""

#region data prep

file_train = glob('../input/train/train.csv')
file_test = glob('../input/test/test.csv')


#ToDo: clean up train data with all missing input but valid label. put zero on label for such data
#if one of the 4 related features (%5..%90) has no value..hard to predict
def load_data(file):
    data = pd.read_csv(file)
    if "test" in file:
        test_id.append(np.array(data['Id']))
    return data

def clean_data(data):
    clean_d = data.drop(['minutes_past','radardist_km'],axis=1)
    clean_d = clean_d.fillna(0) #fill missing values with zero initially
    return clean_d

test_id = []
def normalize_data(clean_d, file):
    #ToDo: instead of mean do standarization
    data_avg =clean_d.groupby(['Id']).mean()
    if "test" in file:
        id = clean_d['Id'].tolist()
        dist_id = set(id)
        global test_id
        test_id = list(dist_id)
    return data_avg

def remove_outlier(train_data):
    #average rainfall per hour historically less than 5 inch or 127 mm #70 is considered strom
    train_data = train_data[train_data['Expected'] < 70]
    #set expected values to zero for examples that has most feature values(< 5) = 0
    #print(train_data.head(5))
    train_data.ix[(train_data == 0).astype(int).sum(axis=1) > 4, 'Expected'] = 0
    #print(train_data.head(5))
    return train_data

def plot_data(data, feature_name):
    #plt.figure()
    data.hist(color='k', alpha=0.5, bins=25)
    #plt.hist(data, bins=25, histtype='bar')
    #plt.title(feature_name+"distribution in train sample")
    #plt.ylabel(feature_name)
    #plt.savefig(feature_name+".png")
    plt.show()

scaler= preprocessing.StandardScaler()
def standardize_transform(data):
    X_prep = scaler.fit_transform(data)
    return  X_prep

def prepare_train_data():
    train_file_path = "./train/train.csv"
    #train_file_path = "./train/train_short.csv"
    train_data = load_data(train_file_path)
    train_clean = clean_data(train_data)
    train_avg = normalize_data(train_clean, train_file_path)
    train_no_outlier = remove_outlier(train_avg)

    labels = train_no_outlier['Expected']
    train_input = train_no_outlier.drop(['Expected'], axis=1)
    X_train = train_input #no_standarization #standardize_transform(train_input)
    return X_train, labels

def prepare_test_data():

    test_file_path = "./test/test.csv"
    #test_file_path = file_test #from kaggle site
    #test_file_path = "./test/test_short.csv"
    test_data = load_data(test_file_path)
    test_clean = clean_data(test_data)
    test_avg = normalize_data(test_clean, test_file_path)
    global test_id
    #test_id = test_avg['Id']
    #test_input = test_avg.drop(['Id'], axis=1)
    return test_avg #test_input

#endregion data prep

#region train




def evaluate_models(labels, train_input):
    print("evaluating models...")
    #regr = linear_model.LinearRegression()
    #ridge = Ridge(alpha=1.0)
    #laso = linear_model.Lasso(alpha=0.1)
    #enet = linear_model.ElasticNet(alpha=0.1)
    #clf_dtr = tree.DecisionTreeRegressor()
    #ada = ensemble.AdaBoostRegressor(n_estimators=500, learning_rate=.75)

    #bag = ensemble.BaggingRegressor(n_estimators=500)

    #extee = ensemble.ExtraTreesRegressor(n_estimators=500, max_depth=None, min_samples_split=1)
    clf_rf = ensemble.RandomForestRegressor(n_estimators=10, max_depth=None, min_samples_split=1,
                                            random_state=0, max_features="auto")
    #params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
    #          'learning_rate': 0.01, 'loss': 'ls'}
    #clf_gbt = ensemble.GradientBoostingRegressor(**params)

    #print(len(train_input))
    #print(len(labels))
    clf = clf_rf
    # model evaluator
    scores = cross_validation.cross_val_score(clf, train_input, labels, cv=5, scoring='mean_absolute_error')
    print("CV score: ", abs(sum(scores) / len(scores)))
    #model evaluator
    """ model evaluation

        NEW:
        Random Forest evaluation score      :  1.13246481639
        Extra Tree evaluation score         :  1.13414660721
        Bagging Regressor evaluation score  :  1.15816159605
        Gradient Boosting evaluation score  :  1.17339099314

        #linear regression evaluation score:  1.74012818638
        #ridge regression evaluation score:  1.72820341712
        #lasso regression evaluation score:  1.58996750817
        #elastic_net evaluation score:  1.60092318938
        #dtree regression evaluation score:  1.64168047513
        #adaBoost regressor evaluation score:  2.81744083141
        #Bagging Regressor evaluation score:  1.1617702616

        #random forest evaluation score:  1.44005742499
        #random forest evaluation score:  1.35075879522 with params
        params_rf = {'n_estimators': 500, 'max_depth':None, 'min_samples_split':1, 'random_state':0}

        #gradient boosted tree evaluation score:  1.47009354892
        #gradient boosted tree evaluation score:  1.42787523525 with params
        #{'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1, 'learning_rate': 0.01, 'loss': 'ls'}
        """
    return clf

def cv_score(clf, X, y):
    scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='mean_absolute_error')
    print("CV score: ", abs(sum(scores) / len(scores)))


model = None
def train():
    print("loading & preparing training data...")
    train_input, labels = prepare_train_data()
    #plot_data(train_input, "Ref")
    print("loaded {0} training examples".format(len(train_input)))

    #clf = evaluate_models(labels, train_input)
    #n_estimators = no. of trees in the forest
    #n_jobs = #no. of cores
    #clf_rf = ensemble.RandomForestRegressor(n_estimators=50, max_depth=None, n_jobs=4, min_samples_split=1,
    #                                        max_features="auto")
    extree = ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=None, min_samples_split=1, n_jobs=-1)

    #params = {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 1,
    #          'learning_rate': 0.01, 'loss': 'ls', 'max_features':5}
    #clf_gbt = ensemble.GradientBoostingRegressor(**params)

    clf = extree

    #print("cross validating...")
    #cv_score(clf, train_input, labels)

    #train_test does shuffle and random splits
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    train_input, labels, test_size=0.05, random_state=0)


    global model

    print("training model...")
    model = clf.fit(train_input, labels)
    #model = clf.fit(X_train, y_train)

    print("testing on holdout set...")
    pred_y = model.predict(X_test)
    no_of_pred = len(pred_y)

    MAE=0
    for i,v in enumerate(pred_y):
        actual = y_test.values[i]
        predicted = v
        error = actual - predicted
        print("rainfall actual: {0} predicted:{1}, error:{2}".
              format(actual, predicted, np.abs(error)))
        MAE = MAE + np.abs(error)
    #print("MAE: ",MAE/no_of_pred)

    print("scikit MAE", mean_absolute_error(y_test, pred_y))

    return model
#endregion train


#test
def predict(model):
    print("predicting....")
    prediction_file = './rain_prediction.csv'

    test_input = prepare_test_data() #replace with test data
    test_y = model.predict(test_input)
    print(len(test_id))
    print(len(test_y))
    print("writing to file....")
    predictionDf = pd.DataFrame(index=test_id, columns=['Expected'], data=test_y)
    # # write file
    predictionDf.to_csv(prediction_file, index_label='Id', float_format='%.6f')


#report
model = train()
predict(model)

