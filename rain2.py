__author__ = 'Raihan Masud'
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pylab as pl
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.externals import joblib

"""
ToDo: Feature Engineering
See more:
http://blog.kaggle.com/2015/07/01/how-much-did-it-rain-winners-interview-1st-place-devin-anzelmo/
https://www.kaggle.com/c/how-much-did-it-rain/forums/t/14242/congratulations
http://blog.kaggle.com/2015/06/08/how-much-did-it-rain-winners-interview-2nd-place-no-rain-no-gain/
http://blog.kaggle.com/2015/05/07/profiling-top-kagglers-kazanovacurrently-2-in-the-world/
"""

# region data prep

#ToDo: clean up train data with all missing input but valid label. put zero on label for such data
#if one of the 4 related features (%5..%90) has no value..hard to predict
def load_data(file):
    if "test" in file:
        data = pd.read_csv(file, nrows=1048576)
        #test_id.append(np.array(data['Id']))
    else: #train data
        data = pd.read_csv(file)

    return data


def clean_data(data):
    data = data.drop(['minutes_past', 'radardist_km'], axis=1)

    #remove data empty rows Ref values all nan
    # data = data.set_index('Id')
    # ref_sums = data['Ref'].groupby(level='Id').sum()
    # null_refs_idx = [i for i in ref_sums.index if np.isnan(ref_sums[i])]
    # data.drop(null_refs_idx, axis = 0, inplace = True)
    return data


test_valid_ids = []
empty_rows_ids = []
def transform_data(data, file):
    data_avg = data.groupby(['Id']).mean()
    #print(data_avg.columns)
    #print(data_avg.index)

    if "test" in file:
        global empty_rows_ids
        empty_rows_ids = data[np.isnan(data['Ref'])].Id.tolist()

        global test_valid_id
        test_valid_id = data[np.isfinite(data['Ref'])].Id.tolist()

        #id = data['Id'].tolist()
        #dist_id = set(id)

        #test_valid_id = list(dist_id)
    return data_avg


def remove_outlier(train_data):
    #average rainfall per hour historically less than 5 inch or 127 mm #70 is considered strom
    train_data = train_data[train_data['Expected'] <= 50]
    #set expected values to zero for examples that has most feature values(< 5) = 0
    #print(train_data.head(5))
    #change expected value where more than four features values are empty (0)
    #train_data.ix[(train_data == 0).astype(int).sum(axis=1) > 4, 'Expected'] = 0
    #print(train_data.head(5))
    return train_data


def remove_empty_rows(data):
    #remove data empty rows Ref values all nan
    #print(data.columns)

    data = data[np.isfinite(data['Ref'])]

    #data = data.set_index('Id')
    #ref_sums = data['Ref'] #.groupby(level='Id').sum()
    #null_refs_idx = [i for i in ref_sums.index if np.isnan(ref_sums[i])]
    #data.drop(null_refs_idx, axis = 0, inplace = True)
    return data


def analyze_plot_data(data, type):
    #if data series data
    if isinstance(data, pd.Series):
        data.hist(color='green', alpha=0.5, bins=50, orientation='horizontal', figsize=(16, 8))
        #plt.title("distribution of samples in -> " + data.name)
        #plt.ylabel("frequency")
        #plt.xlabel("value")
        pl.suptitle("kaggle_rain2_" + type + "_" + data.name)
        #plt.show()
        file_to_save = "kaggle_rain2_" + type + "_" + data.name + ".png"
        path = os.path.join("./charts/", file_to_save)
        plt.savefig(path)
    else:  #plot all data features/columns
        for i in range(0,len(data.columns), 4):
            #plt.title("distribution of samples in -> " + data.columns[i])
            #plt.ylabel("frequency")
            #plt.xlabel("value")
            data[data.columns[i: i+4]].hist(color='green', alpha=0.5, bins=50, figsize=(16, 8))
            pl.suptitle("kaggle_rain2_" + type + "_" + data.columns[i])
            #plt.show()
            file_to_save = "kaggle_rain2_" + type + "_" + data.columns[i] + ".png"
            path = os.path.join("./charts/", file_to_save)
            plt.savefig(path)


            #plt.figure()
            #print(data.min())
            #basic statistics of the data
            #print(data.describe())

            #data.hist(color='k', alpha=0.5, bins=25)
            #plt.hist(data, bins=25, histtype='bar')
            #plt.title(data.columns[0]+"distribution in train sample")
            #plt.savefig(feature_name+".png")
            #plt.show()


scaler = preprocessing.StandardScaler()


def standardize_data(X):
    mean = X.mean(axis=0)
    X -= mean
    std = X.std(axis=0)
    X /= std
    standardized_data = X
    return standardized_data

def normal_distribute_data(X):

    #RhoHV is not normally distributed
    #taking Log
    #transformer = preprocessing.FunctionTransformer(np.log1p)
    #transformer.transform(X['RhoHV'])

    #print(X['RhoHV'].describe())

    #X['RhoHV'] = np.log10(X['RhoHV'])
    #comment if removed as feature
    #X['RhoHV_5x5_10th'] = np.log10(X['RhoHV_5x5_10th'])
    #X['RhoHV_5x5_50th'] = np.log10(X['RhoHV_5x5_50th'])
    #X['RhoHV_5x5_90th'] = np.log10(X['RhoHV_5x5_90th'])

    # rhoFeatures = ['RhoHV']#,'RhoHV_5x5_10th','RhoHV_5x5_50th','RhoHV_5x5_90th']
    # for rhoFeature in rhoFeatures:
    #     shiftBy = 0
    #     rhoMean = X[rhoFeature].mean()
    #     if rhoMean < 0:
    #         shiftBy += abs(rhoMean)
    #     X[rhoFeature] += shiftBy

    return X


def imputeData(non_empty_data):
    non_empty_data.fillna(0)
    return non_empty_data

def prepare_train_data(file_path):
    print("preparing training data...")
    #train_file_path = "./train/train_short.csv"
    train_data = load_data(file_path)
    train_clean = clean_data(train_data)
    train_no_outlier = remove_outlier(train_clean)
    transformed_data = transform_data(train_no_outlier, file_path)
    non_empty_data = remove_empty_rows(transformed_data)


    X_train = standardize_data(non_empty_data)
    #print(X_train.columns)
    X_train = normal_distribute_data(X_train)


    imputed_data = imputeData(X_train)
    labels = imputed_data['Expected']
    X_train = imputed_data.drop(['Expected'], axis=1)

    #drop features
    X_train = X_train.drop(['Ref_5x5_10th','Ref_5x5_50th','Ref_5x5_90th',
                                  'RefComposite_5x5_10th','RefComposite_5x5_50th','RefComposite_5x5_90th',
                                  'RhoHV_5x5_10th','RhoHV_5x5_50th','RhoHV_5x5_90th',
                                  'Zdr_5x5_10th','Zdr_5x5_50th','Zdr_5x5_90th',
                                  'Kdp_5x5_10th','Kdp_5x5_50th','Kdp_5x5_90th'], axis=1)

    X_train = X_train.drop(['RhoHV'], axis=1)
    #print(X_train.head(5000))
    return X_train, labels


def prepare_test_data(file_path):
    #file_path = "./test/test.csv"
    #test_file_path = file_test #from kaggle site
    #test_file_path = "./test/test_short.csv"
    test_data = load_data(file_path)
    test_clean = clean_data(test_data)
    transformed_data = transform_data(test_clean, file_path)
    non_empty_data = remove_empty_rows(transformed_data)
    imputed_data = imputeData(non_empty_data)

    #drop features
    X_test = imputed_data.drop(['Ref_5x5_10th','Ref_5x5_50th','Ref_5x5_90th',
                                  'RefComposite_5x5_10th','RefComposite_5x5_50th','RefComposite_5x5_90th',
                                  'RhoHV','RhoHV_5x5_10th','RhoHV_5x5_50th','RhoHV_5x5_90th',
                                  'Zdr_5x5_10th','Zdr_5x5_50th','Zdr_5x5_90th',
                                  'Kdp_5x5_10th','Kdp_5x5_50th','Kdp_5x5_90th'], axis=1)



    X_test = standardize_data(X_test)
    #global test_id
    #test_id = test_avg['Id']
    #test_input = test_avg.drop(['Id'], axis=1)
    return X_test  #test_input


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
    clf_rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=1,
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


def train(train_input, labels):
    #print("loading & preparing training data...")
    #train_input, labels = prepare_train_data()

    #clf = evaluate_models(labels, train_input)
    #n_estimators = no. of trees in the forest
    #n_jobs = #no. of cores
    clf_rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=None, n_jobs=4, min_samples_split=1, random_state=0)
    #extree = ensemble.ExtraTreesRegressor(n_estimators=100, max_depth=None, min_samples_split=1, n_jobs=-1)

    #params = {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 1,
    #          'learning_rate': 0.01, 'loss': 'ls', 'max_features':5}
    #clf_gbt = ensemble.GradientBoostingRegressor(**params)

    clf = clf_rf

    #print("cross validating...")
    #cv_score(clf, train_input, labels)


    #train_test does shuffle and random splits
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        train_input, labels, test_size=0.07, random_state=0)

    global model

    print("training model...")
    #model = clf.fit(train_input, labels)
    model = clf.fit(X_train, y_train)

    #feature engineering
    print("feature selection...")
    print("feature importance", model.feature_importances_)

    #print("testing on holdout set...")
    pred_y = model.predict(X_test)

    #no_of_pred = len(pred_y)

    # MAE=0
    # for i,v in enumerate(pred_y):
    #     actual = y_test.values[i]
    #     predicted = v
    #     error = actual - predicted
    #     print("rainfall actual: {0} predicted:{1}, error:{2}".
    #           format(actual, predicted, np.abs(error)))
    #     MAE = MAE + np.abs(error)
    # #print("MAE: ",MAE/no_of_pred)

    print("scikit MAE", mean_absolute_error(y_test, pred_y))

    #pickle model
    #joblib.dump(clf, 'rain2.pkl')

    return model


#endregion train

#test
def write_prediction(test_y):
    print("writing to file....")
    predictionDf = pd.DataFrame(index=test_valid_ids, columns=['Expected'], data=test_y)

    empty_test_y = np.asanyarray([0 for i in range(0,empty_rows_ids)])
    emptyRowsDf =  pd.DataFrame(index=empty_rows_ids, columns=['Expected'], data=empty_test_y)

    totalDf = pd.concat(predictionDf,emptyRowsDf)
    totalDf.sort([totalDf.index])
    # write file
    prediction_file = './rain_prediction.csv'
    predictionDf.to_csv(prediction_file, index_label='Id', float_format='%.6f')


def predict(model, test_input):
    print("predicting....")
    #test_input = prepare_test_data()  #replace with test data
    #print(train_input.iloc[[1]], labels[2])
    test_y = model.predict(test_input)
    print(len(test_valid_ids))
    print(len(test_y))

    return test_y



#report
# print("loading & preparing training data...")
train_file_path = "./train/train.csv"
train_input, labels = prepare_train_data(train_file_path)
#analyze_plot_data(train_input, "training")
#analyze_plot_data(labels, "training")

model = train(train_input, labels)

#test_file_path = "./test/test.csv"
#X_test = prepare_test_data(test_file_path)
#analyze_plot_data(X_test, "test")
#test_y=predict(model, X_test)
#write_prediction(test_y)

