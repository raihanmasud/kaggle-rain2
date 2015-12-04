__author__ = 'Raihan Masud'

import os
import numpy as np
import pandas as pd
import ctypes

pd.options.mode.chained_assignment = None
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score
matplotlib.style.use('ggplot')
import pylab as pl
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.preprocessing import Imputer
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.externals import joblib
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
import pickle
import sys
from sklearn.grid_search import GridSearchCV
from joblib import Parallel, delayed

# ToDo: This file needs cleanup

"""
Zdr: Differential reflectivity : it is a good indicator of drop shape and drop shape is a good estimate of average drop size.
RhoHV:  Correlation coefficient: A statistical correlation between the reflected horizontal and vertical power returns.
                                 High values, near one, indicate homogeneous precipitation types,
                                 while lower values indicate regions of mixed precipitation types, such as rain and snow, or hail.
Kdp:Specific differential phase: It is a very good estimator of rain rate and is not affected by attenuation. The range derivative of
                                 differential phase (specific differential phase, Kdp) can be used to localize areas of strong precipitation/attenuation.
"""

"""
ToDo: Feature Engineering
See more:
http://blog.kaggle.com/2015/07/01/how-much-did-it-rain-winners-interview-1st-place-devin-anzelmo/
https://www.kaggle.com/c/how-much-did-it-rain/forums/t/14242/congratulations
http://blog.kaggle.com/2015/06/08/how-much-did-it-rain-winners-interview-2nd-place-no-rain-no-gain/
http://blog.kaggle.com/2015/05/07/profiling-top-kagglers-kazanovacurrently-2-in-the-world/
"""

# region data prep





# ToDo: clean up train data with all missing input but valid label. put zero on label for such data
# if one of the 4 related features (%5..%90) has no value..hard to predict
def load_data(file, load_partial):
  # traing data #of rows 13,765,201
  #test data #of rows 8,022,756
  if "test" in file:
    if load_partial:
      data = pd.read_csv(file, nrows=22757)
    else:
      data = pd.read_csv(file)
      #test_id.append(np.array(data['Id']))
  else:  #train data
    if load_partial:
      data = pd.read_csv(file, nrows=1065201)
    else:
      data = pd.read_csv(file)

  print("loaded data of " + str(data.shape))

  return data


def clean_data(data):
  data = data.drop(['minutes_past'], axis=1)

  # remove data empty rows Ref values all nan
  # data = data.set_index('Id')
  # ref_sums = data['Ref'].groupby(level='Id').sum()
  # null_refs_idx = [i for i in ref_sums.index if np.isnan(ref_sums[i])]
  # data.drop(null_refs_idx, axis = 0, inplace = True)
  return data


def add_class(data):
  #create an empty series
  c_data = [100 for _ in range(data.shape[0])]
  c_series = pd.Series(c_data, name='Class')
  #concat class column with data
  pd.concat([data, c_series], axis=1, join='inner')

  #change the rows of class to 1
  data.loc[data['Expected'] <= 2.0, 'Class'] = 0
  #data.loc[(data['Expected'] > 1.0) & (data['Expected'] <= 3.0), 'Class'] = 'Light'
  data.loc[(data['Expected'] > 2.0) & (data['Expected'] <= 15.0), 'Class'] = 1
  #data.loc[(data['Expected'] > 15.0) & (data['Expected'] <= 25.0), 'Class'] = 'Moderate Heavy'
  data.loc[data['Expected'] > 15.0 , 'Class'] = 2

  #print(data[['Expected','Class']].head(100))

  return data

  #note: non matching values get converted to NaN for -ve values


def add_features(data):
  # Ref
  Ref_MAX = data.groupby(['Id'], sort=False)['Ref'].max()
  Ref_MAX.name = 'Ref_MAX'

  Ref_MIN = data.groupby(['Id'], sort=False)['Ref'].min()
  Ref_MIN.name = 'Ref_MIN'

  Ref_count = data.groupby(['Id'], sort=False)['Ref'].count()
  Ref_count.name = 'Ref_count'

  Ref_std = data.groupby(['Id'], sort=False)['Ref'].std()
  #Ref_std = Ref_std.pow(2)
  Ref_std.name = 'Ref_std'

  Ref_med = data.groupby(['Id'], sort=False)['Ref'].median()
  Ref_med.name = 'Ref_med'

  #Ref_skew = data.groupby(['Id'], sort=False)['Ref'].skew()
  #Ref_skew.name = 'Ref_skew'

  #Ref_mad = data.groupby(['Id'], sort=False)['Ref'].mad()
  #Ref_mad.name = 'Ref_mad'

  #Ref_kurt = data.groupby(['Id'], sort=False)['Ref'].kurtosis()
  #Ref_kurt.name = 'Ref_kurt'


  RefComposite_MAX = data.groupby(['Id'], sort=False)['RefComposite'].max()
  RefComposite_MAX.name = 'RefComposite_MAX'

  RefComposite_MIN = data.groupby(['Id'], sort=False)['RefComposite'].min()
  RefComposite_MIN.name = 'RefComposite_MIN'

  RefComposite_count = data.groupby(['Id'], sort=False)['RefComposite'].count()
  RefComposite_count.name = 'RefComposite_count'

  RefComposite_std = data.groupby(['Id'], sort=False)['RefComposite'].std()
  #RefComposite_std = RefComposite_std.pow(2)
  RefComposite_std.name = 'RefComposite_std'

  RefComposite_med = data.groupby(['Id'], sort=False)['RefComposite'].median()
  RefComposite_med.name = 'RefComposite_med'

  #RefComposite_skew = data.groupby(['Id'], sort=False)['RefComposite'].skew()
  #RefComposite_skew.name = 'RefComposite_skew'

  #RefComposite_mad = data.groupby(['Id'], sort=False)['RefComposite'].mad()
  #RefComposite_mad.name = 'RefComposite_mad'

  #RefComposite_kurt = data.groupby(['Id'], sort=False)['RefComposite'].kurtosis()
  #RefComposite_kurt.name = 'RefComposite_kurt'



  #Zdr_MAX = data.groupby(['Id'], sort=False)['Zdr'].max()
  #Zdr_MAX.name = 'Zdr_MAX'

  #Zdr_MIN = data.groupby(['Id'], sort=False)['Zdr'].min()
  #Zdr_MIN.name = 'Zdr_MIN'

  Zdr_count = data.groupby(['Id'], sort=False)['Zdr'].count()
  Zdr_count.name = 'Zdr_count'

  Zdr_std = data.groupby(['Id'], sort=False)['Zdr'].std()
  #Zdr_std = Zdr_std.pow(2)
  Zdr_std.name = 'Zdr_std'

  Zdr_med = data.groupby(['Id'], sort=False)['Zdr'].median()
  Zdr_med.name = 'Zdr_med'

  #Zdr_skew = data.groupby(['Id'], sort=False)['Zdr'].skew()
  #Zdr_skew.name = 'Zdr_skew'

  #Zdr_mad = data.groupby(['Id'], sort=False)['Zdr'].mad()
  #Zdr_mad.name = 'Zdr_mad'

  #Zdr_kurt = data.groupby(['Id'], sort=False)['Zdr'].kurtosis()
  #Zdr_kurt.name = 'Zdr_kurt'


  Kdp_MAX = data.groupby(['Id'], sort=False)['Kdp'].max()
  Kdp_MAX.name = 'Kdp_MAX'

  Kdp_MIN = data.groupby(['Id'], sort=False)['Kdp'].min()
  #Kdp_MIN = Kdp_MIN.pow(2)
  Kdp_MIN.name = 'Kdp_MIN'

  Kdp_count = data.groupby(['Id'], sort=False)['Kdp'].count()
  Kdp_count.name = 'Kdp_count'

  #todo: kdp std should be added back
  # Kdp_std = data.groupby(['Id'], sort=False)['Kdp'].std()
  #Kdp_std = Kdp_std.pow(2)
  #Kdp_std.name = 'Kdp_std'

  #Kdp_med = data.groupby(['Id'], sort=False)['Kdp'].median()
  #Kdp_med.name = 'Kdp_med'


  #Kdp_skew = data.groupby(['Id'], sort=False)['Kdp'].skew()
  #Kdp_skew.name = 'Kdp_skew'

  #Kdp_mad = data.groupby(['Id'], sort=False)['Kdp'].mad()
  #Kdp_mad.name = 'Kdp_mad'

  #Kdp_kurt = data.groupby(['Id'], sort=False)['Kdp'].kurtosis()
  #Kdp_kurt.name = 'Kdp_kurt'



  #RhoHV_MAX = data.groupby(['Id'], sort=False)['RhoHV'].max()
  #RhoHV_MAX.name = 'RhoHV_MAX'

  #RhoHV_MIN = data.groupby(['Id'], sort=False)['RhoHV'].min()
  #RhoHV_MIN.name = 'RhoHV_MIN'

  RhoHV_count = data.groupby(['Id'], sort=False)['RhoHV'].count()
  RhoHV_count.name = 'RhoHV_count'

  RhoHV_std = data.groupby(['Id'], sort=False)['RhoHV'].std()
  #RhoHV_std = RhoHV_std.pow(2)
  RhoHV_std.name = 'RhoHV_std'

  RhoHV_med = data.groupby(['Id'], sort=False)['RhoHV'].median()
  RhoHV_med.name = 'RhoHV_med'

  #RhoHV_skew = data.groupby(['Id'], sort=False)['RhoHV'].skew()
  #RhoHV_skew.name = 'RhoHV_skew'

  #RhoHV_mad = data.groupby(['Id'], sort=False)['RhoHV'].mad()
  #RhoHV_mad.name = 'RhoHV_mad'

  #RhoHV_kurt = data.groupby(['Id'], sort=False)['RhoHV'].kurtosis()
  #RhoHV_kurt.name = 'RhoHV_kurt'

  return Ref_MAX, Ref_MIN, Ref_count, Ref_std, Ref_med, \
         RefComposite_MAX, RefComposite_MIN, RefComposite_count, RefComposite_std, RefComposite_med, \
         Zdr_count, Zdr_std, Zdr_med, \
         Kdp_MAX, Kdp_MIN, Kdp_count,   \
         RhoHV_count, RhoHV_std, RhoHV_med

#RhoHV_MIN, RhoHV_MAX, Zdr_MIN, Zdr_MAX, Kdp_med, Kdp_std

test_all_ids = []
test_non_empty_ids = []
test_empty_rows_ids = []


def transform_data(data, file):
  #Ref = NaN means no rain fall at that instant, safe to remove

  #avg the valid Ref over the hour
  data_avg = data.groupby(['Id']).mean()  #just using mean CV score:  23.4247096352

  #print('columns', str(data_avg.columns))

  if "test" in file:
    global test_all_ids
    test_all_ids = data_avg.index

    global test_empty_rows_ids
    test_empty_rows_ids = data_avg.index[np.isnan(data_avg['Ref'])]

    global test_non_empty_ids
    test_non_empty_ids = list((set(test_all_ids) - set(test_empty_rows_ids)))

  data = data[np.isfinite(data['Ref'])]

  #data = data[np.isfinite(data['Ref'])] #CV 23.4481724075

  print("creating features...")
  Ref_Max, Ref_Min, Ref_count, Ref_std, Ref_med, \
  RefComposite_MAX, RefComposite_MIN, RefComposite_count, RefComposite_std, RefComposite_med, \
  Zdr_count, Zdr_std, Zdr_med,\
  Kdp_MAX, Kdp_MIN, Kdp_count,  \
  RhoHV_count, RhoHV_std, RhoHV_med = add_features(data)
  #RhoHV_MAX, Zdr_MIN, Kdp_med, # Kdp_std,

  print("adding features...")
  data_avg = pd.concat([data_avg, Ref_Max, Ref_Min, Ref_count, Ref_std, Ref_med,
                         RefComposite_MAX, RefComposite_MIN, RefComposite_count, RefComposite_std, RefComposite_med,
                         Zdr_count, Zdr_std, Zdr_med,
                         Kdp_MAX, Kdp_MIN, Kdp_count,
                         RhoHV_count, RhoHV_std, RhoHV_med], axis=1, join='inner')

  global features
  features = data_avg.columns
  #id = data['Id'].tolist()
  #dist_id = set(id)

  #test_valid_id = list(dist_id)
  return data_avg


def remove_outlier(train_data):
  #average rainfall per hour historically less than 5 inch or 127 mm #70 is considered strom
  #100 gives 23.6765613211
  #70 gives 23.426143398 // keep 70 as acceptable rain fall value
  #50 gives 23.26343648


  data = train_data[train_data['Expected'] <= 50]
  #print(data.shape)


  #data['Expected'].hist(color='green', cumulative=True, alpha=0.5, bins=50, orientation='horizontal', figsize=(16, 8))
  #plt.show()

  """
  data45 = data[data['Expected'] > 40]
  print("40 < data < 50 ", data45.shape)

  data40 = data[data['Expected'] <= 40]
  print(data45.shape)

  data34 = data40[data40['Expected'] > 30]
  print("30 < data <= 40 ", data34.shape)

  data30 = data40[data40['Expected'] <= 30]
  print(data30.shape)

  data23 = data30[data30['Expected'] > 20]
  print("20 < data <= 30 ",data23.shape)

  data20 = data30[data30['Expected'] <= 20]
  print(data20.shape)

  data12 = data20[data20['Expected'] > 10]
  print("10 < data <= 20 ",data12.shape)

  data10 = data20[data20['Expected'] <= 10]
  print(" < data <= 10 ",data10.shape)
  """


  #set expected values to zero for examples that has most feature values(< 5) = 0
  #print(train_data.head(5))
  #change expected value where more than four features values are empty (0)
  #train_data.ix[(train_data == 0).astype(int).sum(axis=1) > 4, 'Expected'] = 0
  #print(train_data.head(5))

  return data


def remove_empty_rows(data):
  # remove data empty rows Ref values all nan
  #print(data.columns)

  data = data[np.isfinite(data['Ref'])]
  #print(data.shape)

  # remove Ref values less than 5 as it's not raining
  #data = data[data['Ref'] >= 5] #CV Score: 23.4583024399
  #print(data.shape)

  #data = data[data['Ref'] <= 50] #CV Score: 23.4583024399
  #print(data.shape)


  #data sanity check
  #Ref [5,10,15,20,25,30,35,40,45,50] vs Rain [0.07,0.15,0.3,0.6,1.3,2.7,5.6,11.53,23.7,48.6]
  ref_rain = [(5,0.07),(10,0.15),(15,0.3),(20,0.6),(25,1.3),(30,2.7),(35,5.6),(40,11.53),(45,23.7),(50,48.6)]
  rho_hv = [1, 0.99, 0.988, 0.986, 0.982, 0.98, 0.978, 0.976, 0.974, 0.0972]
  zdr    = [1.1+0.4*i for i in range(0,10)]
  kdp    = [0.5*i for i in range(0,10)]

  for index, row in data.iterrows():
    if np.isnan(row['RhoHV']): #fill Nan with appropriate value
      mul5 = int(row['Ref'] // 5)
      mul5 = 0 if mul5 < 0 else mul5 #-ve values to 0
      mul5 = 9 if mul5 > 9 else mul5 #>9 values to 9
      #ref_rain[mul5][1]
      data.loc[[index], 'RhoHV'] = rho_hv[mul5]

    if np.isnan(row['Zdr']): #fill Nan with appropriate value
      mul5 = int(row['Ref'] // 5)
      mul5 = 0 if mul5 < 0 else mul5 #-ve values to 0
      mul5 = 9 if mul5 > 9 else mul5 #>9 values to 9
      #ref_rain[mul5][1]
      data.loc[[index], 'Zdr'] = zdr[mul5]

    if np.isnan(row['Kdp']): #fill Nan with appropriate value
      mul5 = int(row['Ref'] // 5)
      mul5 = 0 if mul5 < 0 else mul5 #-ve values to 0
      mul5 = 9 if mul5 > 9 else mul5 #>9 values to 9
      #ref_rain[mul5][1]
      data.loc[[index], 'Kdp'] = zdr[mul5]

  #mul5 = 1
  # suspect_rows = []
  # for index, row in data.iterrows():
  #   mul5 = int(row['Ref'] // 5)
  #   if mul5 < 1 or mul5 > 9:
  #     print("invalid index {0} produced by {1}".format(mul5, row['Ref']))
  #     continue
  #   if abs(row['Expected'] - ref_rain[mul5-1][1]) > 30 :
  #     print("index: {0} Ref: {1} Expected: {2}".format(index,row['Ref'],row['Expected']))
  #     suspect_rows.append(index)
  # print("total suspected data "+str(len(suspect_rows)))
  # print("suspected data indices"+str(suspect_rows))



  #data = data[np.isfinite(data['Ref_5x5_10th'])]

  #data = data.set_index('Id')
  #ref_sums = data['Ref'] #.groupby(level='Id').sum()
  #null_refs_idx = [i for i in ref_sums.index if np.isnan(ref_sums[i])]
  #data.drop(null_refs_idx, axis = 0, inplace = True)
  return data


def analyze_plot_data(data, type):
  # if data series data
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
    for i in range(0, len(data.columns), 4):
      #plt.title("distribution of samples in -> " + data.columns[i])
      #plt.ylabel("frequency")
      #plt.xlabel("value")
      data[data.columns[i: i + 4]].hist(color='green', alpha=0.5, bins=50, figsize=(16, 8))
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


def plot_training_curve(model, X, y):
  params = ["min_samples_leaf", "min_samples_split"]
  p_range = [2, 4, 8, 10, 12, 14, 16, 18, 20]
  # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

  for param in params:
    print("plotting validation curve...")
    train_scores, valid_scores = validation_curve(model, X, y, param_name=param, param_range=p_range, cv=3,
                                                  scoring='mean_absolute_error')
    train_scores_mean = np.absolute(np.mean(train_scores, axis=1))
    valid_scores_mean = np.absolute(np.mean(valid_scores, axis=1))
    plt.title("Validation Curve with GBM")
    plt.xlabel(param)
    plt.ylabel("MAE")

    plt.plot(p_range, train_scores_mean, label="Training Error", color="r", marker='o')
    plt.plot(p_range, valid_scores_mean, label="Cross-validation Error", color="g", marker='s')
    plt.legend(loc="best")
    plt.show()

    # t_sizes = [5000, 10000, 15000, 20000, 25000]
    # train_sizes, lc_train_scores, lc_valid_scores = learning_curve(model, X, y, train_sizes=t_sizes, cv=3)
    # print("plotting learning curve...")
    # lc_train_scores_mean = np.absolute(np.mean(lc_train_scores, axis=1))
    # lc_valid_scores_mean = np.absolute(np.mean(lc_valid_scores, axis=1))
    #
    # plt.title("Learning Curve with GBM")
    # plt.xlabel("no. of examples")
    # plt.ylabel("MAE")
    #
    # plt.plot(train_sizes, lc_train_scores_mean, label="Training score", color="r", marker='o')
    # plt.plot(train_sizes, lc_valid_scores_mean, label="Cross-validation score", color="g", marker='s')
    # plt.legend(loc="best")
    # plt.show()


scaler = preprocessing.StandardScaler()


def standardize_data(X):
  mean = X.mean(axis=0)
  X -= mean
  std = X.std(axis=0)
  X /= std
  standardized_data = X
  return standardized_data


def normal_distribute_data(X):
  # RhoHV is not normally distributed
  #taking Log
  #transformer = preprocessing.FunctionTransformer(np.log1p)
  #transformer.transform(X['RhoHV'])

  #print(X['RhoHV'].describe())

  X['RhoHV'] = X['RhoHV'].apply(lambda x: np.log10(x))

  #comment if removed as feature
  #X['RhoHV_5x5_10th'] = np.log10(X['RhoHV_5x5_10th'])
  #X['RhoHV_5x5_50th'] = np.log10(X['RhoHV_5x5_50th'])
  #X['RhoHV_5x5_90th'] = np.log10(X['RhoHV_5x5_90th'])

  rhoFeatures = ['RhoHV']  #,'RhoHV_5x5_10th','RhoHV_5x5_50th','RhoHV_5x5_90th']
  for rhoFeature in rhoFeatures:
    shiftBy = 0
    rhoMean = X[rhoFeature].mean()
    if rhoMean < 0:
      shiftBy += abs(rhoMean)
    X[rhoFeature] += shiftBy

  return X


imp = Imputer(missing_values='NaN', strategy='mean', axis=0)


def impute_data(non_empty_data):

  """
  calculate rain rate from Ref using marshal algo

  """
  imp.fit(non_empty_data)
  X = imp.transform(non_empty_data)  # 23.2592765571 (better)
  return X  # non_empty_data.fillna(0) #23.2628586644 # X


def prepare_train_data(file_path, load_Partial):
  print("preparing training data...")
  train_data = load_data(file_path, load_Partial)
  train_clean = clean_data(train_data)
  train_no_outlier = remove_outlier(train_clean)
  print("transforming data...")
  transformed_data = transform_data(train_no_outlier, file_path)
  non_empty_examples = remove_empty_rows(transformed_data)

  #non_empty_examples = add_class(non_empty_examples)

  labels = non_empty_examples['Expected']
  #categoy = non_empty_examples['Class']

  #labels.hist(cumulative=True,bins=50)
  #plt.show()

  # print('total data: '+str(non_empty_examples.shape[0]))
  #
  # global train_data_light_rain
  # train_data_light_rain = non_empty_examples[non_empty_examples['Class'] == 0]
  #
  # print('light rain data: '+str(train_data_light_rain.shape[0]))
  #
  #
  # global train_data_moderate_rain
  # train_data_moderate_rain = non_empty_examples[non_empty_examples['Class'] == 1]
  # print('moderate rain data: '+str(train_data_moderate_rain.shape[0]))
  #
  #
  # global train_data_heavy_rain
  # train_data_heavy_rain = non_empty_examples[non_empty_examples['Class'] == 2]
  # print('heavy rain data: '+str(train_data_heavy_rain.shape[0]))

  X_train = non_empty_examples.drop(['Expected'], axis=1)
  #X_train = non_empty_examples.drop(['Expected','Class'], axis=1)


  global X_columns
  X_columns = X_train.columns

  X_train = standardize_data(X_train)
  # labels = standardize_data(labels)

  #X_train = normal_distribute_data(X_train)

  #drop features
  #X_train = X_train.drop([#'Ref_5x5_10th','Ref_5x5_50th'
  #                                'Ref_5x5_90th',
  #                               'RefComposite_5x5_10th','RefComposite_5x5_50th','RefComposite_5x5_90th',
  #'RhoHV_5x5_10th','RhoHV_5x5_50th','RhoHV_5x5_90th'
  #,'Zdr_5x5_10th','Zdr_5x5_50th','Zdr_5x5_90th',
  #                               'Kdp_5x5_10th','Kdp_5x5_50th','Kdp_5x5_90th'
  #                        ], axis=1)

  X_train = impute_data(X_train)

  #X_train = X_train.drop(['RhoHV'], axis=1)
  #print(X_train.head(5000))
  return X_train, labels #, categoy


def prepare_test_data(file_path, load_partial):
  print("preparing test data...")
  # file_path = "./test/test.csv"
  #test_file_path = file_test #from kaggle site
  #test_file_path = "./test/test_short.csv"
  test_data = load_data(file_path, load_partial)
  #test_file_path = file_test #from kaggle site
  #test_file_path = "./test/test_short.csv"
  test_clean = clean_data(test_data)
  transformed_data = transform_data(test_clean, file_path)
  non_empty_data = remove_empty_rows(transformed_data)

  #print('test data', str(non_empty_data.columns))

  X_test = standardize_data(non_empty_data)
  #X_test = normal_distribute_data(X_test)


  #drop features
  #X_test = X_test.drop([#'Ref_5x5_10th','Ref_5x5_50th','Ref_5x5_90th',
  #                               'RefComposite_5x5_10th','RefComposite_5x5_50th','RefComposite_5x5_90th',
  #                                'RhoHV_5x5_10th','RhoHV_5x5_50th','RhoHV_5x5_90th'
  #                               ,'Zdr_5x5_10th','Zdr_5x5_50th','Zdr_5x5_90th',
  #                               'Kdp_5x5_10th','Kdp_5x5_50th','Kdp_5x5_90th'
  #                           ], axis=1)

  X_test = impute_data(X_test)


  #global test_id
  #test_id = test_avg['Id']
  #test_input = test_avg.drop(['Id'], axis=1)
  return X_test, non_empty_data  #test_input


# endregion data prep

#region train

def evaluate_models(train_input, labels):
  print("evaluating models...")
  #regr = linear_model.LinearRegression()
  #ridge = Ridge(alpha=1.0)
  #laso = linear_model.Lasso(alpha=0.1)
  #enet = linear_model.ElasticNet(alpha=0.1)
  #clf_dtr = tree.DecisionTreeRegressor()
  #ada = ensemble.AdaBoostRegressor(n_estimators=500, learning_rate=.75)

  #bag = ensemble.BaggingRegressor(n_estimators=500)

  param_grid = {
                'min_samples_split': [4, 8, 12, 16,20, 24, 30, 35, 40, 45, 50],
                'min_samples_leaf': [8,10,12,14, 18, 20, 22, 25]
                }
  #increase n_estimators to 400ish
  est = ensemble.GradientBoostingRegressor(n_estimators=400)

  extree = ensemble.ExtraTreesRegressor(n_estimators=800, max_features=1.0, n_jobs=-1)  #-1 sets it to #of cores


  gs_cv = GridSearchCV(extree,param_grid,score_func='mean_absolute_error').fit(train_input,labels)

  #best parameters {'min_samples_leaf': 4, 'max_depth': 4, 'min_samples_split': 4, 'max_leaf_nodes': 6, 'max_features': 0.5} from grid search gave score 0.1647636331960101
  #best parameters {'max_depth': 4, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': 0.5, 'max_leaf_nodes': 8} from grid search gave score 0.17065558286341795


  print("best parameters {0} from grid search gave score {1} ".format(gs_cv.best_params_, gs_cv.best_score_))

  params = gs_cv.best_params_
  #clf = ensemble.GradientBoostingRegressor(n_estimators = 150,**params)
  # cv_scre_last = 100
  # for ne in range(20, 400, 10):
  #
  #   #clf_rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0, max_features="auto")
  #
  ne, ss, md = 190, 25, 10  #CV score: 24.340700094525427

  clf = extree

  # n_estimators=150 CV score: 24.401973843565866 //too slow 170 gives 24.39021427337333
  #ne, md, ss = 50, 10, 10
  #clf_gbt = ensemble.GradientBoostingRegressor(n_estimators=ne, max_depth=md, min_samples_split=ss, min_samples_leaf=10, learning_rate=0.1, loss='ls')
  #
  #   #print(len(train_input))
  #   #print(len(labels))
  #   clf = clf_gbt
  #   # model evaluator
  scores = cross_validation.cross_val_score(clf, train_input, labels, cv=5, scoring='mean_absolute_error')
  cv_scre = 21.5 + abs(sum(scores) / len(scores))
  print("CV score: {0} - #of_estimators: {1}".format(cv_scre, ne))
  #
  #   if cv_scre >= cv_scre_last:
  #     print("MAE score didn't improve for #of_estimators: " + str(ne))
  #     continue
  #   else:
  #     cv_scre_last = cv_scre

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
  print("cross validating model...")
  scores = cross_validation.cross_val_score(clf, X, y, cv=3, scoring='mean_absolute_error')
  return abs(sum(scores) / len(scores))


def cross_validate_model(model, X_train, y_train):

  cvs = 21.5 + cv_score(model, X_train, y_train)
  print("MAE on cross validation set: " + str(cvs))


model = None


def pickle_model(model):
  # pickle model
  with open('./pickled_model/rain2.pickle', 'wb') as f:
    pickle.dump(model, f)
  f.close()
  #joblib.dump(model, './pickled_model/rain2.pkl')


def unpickle_model(file):
  with open('./pickled_model/rain2.pickle', 'rb') as f:
    model = pickle.load(f)
  return model


def split_train_data(train_input, labels, t_size):
  # train_test does shuffle and random splits
  X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    train_input, labels, test_size=t_size, random_state=0)
  return X_train, X_test, y_train, y_test


def test_model(model, X_test, y_test):
  # print("testing on holdout set...")
  pred_y = model.predict(X_test)
  print("MAE on holdout test set", 21.5 + mean_absolute_error(y_test, pred_y))


def feature_selection(model, features):
  # feature engineering
  print("feature selection...")
  print(features)
  print("feature importance", model.feature_importances_)
  min_index = np.argmin(model.feature_importances_)
  print("min feature : ({0}){1}, score {2} ".format(min_index, features[min_index], min(model.feature_importances_)))


def train_model(seed, X_train, y_train, isPickled):
  #clf = evaluate_models(labels, train_input)
  #n_estimators = no. of trees in the forest
  #n_jobs = #no. of cores
  #clf_rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=None, n_jobs=4, min_samples_split=1, random_state=0)
  #extree = ensemble.ExtraTreesRegressor(n_estimators=190, max_depth=25, min_samples_split=10, n_jobs=-1)
  #clf_rf = ensemble.RandomForestRegressor(n_estimators=50, max_depth=None, n_jobs=4, min_samples_split=1,
  #                                        max_features="auto")

  #ne=400, md=10, ss = 50, sl=10, 10 MAE 22.7590658805 and with learning rate = 0.01 MAE 23.5737808932
  #clf_gbt = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=10, min_samples_split=10, min_samples_leaf=10, learning_rate=0.01, loss='ls')
  #clf_gbt = ensemble.GradientBoostingRegressor(n_estimators=40, learning_rate=0.1, max_features =0.3, max_depth=4, min_samples_leaf=3,  loss='lad')

  clf = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_features=0.3, max_depth=4,
                                             min_samples_leaf=3, loss='huber', alpha=0.55, random_state=seed)



  #random forest: 24.53496
  #extra tree   : 24.471939 (4619908812 with all features)
  #gbd          : 24.5456

  #print("cross validating...")
  #cvs = cv_score(clf, X_train, y_train)
  #print("CV score: ", 21.5 + cvs)

  global model

  print("training model...")
  #model = clf.fit(train_input, labels)
  model = clf.fit(X_train, y_train)

  if isPickled:
    print("pickling model...")
    pickle_model(model)



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


  return model


#endregion train

#test
def write_prediction(test_y, test_non_empty_total_ids):
  print("writing to file....")
  predictionDf = pd.DataFrame(index=test_non_empty_total_ids, columns=['Expected'], data=test_y)
  #predict 0 for empty rows/ids
  empty_test_y = np.asanyarray([0 for _ in test_empty_rows_ids])
  emptyRowsDf = pd.DataFrame(index=test_empty_rows_ids, columns=['Expected'], data=empty_test_y)

  totalDf = pd.concat([predictionDf, emptyRowsDf])
  #print(totalDf.head(20))
  totalDf.sort_index(inplace=True)
  #print(totalDf.head(20))

  # write file
  prediction_file = './rain_prediction.csv'
  totalDf.to_csv(prediction_file, index_label='Id', float_format='%.6f')
  print("writing prediction to file Done")


def predict(model, test_input, isPickled):
  print("predicting....")

  #model = joblib.load("./pickled_model/rain2.pkl") if isPickled else model
  model = unpickle_model("./pickled_model/rain2.pickle") if isPickled else model
  if model:
    test_y = model.predict(test_input)
    print(len(test_all_ids) - len(test_empty_rows_ids))
    print(len(test_y))
    return test_y
  else:
    print("no model found..")


def analyze_results():
  df_sample = pd.read_csv('C:/Work/kaggle/how_much_rain2/sample_solution.csv/sample_solution.csv')
  #df_my_res = pd.read_csv('C:/Work/kaggle/how_much_rain2/prediction_aws/extratree/rain_prediction.csv')
  df_my_res = pd.read_csv('C:/Work/kaggle/how_much_rain2/prediction_aws/gdb/rain_prediction.csv')

  #df_my_res = pd.read_csv('C:/Work/kaggle/how_much_rain2/prediction_aws/extratree/rain_prediction_inserted.csv')

  df_both_res = pd.merge(df_sample, df_my_res, how='inner', on=['Id'])
  #print(df_both_res.columns)
  df_id = df_both_res[(df_both_res['Expected_x'] == 0) & (df_both_res['Expected_y'] != 0)]
  print(str(df_id.Id))
  print(len(df_id.Id))

  #write diff to file
  #with open('./analysis/empty_rows.csv','w') as f:
  #  f.writelines(str(df_id.Id))
  #f.close()


  print(len(df_my_res[df_my_res.Expected == 0].Id))

  for id_val in df_id.Id:
    df_my_res.loc[df_my_res.Id == id_val, 'Expected'] = 0

  print(len(df_my_res[df_my_res.Expected == 0].Id))

  prediction_file = 'C:/Work/kaggle/how_much_rain2/prediction_aws/gdb/rain_prediction_inserted.csv'
  df_my_res.to_csv(prediction_file, float_format='%.6f')

  print("analysis complete")

def process_data(data):
  data = standardize_data(data)
  data = impute_data(data)
  return data


def categorize_test_data(X_test, test_y):

  #split test data into 3 subsets based on predicted 0, 1, 2
  test_y_s = pd.Series(data=test_y, index=test_non_empty_ids, name='Class')
  #X_test_df = pd.DataFrame(data=X_test, index=test_non_empty_ids, columns=X_columns)
  test_data = pd.concat([X_test, test_y_s], join='inner', axis=1)

  #ToDo:save indices of the these 3 split
  test_data_light_rain = test_data[test_data['Class'] == 0]
  print('test_light ',str(test_data_light_rain.shape))

  test_data_moderate_rain = test_data[test_data['Class'] == 1]
  print('test_moderate ',str(test_data_moderate_rain.shape))

  test_data_heavy_rain = test_data[test_data['Class'] == 2]
  print('test_heavy ',str(test_data_heavy_rain.shape))

  return test_data_light_rain, test_data_moderate_rain, test_data_heavy_rain

#report
# print("loading & preparing training data...")
#train_file_path = "./train/train.csv"
#train_input, labels = prepare_train_data(train_file_path, True)

#analyze_plot_data(train_input, "training")
#analyze_plot_data(labels, "training")

#for regression
#X_train, X_train_test, y_train, y_train_test = split_train_data(train_input, labels, 0.3)

#evaluate_models(X_train,y_train)

#est_r = ensemble.ExtraTreesRegressor(n_estimators=800, max_depth=4, max_features=1.0, max_leaf_nodes=14, min_samples_leaf=3, n_jobs=-1)


#est_c = ensemble.ExtraTreesClassifier(n_estimators=10, max_features=1.0, max_leaf_nodes=None, min_samples_split=10, min_samples_leaf=5, n_jobs=-1)

#Best GBRT
#est_r = ensemble.GradientBoostingRegressor(n_estimators=600, learning_rate=0.1, max_features=0.3, max_depth=4,
#                                         min_samples_leaf=3, loss='huber', alpha=0.55)

#est_c = ensemble.GradientBoostingClassifier(n_estimators=400, learning_rate=0.1, max_features=0.7, #max_depth=4,
#                                         min_samples_leaf=3, loss='deviance')

#region regressor for light rain
# print("building regression models for training data sets...")
# y_train_data_light_rain = train_data_light_rain['Expected']
# train_data_light_rain = train_data_light_rain.drop(['Class','Expected'], axis=1)
# X_train_data_light_rain = process_data(train_data_light_rain)
#
# X_train_light, X_train_test_light, y_train_light, y_train_test_light = split_train_data(X_train_data_light_rain, y_train_data_light_rain, 0.3)
#
# #cross_validate_model(est_r, X_train_light, y_train_light)
# light_rain_regressor = est_r.fit(X_train_light, y_train_light)
# feature_selection(light_rain_regressor, features)
# #test_model(light_rain_regressor, X_train_test_light, y_train_test_light)
# pred_light_y = light_rain_regressor.predict(X_train_test_light)
# mae_light = mean_absolute_error(y_train_test_light,pred_light_y)
# light_test_cases = len(y_train_test_light)
# print("light rain mae {0} for {1} samples".format(mae_light, light_test_cases))
#
#endregion light rain
#region regressor for moderate rain
# y_train_data_moderate_rain = train_data_moderate_rain['Expected']
# train_data_moderate_rain = train_data_moderate_rain.drop(['Class','Expected'], axis=1)
# X_train_data_moderate_rain = process_data(train_data_moderate_rain)
#
# X_train_moderate, X_train_test_moderate, y_train_moderate, y_train_test_moderate = split_train_data(X_train_data_moderate_rain, y_train_data_moderate_rain, 0.3)
#
# #cross_validate_model(est_r, X_train_moderate, y_train_moderate)
# moderate_rain_regressor = est_r.fit(X_train_moderate, y_train_moderate)
# #feature_selection(moderate_rain_regressor, features)
# #test_model(moderate_rain_regressor, X_train_test_moderate, y_train_test_moderate)
# pred_moderate_y = light_rain_regressor.predict(X_train_test_moderate)
# mae_moderate = mean_absolute_error(y_train_test_moderate, pred_moderate_y)
# moderate_test_cases = len(y_train_test_moderate)
# print("moderate rain mae {0} for {1} samples".format(mae_moderate, moderate_test_cases))
#
#endregion regressor for moderate rain
#region regressor for heavy rain
# y_train_data_heavy_rain = train_data_heavy_rain['Expected']
# train_data_heavy_rain = train_data_heavy_rain.drop(['Class','Expected'], axis=1)
# X_train_data_heavy_rain = process_data(train_data_heavy_rain)
#
# X_train_heavy, X_train_test_heavy, y_train_heavy, y_train_test_heavy = split_train_data(X_train_data_heavy_rain, y_train_data_heavy_rain, 0.3)
#
# #cross_validate_model(est_r, X_train_heavy, y_train_heavy)
# heavy_rain_regressor = est_r.fit(X_train_heavy, y_train_heavy)
# #feature_selection(heavy_rain_regressor, features)
# #test_model(heavy_rain_regressor, X_train_test_heavy, y_train_test_heavy)
#
# pred_heavy_y = light_rain_regressor.predict(X_train_test_heavy)
# mae_heavy = mean_absolute_error(y_train_test_heavy, pred_heavy_y)
# heavy_test_cases = len(y_train_test_heavy)
# print("heavy rain mae {0} for {1} samples".format(mae_heavy, heavy_test_cases))
#
#
# endregion regressor for heavy rain
#
# total_mae = (mae_light * light_test_cases + mae_moderate * moderate_test_cases + mae_heavy * heavy_test_cases ) / (light_test_cases + moderate_test_cases + heavy_test_cases)
#
# print('total MAE'+str(21.5+total_mae))
#
#
# print("building classifier model ...")
# X_train_class, X_train_test_class, y_train_class, y_train_test_class = split_train_data(train_input, category, 0.3)
# scores = cross_validation.cross_val_score(est_c, X_train_class, y_train_class, cv=3, scoring='accuracy')
# cv_scre =  abs(sum(scores) / len(scores))
# print("CV classification score: {0}".format(cv_scre))

#cross_validate_model(est_r, X_train, y_train)
#model = train_model(est_r, X_train, y_train, False)
#feature_selection(model, features)
# #plot_training_curve(model, X_train, y_train)
#test_model(model, X_train_test, y_train_test)
# test_class_predicted_y = model.predict(X_train_test_class)
# print('classifier accuracy score on hold out set: ', accuracy_score(y_train_test_class, test_class_predicted_y))

#test_file_path = "./test/test.csv"
#X_test, X_test_unprocessed = prepare_test_data(test_file_path, True)
#analyze_plot_data(X_test, "test")
#test_y=predict(model, X_test, False)
# print("predicting test class...")
# test_y=predict(model_c, X_test, False)
#write_prediction(test_y, test_non_empty_ids)


#split test data based on classifier
#test_data_light_rain, test_data_moderate_rain, test_data_heavy_rain = categorize_test_data(X_test_unprocessed, test_y)

#predict light rain
# test_data_light_rain.drop(['Class'], axis=1)
# print(test_data_light_rain.shape)
# X_test_data_light_rain = process_data(test_data_light_rain) #standardize and impute
# print(X_test_data_light_rain.shape)
# y_train_data_light_rain = light_rain_regressor.predict(X_test_data_light_rain)
# print(len(y_train_data_light_rain))
# print(type(y_train_data_light_rain))
# print(y_train_data_light_rain)
# #predict moderate rain
# test_data_moderate_rain.drop(['Class'], axis=1)
# X_test_data_moderate_rain = process_data(test_data_moderate_rain) #standardize and impute
# y_train_data_moderate_rain = moderate_rain_regressor.predict(X_test_data_moderate_rain)
# print(len(y_train_data_light_rain))
#
# #predict heavy rain
# test_data_heavy_rain.drop(['Class'], axis=1)
# X_test_data_heavy_rain = process_data(test_data_heavy_rain) #standardize and impute
# y_train_data_heavy_rain = heavy_rain_regressor.predict(X_test_data_heavy_rain)
#
# total_test_id = list(test_data_light_rain.index.values) + list(test_data_moderate_rain.index.values) + list(test_data_heavy_rain.index.values)
#
# print(len(total_test_id))
#
# y_test_total = np.concatenate([y_train_data_light_rain, y_train_data_heavy_rain, y_train_data_moderate_rain])
# print(y_test_total)
# print(len(y_test_total))
# write_prediction(y_test_total, total_test_id)



#analyze_results()

#ToDo: 1. Ensemble GBR and ExtraTree 2. Try creating data set for 10, 20, 30+ mm rain 3. Try stacking models


def begin_train():
  global train_file_path, train_input, labels, X_train, X_train_test, y_train, y_train_test, est_r, model
  train_file_path = "./train/train.csv"
  train_input, labels = prepare_train_data(train_file_path, True)
  X_train, X_train_test, y_train, y_train_test = split_train_data(train_input, labels, 0.3)
  #est_r = ensemble.GradientBoostingRegressor(n_estimators=800, learning_rate=0.1, max_features=0.3, max_depth=4,
  #                                           min_samples_leaf=3, loss='huber', alpha=0.55, random_state=seed)

  # cross_validate_model(est_r, X_train, y_train)

  models = Parallel(n_jobs=-1, backend="threading")(delayed(train_model)(seed, X_train, y_train, False) for seed in range(4))

  for model in models:
    feature_selection(model, features)
    test_model(model, X_train_test, y_train_test)


def begin_test():
  global test_file_path, X_test, X_test_unprocessed, test_y
  test_file_path = "./test/test.csv"
  X_test, X_test_unprocessed = prepare_test_data(test_file_path, True)
  test_y = predict(model, X_test, False)
  write_prediction(test_y, test_non_empty_ids)

begin_train()
#begin_test()

