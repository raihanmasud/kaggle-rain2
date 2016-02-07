# kaggle-rain2
<b>https://www.kaggle.com/c/how-much-did-it-rain-ii
<pre>
Kaggle Rain2
Raihan Masud

Data Exploration:
•	Loaded data in pandas data frame. Couldn’t load the 13million records on 8GB machine so, loaded 1065201 examples in local machine. (loaded all in EC2 g2.2xlarge GPU with 8 cpu cores)
•	Looked at data in excel/csv to get a feel of feature, missing values, data types, ranges etc
•	Printed statistics of the data of different features/columns in pandas.dataframe.describe() – max, min, mean, percentiles etc
•	Plotted data distribution histograms of 20 features(5 main types * 4 percentiles) to understand data distribution 
•	     
 
Data Preparation: 
•	Removed noisy data – rain mm>50 (heavy rain) assumed to be useless.
•	Removed missing values of ‘Radar Reflectivity’ with no data(NaN) for all radar scan of same ID
•	Standardize data – 0 mean and unit standard deviation
•	Imputed missing values(NaN) with average initially, later based on knowledge of the features Kdp, Zdr with relation to Ref values used appropriate values instead of average, gave little boost 0.002 but did not use later as it was slow to iterate over 13M records
Features:
•	At first used all given features first that made sense, removed minutes_past, id, less relevant to rain amount.
•	Tried converting some features to normal distribute or log normal and shifted, later didn’t use it as it didn’t relate to the prediction target y
•	Created meta featues like MIN, MAX, MEAN, MADEAN, STD, etc of original features – about 40 features
•	Later removed some feature based on feature importance calculated after creating model

Model Selection:
•	As a regression problem, picked up several models – RF, Extra Tree, GBR, LinearRegression, Lasso, Ridge, ElasticNet etc and ran quickly and saw the performances  and picked RF, ExTree and GBR for through evaluations
•	Splited data on 70% training 30% test set
•	Performed CV with k=5 fold later k=3 fold as it was taking time
•	Plotted learning curve to see Bias vs Varaince, as enough data and features were there didn’t suffer from bias, focused on lowering variance
•	Local Mean Absolute Error(MAE) was not reflecting LeaderBoard MAE (due to noisy data was there and had to predict values for those) so added a constant to have a similar match(just for mental mapping) 
•	GBR performed better on the LeaderBoard score on Kaggle but ExtraTree was better locally, so finally picked GBR assuming it’s more generalized with less variance
•	Started parameter tuning on RF, ExTree and GBR – used GridSearchCV and  brute force changes
•	Also plotted validation curves over parameters for picking up right values with just enough regularization like shrinkage learning rate 0.1, n_estimators=500-800 and tree structures min_leaf_samples, max_depth, and stocastics gradient boosting max_features 0.3 for randomization, etc
     
•	Played the loss function = initially sklearn ls, the Huber, tried also quantile but Huber was best performing
•	Final Model GBM with ensembles of 8 GBMs with different seeds:
•	est_r = ensemble.GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_features=0.3, max_depth=4,
                                           min_samples_leaf=3, loss='huber', alpha=0.55)
note: random_state= 0 - 7(seed) was used for parallelizing multiple models on different cores
What didn’t work:
•	Splitting data into 4 sets based on rain in mm (4 classes - <5mm, 15, 25, >40mm), and train 4 different GBRegressors and a Classfier to Classify the class on training set and then split the test set into 4 subsets and use the 4 GBRegressors to predict rain amount and merge them
•	Later did the similar to above bullet, but based on Radar Reflectivity <5dBz, 20, 35 >50dBZ, it was producing better local CV, test score but didn’t do well in LeaderBoard 


 
Future : 
Do GridSearchCV in joblib parallel to save time
Lesson Learned: GBRegressor needs careful parameters tuning.
max_depth = 4-6max, try 5
n_estimators: 500 -1000 learning rate: 0.1 or less
max_features: < 0.2 of total_number of features if enough features (~50 or more)
max_samples_leaf: 3 or more (5) or try grid search
GBRegression training slowest, but testing is faster.
</pre>
