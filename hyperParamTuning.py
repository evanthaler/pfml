from osgeo import gdal,gdalnumeric
import pandas as pd
import os,glob,pickle
import numpy as np
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV,GridSearchCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF,RationalQuadratic
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
mpl.rcParams.update({'font.size': 16})
os.environ['PROJ_LIB'] = r'C:\Users\361045\Anaconda3\envs\pygeo\Library\share\proj'
os.environ['GDAL_DATA'] = r'C:\Users\361045\Anaconda3\envs\pygeo\Library\share'





def MLHyperParamTuning(wd,outras,trainCSV,modelName,veg,method='ERF',scale=True):
# wd =r'C:\Users\361045\Documents\projects\ngee\machineLearningData\trainingdata'
# outras = r'C:\Users\361045\Documents\projects\ngee\machineLearningData\trainingdata\prediction'+s+'_wCHM.tif'
# trainCSV =r'C:\Users\361045\Documents\projects\ngee\machineLearningData\trainingdata\SewardTrainingData\trainingVals_'+s+'_meanwindow.csv'
# modelname = s+'_wNDVI',
# veg ='ndvi'
# method = 'ERF'
# scale=True
    '''
    

    Parameters
    ----------
    wd : string
        Full path to directory where input rasters are stored
    outras : string
        path and name of output prediction raster
    trainCSV: string
        Full path to csv file with training data. This file will have data from each of the rasters in the wd
    modelName : string
        name of pickeled model. the method string will be appended to the model name
    method : string, optional
        Classification method for prediction. 'ERF' = Extra random forest; 'SVM' = support vector machine; 
        'GPC' = gaussian processes; 'ADA' = adaptive boost 
        The default is 'ERF'.
        THe algorithm abbreviation will be appended to the outras string
    veg = : string
        which veg ifeature to use. 'chm' or 'ndvi'

    predict_full:boolean
        Predict permafrost extent for full raster domain. default=True
    scale: boolean
        Standarize (normalize) data?
    useAllData: boolean
        Use all available data for model training?

    Returns
    -------
    None.

    '''
    os.chdir(wd)
    

    ##############################################
    '''
    Now we need to load in the training data set, which has the same 
    parameters as those in the raster dataframe, but we also have a binary column with
    PF presence/absence.    
    '''
    training_df = pd.read_csv(trainCSV)
    training_df = training_df.dropna()
    target = training_df['PF']
    #print('Full data class ratio:',np.sum(np.array(training_df.PF))/len(np.array(training_df.PF)))
    ##Add a variable that is just random noise---this will help us understand if the model is capturing
    ##physically meaningful correlations
    trnRand = np.random.rand(len(target))
    #training_df['rand']=trnRand
    
    training_data= training_df[['aspect','slope','curv','snow',veg]]
    if scale == True:
        print('Scaling data')
        ##need to standardize data to be format ready for RBF kernel for SVM and gaussian processes
        #scaler = QuantileTransformer(output_distribution='normal')
        scaler = StandardScaler()
        training_data = scaler.fit_transform(training_data.values)
    X_train, X_test, y_train, y_test = train_test_split(training_data, target, test_size=0.20,stratify=target) # use 80% training data
    testacc=[]

        
    if method == 'ERF':
        '''
        We need to generate a parameter grid to sample from, so we can tune the hyperparameters of our models
        First, we will set the grid for the random forest model...these parameters are the ones shown to be the most important
        in influencing predictions
        '''
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.arange(10,1010,9)]
        # Number of features to consider at every split
        max_features = ['sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        
        clf_erf = ExtraTreesClassifier(n_estimators=10,max_features='sqrt')
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = clf_erf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=0, random_state=42, n_jobs = -1)
        # Fit the random search model
        rf_random.fit(X_train, y_train)
        print(rf_random.best_params_)

        
        
        best_random = rf_random.best_estimator_
        best_random.fit(X_train, y_train)
        random_pred = best_random.predict(X_test)
        print(metrics.accuracy_score(y_test,random_pred))
        
        '''
        can further refine the hyperparams using the gridsearch CV
        '''
        # param_grid = {
        # 'bootstrap': [True],
        # 'max_depth': [100,105,110,120],
        # 'max_features': ['sqrt'],
        # 'min_samples_leaf': [2,3,4],
        # 'min_samples_split': [5,10,15],
        # 'n_estimators': [200,300]
        #             }
        
        # grid_search = GridSearchCV(estimator = clf_erf, param_grid = param_grid, 
        #                   cv = 5, n_jobs = -1, verbose = 2)
        # grid_search.fit(X_train, y_train)
        # print(grid_search.best_params_)
        
        # best_grid= grid_search.best_estimator_
        # best_grid.fit(X_train, y_train)
        # grid_pred = best_grid.predict(X_test)
        # print(metrics.accuracy_score(y_test,grid_pred))
        
    elif method == 'SVM':
        #Create a svm Classifier
        clf_svm = svm.SVC(C=100,kernel="linear") 
        
        param_list = {'C': [0.1,1,10,100],
                       'kernel': ['poly','sigmoid','rbf','linear']
                       }
        svm_random = RandomizedSearchCV(estimator = clf_svm, param_distributions = param_list, n_iter = 16, cv = 5, verbose=0, random_state=42, n_jobs = -1)
        svm_random.fit(X_train, y_train) 
        print(svm_random.best_params_)
        
        # #cv_results=cross_validation(clf_svm,training_data,target)
        # clf_svm.fit(X_train, y_train)
        # svm_pred = clf_svm.predict(X_test)
        # print("SVM Accuracy: ",metrics.accuracy_score(y_test,svm_pred))   
        #testacc.append(metrics.accuracy_score(y_test,svm_pred))          
        svm_random_pred = svm_random.predict(X_test)
        print("SVM Accuracy: ",metrics.accuracy_score(y_test,svm_random_pred))  
    

    
    

    
sitelist = ['KG','T27','T47','T27KG','T27T47','T47KG','T27T47KG']
for s in sitelist:
    print('Training with data from:',s)
    MLHyperParamTuning(r'C:\Users\361045\Documents\projects\ngee\machineLearningData\trainingdata',
               r'C:\Users\361045\Documents\projects\ngee\machineLearningData\trainingdata\prediction'+s+'_wCHM.tif',
               r'C:\Users\361045\Documents\projects\ngee\machineLearningData\trainingdata\SewardTrainingData\trainingVals_'+s+'_meanwindow.csv',
               s+'_wNDVI',
               'ndvi','SVM',scale=False)





'''
Parameters
----------
wd : string
    Full path to directory where input rasters are stored
outras : string
    path and name of output prediction raster
trainCSV: string
    Full path to csv file with training data. This file will have data from each of the rasters in the wd
modelName : string
    name of pickeled model. the method string will be appended to the model name
method : string, optional
    Classification method for prediction. 'ERF' = Extra random forest; 'SVM' = support vector machine; 
    'GPC' = gaussian processes; 'ADA' = adaptive boost 
    The default is 'ERF'.
    THe algorithm abbreviation will be appended to the outras string
predict_full:boolean
    Predict permafrost extent for full raster domain. default=True
scale: boolean
    Standardize data? default is True
useAllData: boolean
    Use all training data? Default is true

Returns
-------
None.

'''

