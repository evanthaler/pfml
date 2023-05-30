from osgeo import gdal,gdalnumeric
import pandas as pd
import os,glob,pickle
import numpy as np
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split,cross_val_score
#from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import ExtraTreesClassifier #RandomForestClassifier,AdaBoostClassifier
#from sklearn.gaussian_process.kernels import ConstantKernel, RBF,RationalQuadratic
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
mpl.rcParams.update({'font.size': 16})
os.environ['PROJ_LIB'] = r'C:\Users\361045\Anaconda3\envs\pygeo\Library\share\proj'
os.environ['GDAL_DATA'] = r'C:\Users\361045\Anaconda3\envs\pygeo\Library\share'

def MLClassify(wd,outras,trainCSV,modelName,veg,method='ERF',predict_full=True,scale=True,useAllData=True):
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
        ##These have been removed:'GPC' = gaussian processes; 'ADA' = adaptive boost 
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
    flist = glob.glob('*.tif')
    
    if os.path.exists('./PF_predictions'):
        print('prediction directory exists')
        
    else:
        print('gotta make a new directory for prediction files')
        os.mkdir('./PF_predictions')
        
    newRasterfn1 = outras[:-4]+'_'+method+'.tif'
    
    '''
    Below are a lot of GDAL functions for getting the nodata value from the input rasters
    and generating a new output raster with predicted PF values. 
    '''
    
    tif = gdal.Open(flist[0])
    #get image metadata
    band = tif.GetRasterBand(1)
    bandarr = band.ReadAsArray()
    nodat = band.GetNoDataValue()
    geotransform = tif.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = tif.RasterXSize
    rows = tif.RasterYSize
    
    ##need to standardize data to be format ready for RBF kernel for SVM and gaussian processes
    scaler = StandardScaler()

    
    ###
    '''
    First we will read in the rasters of the full study location
    We will extract the data from each raster into a 1-D array and set the nodata values to np.nan
    '''
    aspect=gdalnumeric.LoadFile(glob.glob('*aspect*.tif')[0]).astype('float32').flatten();aspect[aspect==nodat]=np.nan
    #aspect =scaler.fit_transform(aspect)
    snow=gdalnumeric.LoadFile(glob.glob('*snow*.tif')[0]).astype('float32').flatten();snow[np.isnan(aspect)]=np.nan
    #snow =StandardScaler([snow])
    chm = gdalnumeric.LoadFile(glob.glob('*chm*.tif')[0]).astype('float32').flatten();chm[np.isnan(aspect)]=np.nan
    #chm =StandardScaler([chm])
    elev = gdalnumeric.LoadFile(glob.glob('*elev*.tif')[0]).astype('float32').flatten();elev[np.isnan(aspect)]=np.nan
    #elev =StandardScaler([elev])
    slope=gdalnumeric.LoadFile(glob.glob('*slope*.tif')[0]).astype('float32').flatten();slope[np.isnan(aspect)]=np.nan
    #slope =StandardScaler([slope])
    curv = gdalnumeric.LoadFile(glob.glob('*curv*.tif')[0]).astype('float32').flatten();curv[np.isnan(aspect)]=np.nan
    #curv =StandardScaler([curv])
    ndvi = gdalnumeric.LoadFile(glob.glob('*ndvi*.tif')[0]).astype('float32').flatten(); ndvi[np.isnan(aspect)]=np.nan
    #ndvi =StandardScaler([ndvi])
    ##Add a variable that is just random noise---this will help us understand if the model is capturing
    ##physically meaningful correlations
    #rand = np.random.rand(len(ndvi))
    
    
    '''
    Now we need to combine the raster data into a dataframe
    We will later use the model to predict PF extent based on these data
    '''
    df_fullData = pd.DataFrame({'aspect':aspect,'elev':elev,'slope':slope,'curv':curv,'snow':snow,'ndvi':ndvi,'chm':chm})
    df_fullDataNoDat=df_fullData.dropna()

    
    X_FULL=df_fullDataNoDat[['aspect','slope','curv','snow',veg]]
    if scale == True:
        X_FULL = scaler.fit_transform(X_FULL)
    ##############################################
    '''
    Now we need to load in the training data set, which has the same 
    parameters as those in the raster dataframe, but we also have a binary column with
    PF presence/absence.    
    '''
    training_df = pd.read_csv(trainCSV)
    training_df = training_df.dropna()
    target = training_df['PF']
    ##Add a variable that is just random noise---this will help us understand if the model is capturing
    ##physically meaningful correlations
    trnRand = np.random.rand(len(target))
    training_df['rand']=trnRand
    
    training_data= training_df[['aspect','slope','curv','snow',veg]]
    trd = training_df[['aspect','slope','curv','snow',veg]]
    if scale == True:
        print('Scaling data')
        training_data = scaler.fit_transform(training_data.values)
    if useAllData==True:
        print('Using all available training data')
        #X_train, X_test, y_train, y_test = train_test_split(training_data, target, test_size=0.01) # use all training data
        X_train,y_train  =training_data, target
        X_test=X_train;y_test=y_train
    else:
        X_train, X_test, y_train, y_test = train_test_split(training_data, target, test_size=0.20) # use 80% training data
    '''
    Now we will train and test the model using the training and target datasets
    '''
    
    '''
    Now let's use the model to predict PF using the full raster dataset
    change the model before .predict(X_FULL) to use different ML model
    '''
    if method == 'ERF':
        clf_erf = ExtraTreesClassifier(n_estimators=10,max_features='sqrt')
        clf_erf.fit(X_train, y_train)
        #y_pred_erf=clf_erf.predict(X_test)
        #print("ExtraRF Accuracy: ",metrics.accuracy_score(y_test,y_pred_erf))
        #feature_imp = pd.Series(clf_erf.feature_importances_,index=training_data.columns).sort_values(ascending=False)
        if veg == 'ndvi':
            vegname = 'NDVI'
        elif veg == 'chm':
            vegname = 'Canopy height'
        featNames = ['Aspect','Slope','Curvature','Snow',vegname]
        
        #objects = list(trd.columns)
        
        #Impurity based importance
        importances = clf_erf.feature_importances_
        forest_importances = pd.Series(importances, index=featNames)
        std = np.std([tree.feature_importances_ for tree in clf_erf.estimators_], axis=0)
        
        fig, ax = plt.subplots(figsize=(6.5,6.5))
        forest_importances.plot.bar(yerr=std, ax=ax)
        #ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Impurity feature importance")
        plt.xticks(rotation=45)
        fig.tight_layout()
        plt.savefig(wd+r'\Figs\\'+modelName+'FulldatafeatureImportance.jpg',dpi=600)
        plt.show()
        
        
        
        ##Permutation based importance
        result = permutation_importance(
            clf_erf, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
        )
        
        forest_importances = pd.Series(result.importances_mean, index=featNames)
        fig, ax = plt.subplots(figsize=(6.5,6.5))
        forest_importances.plot.bar(yerr=result.importances_std, ax=ax,color='darkblue')
        #ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Permutation feature importance")
        plt.xticks(rotation=45)
        plt.ylim(bottom=0)
        fig.tight_layout()
        plt.savefig(wd+r'\Figs\\'+modelName+'FulldatafeatureImportanceFullpermutation.jpg',dpi=600)
        plt.show()
        classifier = clf_erf
    elif method == 'SVM':
        #Create a svm Classifier
        clf_svm = svm.SVC(kernel="linear") # radial basis function Kernel
        #Train the model using the training sets
        clf_svm.fit(X_train, y_train)
        svm_pred = clf_svm.predict(X_test)
        if veg == 'ndvi':
            vegname = 'NDVI'
        elif veg == 'chm':
            vegname = 'Canopy height'
        featNames = ['Aspect','Slope','Curvature','Snow',vegname]
        coeffs = np.abs(clf_svm.coef_)
        plt.figure(figsize=(6.5,6.5))
        plt.bar(np.arange(len(featNames)),coeffs[0],align='center')
        plt.xticks(np.arange(len(featNames)), featNames,rotation=45)
        plt.ylabel('Relative importance')
        plt.tight_layout()
        plt.savefig(wd+r'\Figs\\'+modelName+'FulldatafeatureImportance_SVM.jpg',dpi=600)
        plt.show()
        classifier = clf_svm







    

        
    # pickle the model to disk
    if os.path.exists('./pickledModels'):
        print('pickling that special model')
        modelpickle = './pickledModels/TrainedOnFullData//'+method+'_'+modelName+'_Trainedmodel.pkl'
        pickle.dump(classifier, open(modelpickle, 'wb'))
    else:
        print('gotta make a new directory, but then I am pickling that special model')
        os.mkdir('./pickledModels')
        modelpickle = './pickledModels/TrainedOnFullData/'+method+'_'+modelName+'_Trainedmodel.pkl'
        pickle.dump(classifier, open(modelpickle, 'wb'))

    if predict_full==True:
        #Run the trained model on the full given domain        
        y_full_pred=classifier.predict(X_FULL)
    
        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(newRasterfn1, cols, rows, 1, gdal.GDT_Float32,["COMPRESS=LZW"])
        outband=outRaster.GetRasterBand(1)
        
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        outRaster.SetGeoTransform(geotransform)
        outRaster.SetProjection(tif.GetProjection())
        outband = outRaster.GetRasterBand(1)
        '''
        Now we can write the prediction raster to a new tif
        First we need to fill in the places in y_full_pred where we removed nans from the original input dems, which were compiled
        into df_fullData. we'll just be adding NaNs where the NaNs were in the original input data.
        '''
        
        
        def add_nans(x,y):
            lst = []
            index = 0
            for val in y:
                if np.isnan(val):
                    lst.append(np.nan)
                else:
                    lst.append(x[index])
                    index +=1
        
            return np.array(lst)
        
        #need to reshape prediction array to match the raster domain
        y_fullpredResize = add_nans(y_full_pred,aspect)
        y_full_pred=y_fullpredResize.reshape(bandarr.shape)
        
        
        outband.SetNoDataValue(-9999)
        outband.WriteArray(y_full_pred)
        
        
        outband.FlushCache()
    else:
        print('not predicting on full raster')
    outRaster=None
    tif = None
    
sitelist = ['KG','T27KG','T27','T47KG','T47','T27T47','T27T47KG']
for s in sitelist:
    print('Training with data from:',s)
    MLClassify(r'C:\Users\361045\Documents\projects\ngee\machineLearningData\trainingdata',
               r'C:\Users\361045\Documents\projects\ngee\machineLearningData\trainingdata\prediction'+s+'_wCHM.tif',
               r'C:\Users\361045\Documents\projects\ngee\machineLearningData\trainingdata\SewardTrainingData\trainingVals_'+s+'_meanwindow.csv',
               s+'_wCHM',
               'chm','ERF',predict_full=False,scale=True,useAllData=True)





'''
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
        ##These have been removed:'GPC' = gaussian processes; 'ADA' = adaptive boost 
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


'''

