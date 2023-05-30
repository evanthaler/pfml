from osgeo import gdal,gdalnumeric
import pandas as pd
import os,glob,pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix

os.environ['PROJ_LIB'] = r'C:\Users\361045\Anaconda3\envs\pygeo\Library\share\proj'
os.environ['GDAL_DATA'] = r'C:\Users\361045\Anaconda3\envs\pygeo\Library\share'


def runANN(trainingData,wd,output,modelName,predict_full=True):
    '''
    

    Parameters
    ----------
    trainingData : string
        path to csv file with training data
    wd : string
        path to directory with tif files for prediction
    output : string
        path to output prediction raster
    modelName : string
        specific model name to add to saved model string

    Returns
    -------
    None
        

    '''
    os.chdir(wd)
    flist = glob.glob('*.tif')
    
    if os.path.exists('./PF_predictions'):
        print('PF prediction directory exists')
        
    else:
        print('gotta make a new directory for PF prediction files')
        os.mkdir('./PF_predictions')
    
    newRasterfn1 = './PF_predictions//'+output
    
    
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
    
    

    
    ###
    '''
    First we will read in the rasters of the full study location
    We will extract the data from each raster into a 1-D array and set the nodata values to np.nan
    '''
    aspect=gdalnumeric.LoadFile(glob.glob('*aspect*.tif')[0]).astype('float32').flatten();aspect[aspect==nodat]=np.nan
    snow=gdalnumeric.LoadFile(glob.glob('*snow*.tif')[0]).astype('float32').flatten();snow[np.isnan(aspect)]=np.nan
    #chm = gdalnumeric.LoadFile(glob.glob('*chm*.tif')[0]).astype('float32').flatten();chm[np.isnan(aspect)]=np.nan
    
    elev = gdalnumeric.LoadFile(glob.glob('*elev*.tif')[0]).astype('float32').flatten();elev[np.isnan(aspect)]=np.nan
    slope=gdalnumeric.LoadFile(glob.glob('*slope*.tif')[0]).astype('float32').flatten();slope[np.isnan(aspect)]=np.nan
    curv = gdalnumeric.LoadFile(glob.glob('*curv*.tif')[0]).astype('float32').flatten();curv[np.isnan(aspect)]=np.nan
    ndvi = gdalnumeric.LoadFile(glob.glob('*ndvi*.tif')[0]).astype('float32').flatten(); ndvi[np.isnan(aspect)]=np.nan
    
    
    '''
    Now we need to combine the raster data into a dataframe
    We will be later use the model to predict PF extent based on these data
    '''
    df_fullData = pd.DataFrame({'aspect':aspect,'elev':elev,'slope':slope,'curv':curv,'snow':snow,'ndvi':ndvi})
    df_fullDataNoDat=df_fullData.dropna()
    
    X_FULL=df_fullDataNoDat[['aspect','slope','curv','snow','ndvi']]
    
    #standardize the input features because each input is at a different scale
    sc=StandardScaler()
    X_FULL=sc.fit_transform(X_FULL)
    ##############################################
    '''
    Now we need to load in the training data set, which has the same 
    parameters as those in the raster dataframe, but we also have a binary column with
    PF presence/absence. 
    '''
    training_df = pd.read_csv(trainingData)
    target = training_df['PF']
    #training_data= training_df[['chm','aspect','elev','tpi','slope','tri','curv','da','snow']]
    training_data= training_df[['aspect','slope','curv','snow','ndvi']]
    
    
    
    '''
    Now we will train and test the model using the training and target datasets
    '''
    
    # MLP with manual validation set
    
    
    # fix random seed for reproducibility
    seed = 5
    np.random.seed(seed)
    
    # split into input (X) and output (Y) variables
    X = training_data
    #standardize the input features because each input is at a different scale
    sc=StandardScaler()
    X=sc.fit_transform(X)
    Y = target
    # split into 80% for train and 20% for test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    
    
    
    classifier=Sequential()
    #Define input later and first hidden layer
    classifier.add(Dense(16,activation='relu',kernel_initializer='random_normal',input_dim = (len(training_data.columns),)))
    #Second hidden layer
    classifier.add(Dense(16,activation='relu',kernel_initializer='random_normal'))
    #Output layer
    classifier.add(Dense(1,activation='sigmoid',kernel_initializer='random_normal'))
    #compile the neural network
    classifier.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
    classifier.fit(X_train,y_train,batch_size=int(len(training_df.aspect)/100),epochs=1000)
    
    loss,accuracy=classifier.evaluate(X_train, y_train)
    print('loss: ',loss, 'accuracy: ',accuracy)
    
    
 
    
    y_pred = classifier.predict(X_test)
    y_pred=(y_pred>0.5)
    cm = confusion_matrix(y_test, y_pred)
    tp= cm[0,0];tn=cm[1,1]
    test_acc = (tp+tn)/(sum(cm.flatten()))
    print('accuracy for test points: ', test_acc)
    
    
    
    # pickle the model to disk
    if os.path.exists('./pickledModels'):
        print('pickling that special model')
        modelpickle = './pickledModels/TrainedmodelANN'+modelName
        classifier.save(modelpickle)
    else:
        print('gotta make a new directory, but then I am pickling that special model')
        os.mkdir('./pickledModels')
        modelpickle = './pickledModels/TrainedmodelANN'+modelName
        classifier.save(modelpickle)
        
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
    
    if predict_full==True:
        ann_pred = classifier.predict(X_FULL)
    #need to reshape prediction array to match the raster domain
        y_fullpredResize = add_nans(ann_pred,aspect)
        ##Here we set a thredhold of 0.5 to generate a binary classification map
        y_fullpredResize[y_fullpredResize<0.5]=0
        y_fullpredResize[y_fullpredResize>=0.5]=1
        
        cnn_pred=y_fullpredResize.reshape(bandarr.shape)
        
        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(newRasterfn1, cols, rows, 1, gdal.GDT_Float32,["COMPRESS=LZW"])
        outband=outRaster.GetRasterBand(1)
        
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        outRaster.SetGeoTransform(geotransform)
        outRaster.SetProjection(tif.GetProjection())
        outband = outRaster.GetRasterBand(1)
        
        outband.SetNoDataValue(-9999)
        outband.WriteArray(cnn_pred)
        
        
        outband.FlushCache()
    else:
        print('not predicting for raster')
    outRaster=None
    tif=None
    
t27=r'C:\Users\361045\Documents\projects\ngee\machineLearningData\t27\3mRasters'
kg=r'C:\Users\361045\Documents\projects\ngee\machineLearningData\kg\3mRasters'
t47=r'C:\Users\361045\Documents\projects\ngee\machineLearningData\t47\3mRasters'   

runANN(r'C:\Users\361045\Documents\projects\ngee\machineLearningData\trainingdata\NDVItrainingData\trainingVals_T47KG.csv',
       kg,
       'kg_ANN.tif','t47kg',predict_full=False)


'''


Parameters
----------
trainingData : string
    path to csv file with training data
wd : string
    path to directory with tif files for prediction
output : string
    path to output prediction raster
modelName : string
    specific model name to add to saved model string

Returns
-------
None
    

'''