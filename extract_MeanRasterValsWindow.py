from osgeo import gdal
import pandas as pd
import numpy as np
import os,glob,math

def get_index(x: float, y: float, ox: float, oy: float, pw: float, ph: float) -> tuple:
    """
    Convert real world coordinates to array coordinates (i.e. positive integer indices)
    Gets the row (i) and column (j) indices in an NumPy 2D array for a given
    pair of coordinates.

    Parameters
    ----------
    x : float
        x (longitude) coordinate
    y : float
        y (latitude) coordinate
    ox : float
        Raster x origin (minimum x coordinate)
    oy : float
        Raster y origin (maximum y coordinate)
    pw : float
        Raster pixel width
    ph : float
        Raster pixel height

    Returns
    -------
    Two-element tuple with the column and row indices.

    Notes
    -----
    This function is based on: https://gis.stackexchange.com/a/92015/86131.

    Both x and y coordinates must be within the raster boundaries. Otherwise,
    the index will not correspond to the actual values or will be out of
    bounds.
    """
    # make sure pixel height is positive
    ph = abs(ph)

    i = math.ceil((oy-y) / ph)
    j = math.ceil((x-ox) / pw)

    return i, j


def getValuesinWindow(pFile,wd,outFile,value='PF'):
    '''
    

    Parameters
    ----------
    pFile : string
        path to csv file with x,y,value (default value is 'PF')
        
    wd : string
        path to directory with raster files
    outFile : string
        path to output csv file location
    value : string
        value at x,y location. default is 'PF'
    Returns
    -------
    None.

    '''
    posFile = pd.read_csv(pFile)
    x=posFile.X;y=posFile.Y;pf=posFile[value]
    os.chdir(wd)
    rasList=glob.glob('*.tif')
    xc=[]
    yc=[]
    df=pd.DataFrame()
    for ras in np.arange(0,len(rasList)):
        v=[];pfval=[]
        raster=rasList[ras]
        rasname = raster[:-4]
    
    
        ds = gdal.Open(raster)  
        ox, pw, xskew, oy, yskew, ph = ds.GetGeoTransform()
        '''
        For explanation of GetGeoTransform, see https://gdal.org/tutorials/geotransforms_tut.html
        ox: x-coordinate of the upp-left corner of the upper-left pixel
        pw: w-e pixel resolution (pixel width)
        xskew: row rotation (typically zero)
        oy: y-coordinate of the upper-left corner of the upper-left pixel
        yskew: column rotation (typically zero)
        ph: n-s pixel resolution (pixel height) (negative for a north-up image)
        '''
        
        nd_value = ds.GetRasterBand(1).GetNoDataValue()
        arr = ds.ReadAsArray()
        del ds
        
        
        
        #Generate padding--this will keep the extract value in the center of a 3x3 window
        padding_y = (1, 1)  # 1 rows above and 1 rows below
        padding_x = (1, 1)  # 1 columns to the left and 1 columns to the right
        padded_arr = np.pad(arr, pad_width=(padding_y, padding_x), mode='constant', constant_values=nd_value)
        
        '''
        For explanation of np.pad, see https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        pad_width:{sequence, array_like, int}
            Number of values padded to the edges of each axis. ((before_1, after_1), … (before_N, after_N)) unique pad widths for each axis. ((before, after),) yields same before and after pad for each axis. (pad,) or int is a shortcut for before = after = pad width for all axes.
        a = [1, 2, 3, 4, 5]
        np.pad(a, (2, 3), 'constant', constant_values=(4, 6))
        array([4, 4, 1, ..., 6, 6, 6])
        '''
        
        
        
        offset = 1
        '''
        offset: number of cells to extract on each side of the center pixel
        '''
        
        count=0
        #Now we loop through the list of x,y values in the original dataset and will get the coordinates and raster values of the 8 cells surrounding the original point value
        for x_coord, y_coord in zip(x, y):
            # get index. i = y coord values, j= x coord values
            i, j = get_index(x_coord, y_coord, ox, oy, pw, ph)
            #upper left 
            ul=(x_coord-pw,y_coord+pw)  #values.flatten(0)
            #upper center 
            uc=(x_coord,y_coord+pw) #values.flatten(1)
            #upper right 
            ur=(x_coord+pw,y_coord+pw) #values.flatten(2)
            #left 
            l=(x_coord-pw,y_coord) #values.flatten(3)
            #center
            cent=(x_coord,y_coord) #values.flatten(4)
            #right
            r=(x_coord+pw,y_coord) #values.flatten(5)
            #lower left 
            ll=(x_coord-pw,y_coord-pw) #values.flatten(6)
            #lower center 
            lc=(x_coord,y_coord-pw) #values.flatten(7)    
            #lower right 
            lr=(x_coord+pw,y_coord-pw) #values.flatten(8)
            c=(ul,uc,ur,l,cent,r,ll,lc,lr)
            
            # get pixel value and its 8 neighbours
            values = padded_arr[i-offset:i+offset+1, j-offset:j+offset+1]
            values=values.flatten()
            v.append(np.mean(values))
            pfval.append(pf[count])
            if ras ==0:
                xc.append(cent[0]);yc.append(cent[1])
            
            count+=1
        if ras ==0:
            permvals=np.array(pfval)
        df[rasname]=np.array(v)
    df['X']=np.array(xc);df['Y']=np.array(yc);df['PF']=permvals
    df.to_csv(outFile)

getValuesinWindow(r'C:\Users\361045\Documents\projects\ngee\machineLearningData\sebData\GroundTruthLocations\Teller27_points_trimmed.csv',
                  r'C:\Users\361045\Documents\projects\ngee\machineLearningData\t27\3mRasters',
                  r'C:\Users\361045\Documents\projects\ngee\machineLearningData\trainingdata\SewardTrainingData\trainingVals_T27_meanwindow.csv')




    
    