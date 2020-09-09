# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:40:04 2020

@author: Guillaume Leclerc
@email: guillaume.leclerc3@usherbrooke.ca

"""

import os
import ee
import gdal, ogr, osr
import json
import datetime
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import wiener
from scipy import stats
import requests
import matplotlib.pyplot as plt
import cv2
from xml.dom import minidom
import timeit
import zipfile
import sys
import shutil


def readFeature(inputFeature):
    '''
    Fonction permettant de lire un fichier de formes sous divers formats.
    
    inputFeature: chemin d'accès vers le fichier.
    
    '''
        
    fn = r'%s' % (inputFeature)
    ds = ogr.Open(fn,0)
    
    lyr = ds.GetLayer()
    
    outdriver = ogr.GetDriverByName('MEMORY')
    source = outdriver.CreateDataSource('memData')

    source.CopyLayer(lyr,'feature')
    
    return source
    
def getProjection(layer):
    '''
    Fonction permettant d'obtenir le code SRID de projection d'un fichier de 
    formes.
    
    layer: layer du fichier de formes.
    
    '''
     
    inProj = layer.GetSpatialRef()
    
    # Détecter le code EPSG/SRID
    srid = inProj.GetAuthorityCode(None)
        
    return int(srid)

def reprojeter(geom, inProj, outProj):
    '''
    Fonction permettant de reprojeter une géométrie sous le format WKT.
    
    geom: géométrie sous forme WKT
    inProj: projection d'origine (code SRID ex.: 4326)
    outProj: projection en sortie (code SRID ex.: 4326)
    '''
    
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inProj)
    # Nécessaire pour garder l'ordre traditionnel des coordonnée.
    inSpatialRef.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    
    # Output SpatialReference
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outProj)
    # Nécessaire pour garder l'ordre traditionnel des coordonnée.
    outSpatialRef.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)


    # Calculer la transformation entre les deux systèmes de coordonnées
    transform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
        
    for i in geom:

        # Reprojection de la géométrie
        poly = ogr.CreateGeometryFromWkt(str(i))
        poly.Transform(transform)
        
        wkt = poly.ExportToWkt()
        
        return wkt
    

def getStudyAreaCoordinates(geom):
    '''
    Extraction des coordonnées d'un fichier de formes. Cette fonction fournie
    les coordonnées sous forme de geojson
    
    geom: géométrie en intrant sous forme WKT.
    
    '''
        
    wkt = ogr.CreateGeometryFromWkt(str(geom))
    
    geojson = wkt.ExportToJson()
    
    x = json.loads(geojson)
    
    return x["coordinates"]
    


def getLatLongImg(img):
    '''
    Exporter la latitude, la longitude et la valeur du pixel prêt à être introduite dans une matrice NumPy.
    '''
    
    img = img.addBands(ee.Image.pixelLonLat())
 
    img = img.reduceRegion(reducer=ee.Reducer.toList(),\
                                        geometry=area,\
                                        maxPixels=1e13,\
                                        scale=15);
 
    data = np.array((ee.Array(img.get("result")).getInfo()))
    lats = np.array((ee.Array(img.get("latitude")).getInfo()))
    lons = np.array((ee.Array(img.get("longitude")).getInfo()))
    
    return lats, lons, data


def imgToArray(img,capteur,zoneEtude):
    '''
    Converti la latitude, longitude et la valeur du pixel dans une matrice Numpy.
    Les variables proviennent de la fonction ''getLatLongImg(img)''.
    '''
    
    latlon = ee.Image.pixelLonLat().addBands(img)
    
    latlon_new = latlon.reduceRegion(
    reducer=ee.Reducer.toList(),
    geometry=zoneEtude,
    maxPixels=1e10,
    scale=15)
    
    if capteur=='L8':
        
        B = np.array((ee.Array(latlon_new.get("B2")).getInfo()))
        G = np.array((ee.Array(latlon_new.get("B3")).getInfo()))
        R = np.array((ee.Array(latlon_new.get("B4")).getInfo()))
        N = np.array((ee.Array(latlon_new.get("B5")).getInfo()))
        S1 = np.array((ee.Array(latlon_new.get("B6")).getInfo()))
        S2 = np.array((ee.Array(latlon_new.get("B7")).getInfo()))
        
        lats = np.array((ee.Array(latlon_new.get("latitude")).getInfo()))
        lons = np.array((ee.Array(latlon_new.get("longitude")).getInfo()))
        
        # print(B.shape,G.shape,R.shape,lats.shape,lons.shape)
        # print(B.dtype.name,G.dtype.name,R.dtype.name,N.dtype.name,lats.dtype.name,lons.dtype.name)
        
        img_np = np.stack((R,G,B,N,S1,S2))
        
        uniqueLats = np.unique(lats)
        uniqueLons = np.unique(lons)
        
        ncols = len(uniqueLons)    
        nrows = len(uniqueLats)
        
        ys = uniqueLats[1] - uniqueLats[0] 
        xs = uniqueLons[1] - uniqueLons[0]
        
        arr = np.zeros([nrows, ncols,len(img_np)], np.float32)
        
        for z in range(0,len(arr[0][0]),1):
          counter =0
          for y in range(0,len(arr),1):
           for x in range(0,len(arr[0]),1):
                if lats[counter] == uniqueLats[y] and lons[counter] == uniqueLons[x] and counter < len(lats)-1:
                       counter+=1
                       arr[len(uniqueLats)-1-y,x,z] = img_np[z,counter]
        return arr

    if capteur=='S2':
        
        B = np.array((ee.Array(latlon_new.get("B2")).getInfo()))
        G = np.array((ee.Array(latlon_new.get("B3")).getInfo()))
        N = np.array((ee.Array(latlon_new.get("B8")).getInfo()))
        
        lats = np.array((ee.Array(latlon_new.get("latitude")).getInfo()))
        lons = np.array((ee.Array(latlon_new.get("longitude")).getInfo()))
        
        # print(B.shape,G.shape,N.shape,lats.shape,lons.shape)
        # print(B.dtype.name,G.dtype.name,N.dtype.name,lats.dtype.name,lons.dtype.name)
        
        img_np = np.vstack((B,G,N))
        
        uniqueLats = np.unique(lats)
        uniqueLons = np.unique(lons)
        
        ncols = len(uniqueLons)    
        nrows = len(uniqueLats)
        
        ys = uniqueLats[1] - uniqueLats[0] 
        xs = uniqueLons[1] - uniqueLons[0]
        
        arr = np.zeros([nrows, ncols,len(img_np)], np.float32)
        
        for z in range(0,len(arr[0][0]),1):
          counter =0
          for y in range(0,len(arr),1):
           for x in range(0,len(arr[0]),1):
                if lats[counter] == uniqueLats[y] and lons[counter] == uniqueLons[x] and counter < len(lats)-1:
                       counter+=1
                       arr[len(uniqueLats)-1-y,x,z] = img_np[z,counter]
        return arr, uniqueLats, uniqueLons, ys, xs, ncols, nrows
    

def array2img(array,uniqueLats,uniqueLons,pixelWidth,pixelHeight,ncols,nrows,img_name, bands):
    '''
    '''
    # Appliquer le geotransform
    #SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    transform = (np.min(uniqueLons),pixelWidth,0,np.max(uniqueLats),0,-pixelHeight)
    
    # set the coordinate system
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    
    # set driver
    driver = gdal.GetDriverByName('GTIFF')

    count = -1    

    for b in range(array.shape[2]):     
       
        count += 1
        
        outputDataset = driver.Create("%s_%s.tif" %(img_name,bands[count]),ncols,nrows,1,gdal.GDT_Float32)
        
        outputDataset.SetGeoTransform(transform)
        
        outband = outputDataset.GetRasterBand(1)
        
        outband.WriteArray(array[:,:,b])
        
        outputDataset.SetProjection(target.ExportToWkt())
        
        # Read in the band's data into the third dimension of our array
        # outputDataset.WriteArray(b)
        
        # outband = outputDataset.GetRasterBand(b + 1)
        # outputDataset.SetNoDataValue(-9999)
        
        print("Image %s saved" % ("%s_%s.tif" %(img_name,bands[count])))
        
        outband.FlushCache()
        
        outputDataset = None

        # break


    # timestring = time.strftime("%Y%m%d_%H%M%S")
    # outputDataset = driver.Create("test_S2.tif", ncols,nrows, 1,gdal.GDT_Float32)
 
    # # # add some metadata
    # # outputDataset.SetMetadata( {'someotherInfo': 'lala'} )
 
    # outputDataset.SetGeoTransform(transform)
    # outputDataset.SetProjection(target.ExportToWkt())
    # outputDataset.GetRasterBand(3).WriteArray(array)
    # outputDataset.GetRasterBand(3).SetNoDataValue(-9999)
    # outputDataset = None

# https://stackoverflow.com/questions/6791233/gdal-writearray-issue
# https://gis.stackexchange.com/questions/293540/writing-a-raster-file-from-multi-dimension-array-using-python

def download_S2(imgName,filepath,username,password):
    
    """
    Fonction permettant d'extraire le ID des images S2 à télécharger' et d'extraire la localisation des
    évènements se produisant sur un territoire donné.
    
    imgName: Nom de l'image Sentinel 2. ex.: 'S1A_IW_SLC__1SDV_20141101T165548_20141101T165616_003091_0038AA_558F'.
    
    username: Nom d'utilisateur pour se connecter à Scihub (serveur contenant les images S2.
    
    password: Mot de passe pour se connecter à Scihub
 
    """

    server = "https://scihub.copernicus.eu/apihub/odata/v1/Products"
    
    imgQuery = "?$filter=Name+eq+'%s'" % imgName
    
    
    response = requests.get(server+imgQuery,auth=(username,password))
    

    # Lecture et parser la réponse du serveur
    xmlFile = minidom.parseString(response.text)
        
    # Chercher les noeuds ENTRY
    for node in xmlFile.getElementsByTagName('entry'):
        
        for i in node.getElementsByTagName('d:Id'):
            id = i.firstChild.nodeValue
        
            for d in node.getElementsByTagName('d:Online'):
                
                status = d.firstChild.nodeValue
            
                if status == 'false':
                    
                    
                    # Création du nom de la requête à envoyer aux serveurs. La requête permet
                    # de télécharger les images
                    dwnQuery = "('%s')/$value" % id
                    dwn_url = server+dwnQuery
                    
                    r = requests.get(dwn_url,auth=(username,password), stream=True)
                    print('Réponse du serveur :',r)
                    r.close()
                    
                    return status
                    
                else:
                 
                    # Création du nom de la requête à envoyer aux serveurs. La requête permet
                    # de télécharger les images
                    dwnQuery = "('%s')/$value" % id
                    dwn_url = server+dwnQuery
                    
                    print('URL de téléchargement :', dwn_url)
                    
                    # Création du nom du fichier à télécharger
                    fileName = r"%s\%s.zip" % (filepath,imgName)
                    
                    
                    
                    with open(fileName, "wb") as file:
                        
                        r = requests.get(dwn_url,auth=(username,password), stream=True)       
                        file.write(r.content)
                        file.flush()
                        os.fsync(file.fileno())
                        
                        r.close()
                    
                    return status
    response.close()    


def atmCorrS2(locSen2Cor,locFileName,outfilepath,param,bands):
    
    '''
    Fonction utilisant le plugin Sen2Cor intégré dans SNAP. Dans ce cas-ci, 
    Sen2Cor est utilisé à partir de son fichier .bat.
    
    La fonction convertie les images S2 de L1C (TOA) vers L2A (BOA).
    
    locSen2Cor:     Chemin d'accès vers l'exécutable de Sen2Cor.
    locFileName:    Chemin d'accès vers le répertoire contenant le fichier .SAFE.
    outfilepath:    Chemin d'accès pour les images L2A géoréférencées. Ces images 
                    seront celles qui seront utilisées pour de futures analyses.
    '''
    
    file = os.listdir(locFileName)
       
    count = 0
    
    img_10m = ['B02','B03','B04','B08','TCI','WVP']
    img_20m = ['B05','B06','B07','B8A','B11','B12','SCL']
    
    for f in file:
        
        if "L1C" in f and f.endswith('.SAFE'):
            
            count += 1
            
            print("Correction atmosphérique avec Sen2Cor")
            print("\nCorrection atmosphérique de l'image %d\n" % (count))
            
            # Identification du lecteur à utiliser
            lecteur = locFileName[:2]
    
            # Identification du chemin d'accès au dossier contenant le fichier .SAFE
            filePath = locFileName[3:]
                    
            # Activation de la console CMD pour lancer Sen2Cor
            os.system('cmd /k "cd\ & %s & cd %s & %s --GIP_L2A %s --tif %s & exit"' % (lecteur,filePath,locSen2Cor,param,f))
            
            granule = lecteur+'\\'+filePath+'\\'+f+'\\'+'GRANULE'
           
            granule_dir = os.listdir(granule)
            
            
            imgFile = granule+'\\'+granule_dir[0]+'\\'+'IMG_DATA'
            
            imgFile_dir = os.listdir(imgFile)
            
            imgFilePath = imgFile+'\\'+imgFile_dir[1]
            
            proj,geotransform = getRasterProj(imgFilePath)
            
            # Supression des fichiers 'L1C' qui ne sont plus nécessaire pour libérer de l'espace disque.
            cleanFolder(locFileName+'\\'+f)
            
            product_date = f[11:26]
            
            file2 = os.listdir(locFileName)
            
            
            # Sélection des images (bandes) avec correction atmosphérique désirées
            for f2 in file2:
                            
                if "L2A" in f2 and product_date in f2 and f2.endswith('.SAFE'):
                      
                    granule2 = lecteur+'\\'+filePath+'\\'+f2+'\\'+'GRANULE'
                   
                    granule_dir2 = os.listdir(granule2)
                                        
                    imgFile_10m = granule2+'\\'+granule_dir2[0]+'\\'+'IMG_DATA'+'\\'+'R10m'
                    
                    imgFile_dir_10m = os.listdir(imgFile_10m)
                    
                    imgFile_20m = granule2+'\\'+granule_dir2[0]+'\\'+'IMG_DATA'+'\\'+'R20m'
                    
                    imgFile_dir_20m = os.listdir(imgFile_20m)
                    
               
                    for b in bands:
                        
                        if b in img_10m:

                            for i in imgFile_dir_10m:
                            
                                if b in i and i.endswith('.tif'):
                        
                                    xsize,ysize,geotransform2,proj2,array = readRaster(imgFile_10m+'\\'+i)
                                    
                                    writeRaster(outfilepath+'\\'+i,geotransform,proj,array,0)
        
                        if b in img_20m:
                            
                            for i in imgFile_dir_20m:
                            
                                if b in i and i.endswith('.tif'):
                        
                                    xsize,ysize,geotransform2,proj2,array = readRaster(imgFile_20m+'\\'+i)
                                    
                                    # Créer un geotransform adapté aux images de résolution de 20 m
                                    geotransform_20m = (geotransform[0], 20, 0.0, geotransform[3], 0.0, -20)
                                    writeRaster(outfilepath+'\\'+i,geotransform_20m,proj,array,0)
                    
                    # Supression des fichiers 'L2A' pour libérer de l'espace disque.
                    cleanFolder(locFileName+'\\'+f2)
        
    print('Correction atmosphérique terminée')            
                        
            
def unzip(fileName,outPath):
    '''
    Fonction pour décompresser un fichier .zip.
    
    fileName: chemin d'accès vers le fichier à décompresser 
    outPath: direction ou le fichier sera décompressé.
    '''

    print('Extraction du fichier %s en cours' %fileName)
    with zipfile.ZipFile("%s" % fileName,"r") as unzip:
        unzip.extractall(outPath)
        
        print('Extraction du fichier %s terminé' %fileName)

def export2json(data, filename, filepath):
    '''
    Fonction permettant d'exporter une liste, dictionnaire, etc. en fichier .json.
    
    data: liste ou dictionnaire à exporter
    filename: nom du fichier .json
    filepath: chemin d'accès ou sera enregistré le fichier .json
    '''

    with open("%s/%s.json" % (filepath,filename), 'w') as f:
        json.dump(data, f, indent=2)
    
    
def importJson(filepath):
    '''
    Fonction permettant d'importer un fichier .json contenant une liste ou un dictionnaire
    
    filepath: chemin d'accès vers le fichier .json
    '''
    
    with open(r"%s" % filepath, 'r') as f:
        data = json.load(f)
                  
    return data


def getRasterProj(filename):
    '''
    Fonction permettant d'obtenir la projection d'une image, ainsi que le GeoTransform.
    
    filename: chemin d'accès pour l'image.
    '''
   
    raster = gdal.Open(filename)
    geotransform = raster.GetGeoTransform()
    proj = raster.GetProjection()

    print('Extraction de la projection de l\'image %s' % filename)    
    
    return proj,geotransform

def getNodata(filename, band):
    '''
    Fonction permettant d'obtenir les valeurs de pixels NoData.
    
    filename: chemin d'accès de l'image
    band: choix de la bande à obtnenir le NoData.
    '''
    raster = gdal.Open(filename)
    band = raster.GetRasterBand(band)
    
    return band.GetNoDataValue()
             
def readRaster(filename,*band):
    '''
    Fonction pour lire un raster. En sortie, les dimensions en X et Y, le 
    geotransform, la projection et la matrice de l'image sont produits.
    
    Il est possible d'extraire l'ensemble des bandes du raster ou seulement les
    bandes désirées. L'arguement *band est activé uniquement lorsqu'il est déclaré,
    donc est facultatif.
    
    filename: chemin vers le raster
    *band: Argument facultatif. Il est activé uniquement lorsqu'il est déclaré.
            ex. d'utilisation de *band -> band = [1,2,3,...]
    '''
    
    print('Lecture de l\'image %s' % filename)
    
    raster = gdal.Open(filename)
    geotransform = raster.GetGeoTransform()
    proj = raster.GetProjection()
 
    xsize = raster.RasterXSize
    ysize = raster.RasterYSize

    # Si une ou des bandes particulières sont fournies celles-ci seront extraites.
    if band:
        array = np.empty([len(band[0]),ysize,xsize])
        
        count = -1
         
        for i in band[0]:

            count += 1
           
            b = raster.GetRasterBand(i)
            a = b.ReadAsArray()
           
            array[count,:,:] = a

    else:
        array = raster.ReadAsArray()
    
    
    return xsize,ysize,geotransform,proj,array

def writeRaster(outfilename,geotransform,projection,array,noData):
    '''
    Fonction permettant d'écrire une matrice 2D ou 3D sous forme de raster.
    
    '''
    
    array[np.isnan(array)] = noData
    
    format = "GTiff"
    
    driver = gdal.GetDriverByName(format)
    
    dtype = gdal.GDT_Float32
    
    try:
        
        # Si l'image à écrire contient plusieurs bandes, cette section est utilisée.
        
        band,x,y = array.shape
    
        dst_ds = driver.Create(outfilename,y,x,band,dtype)
    
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projection)
        
        
        for i,image in enumerate(array,1):
            
            dst_ds.GetRasterBand(i).WriteArray(image)
            dst_ds.GetRasterBand(i).SetNoDataValue(noData)
    
    
    except:
        
        # Si l'image a seulement une bande cette section est utilisée.
        
        x,y = array.shape
    
        dst_ds = driver.Create(outfilename,y,x,1,dtype)
    
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projection)
        
        dst_ds.GetRasterBand(1).WriteArray(array)
        dst_ds.GetRasterBand(1).SetNoDataValue(noData)
    
    
    
    dst_ds.FlushCache()
    dst_ds = None
    array = None

    print('Exportation de l\'image avec projection terminée')
    
    
def cleanFolder(directory):
    '''
    Fonction permettant de supprimer un dossier et son contenu.
    
    directory: chemin du dossier
    '''
    
       
    print('Nettoyage des fichiers')
    
    for i in os.listdir(filePath):
        
        if i.endswith('.SAFE'):
            
            print('Supression de %s' % filePath+'\\'+i)
            shutil.rmtree(filePath+'\\'+i)   
    

def ndvi(r,nir):
    '''
    Fonction calculant le Normalized Difference Water Index.
    
    
    g: Array représentant une bande verte
    nir: Array représentant une bande proche infrarouge
    '''
    
    r = r.astype(np.float32)
    nir = nir.astype(np.float32)

    NDVI = (nir-r)/(nir+r)
    
    return np.array(NDVI)

def ndwi(g,nir):
    '''
    Fonction calculant le Normalized Difference Water Index.
    
    
    g: Array représentant une bande verte
    nir: Array représentant une bande proche infrarouge
    '''
    
    g = g.astype(np.float32)
    nir = nir.astype(np.float32)

    NDWI = (g-nir)/(g+nir)
    
    return NDWI

def depth(b,g):
    '''
    Fonction permettant de calculer un indice de profondeur utilisant un
    ratio logarithmique (Stumpf, 2003)
    
    b: Array représentant une bande bleue
    g: Array représentant une bande verte

    '''
    
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    
    d = np.log(b)/np.log(g)
    
    return np.array(d)



def decimalYear(jour,mois,annee):
    ''' 
    Fonction pour convertir les dates en années décimales.
    
    jour: jour à convertir
    mois: mois à convertir (format numérique ex: janvier = 1)
    annee: annee à convertir
    
    '''
    
    year = int(annee)
    month = int(mois)
    day = int(jour)
    
        
    # Cette condition vérifie si l'année est bissextile. Ces conditions doivent
    # être rencontrées pour être une année bissextile.
    if (year % 4) == 0 and (year % 100) != 0 or (year % 400) == 0:
        
        dyear = (1/366) * julianDay(day,month,year) + year
        
        return dyear
    
    else:
         dyear = (1/365) * julianDay(day,month,year) + year
         
         return dyear
     
        

def julianDay(jour,mois,annee):
    '''
    Convertion des dates en jours Juliens.
    
    jour: jour à convertir
    mois: mois à convertir (format numérique ex: janvier = 1)
    annee: annee à convertir
    
    '''    
    
    # Liste du nombre de jours dans les mois
    m1 = [31,28,31,30,31,30,31,31,30,31,30,31]
    
    # Liste du nombre de jours dans les mois des années bissextiles.
    m2 = [31,29,31,30,31,30,31,31,30,31,30,31]
    
    year = int(annee)
    month = int(mois)
    day = int(jour)
    
    
    # Cette condition vérifie si l'année est bissextile. Ces conditions doivent
    # être rencontrées pour être une année bissextile.
    if (year % 4) == 0 and (year % 100) != 0 or (year % 400) == 0:
        
        jd = day + sum(m2[0:month-1])

        return jd

    else:
        
        jd = day + sum(m1[0:month-1])
            
        return jd

# Contient la géométrie sous forme WKT.
geomLayer = []

# Liste contenant les matrice numpy des images S2
array_s2_clip = {}

# Compteur pour identifier des scènes différentes prises au même moment pour les fusionner
name_img_dict = {}

# Contient le nom des images à fusionner à l'aide de gdal_merge.
img_to_merge = {}


def main():
    
    tic = timeit.default_timer()
    
    ##########################################################################
    ######################### Section paramètres #############################
    ##########################################################################
    
    # Localisation du fichier de formes contenant la zone d'étude
    inputFeature = r"C:\Users\user\Documents\zone_etude.shp"
    
    # Localisation de la carte bathymétrique
    bathymetrie = r"C:\Users\user\Documents\DEM_AlguesEauTerre_BG_FGA_L5_L7_L8_S2_10m_IDLMsud_1985_2019_Sud1.tif"
    
    # Localisation de la carte d'occupation du sol
    classes_sol = r"C:\Users\user\Documents\occ_sol.tif"
    
    # Chemin vers le fichier recevant les images téléchargées
    filePath = r"D:\img_S2\IDLM"
    
    # Chemin vers le fichier contenant les images S2 corrigées (L2A)
    locL2AFile = r"D:\img_S2\IDLM\L2A_S2_georef"
    
    # Chemin vers le fichier contenant les images S2 corrigées et clippées à la ZE.
    locL2AClip = r"D:\img_S2\IDLM\L2A_S2_georef\img_clip"
    
    # Pourcentage du couvert nuageux à appliquer pour le filtrage des images
    cloudPerc = 10
    
    # Mois de début et de fin à prendre en compte pour l'étude.
    startMonth = 6
    endMonth = 10
    
    # Bandes spectrale des images Sentinel-2 à utiliser dans l'analyse.
    bands = ['B02','B03','B04','B08','SCL']
    
    # Valeur de noData à attribuer aux rasters.
    noData = -9999
    
    # Information pour la connection au serveur de Copernicus pour le téléchargement des images.
    username = 'guillec92'
    password = 'Gl5519384!'
    
    # Localisation de l'exécutable de Sen2Cor
    sen2cor = r"C:\Users\user\.snap\auxdata\Sen2Cor-02.08.00-win64\L2A_Process.bat"
    
    # Localisation du fichier des paramètres pour la correction atmosphérique de sen2cor.
    atmcorr_param = r"C:\Users\user\Documents\L2A_GIPP.xml"
    
    # Chemin d'accès au répertoire OSGeo.
    OSGeoPath = 'C:\\OSGeo4W64\\bin'
    
    ##########################################################################
    ################# Commandes pour exécuter le code ########################
    ##########################################################################
    

    # Fait la lecture du fichier de forme et l'ouvre
    ds = readFeature(inputFeature)
        
    lyr = ds.GetLayer('feature')
       
    for item in lyr:
        geom = item.GetGeometryRef()
        wkt = geom.ExportToWkt()
        extent = geom.GetEnvelope()
        geomLayer.append(wkt)
  
    # Active la couche pour obtenir le SRID de la couche
    proj = getProjection(lyr)

    # Reprojeter le fichier de formes
    wkt = reprojeter(geomLayer,proj,4326)
      
    # Convertie la géométrie WKT vers GeoJSON pour être compatible avec EarthEngine
    coord = getStudyAreaCoordinates(wkt)

    # Initialise la librairie earthengine. Un compte google earthengine doit être connecté à l'ordinateur.
    ee.Initialize()
    
    
    # Création du polygone délimitant la zone d'étude
    study_area = ee.Geometry.Polygon(coord, 'EPSG:4326').bounds()
           
    # Obtenir l'année en cours et calculer 24 mois avant l'année en cours
    actual_year = datetime.datetime.now().year
    start_year = actual_year - 2  
    
    #Identification du capteur à utiliser
    s2 = ee.ImageCollection('COPERNICUS/S2')
    
    # define your collection
    time_series_s2 = s2.filter(ee.Filter.calendarRange(start_year,actual_year,'year'))\
    .filter(ee.Filter.calendarRange(startMonth,endMonth,'month'))\
    .filterBounds(study_area)\
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',cloudPerc))


    nb_img_s2 = time_series_s2.size().getInfo()
    print('Nombre d\'images recouvrant la zone d\'étude: ',nb_img_s2)
    
     # Tous les noms des images sont stockés.
    images_names_s2 = [item.get('id') for item in time_series_s2.getInfo().get('features')]
    
   
    # Enregistrer les noms des images 'offline' lors du téléchargement pour y
    # accéder lorsqu'elles seront mises en ligne
    imgOffline = []
    
     
    # Si des images n'ont pas été téléchargées en raison de leur status
    # 'offline', le serveur les remet en ligne maximum 24 heures après la demande.
        
    for i in os.listdir(filePath):
        
        if i.endswith('.json'):
            
            imgOnline = importJson(filePath+'\\'+i)
            
            count = 0
            for j in imgOnline: 
                count += 1
                
                print("\nImage %d sur %d \n" % (count,len(imgOnline)))
                print("Téléchargement de l'image :", j)
                
                # Activation de la fonction permettant de télécharger les images S2 
                # provenant des serveurs de Copernicus
                online = download_S2(j,filePath,username,password)
                
                if online == 'false':
          
                    imgOffline.append(j)
                    print("Image non téléchargée, car non disponible :", j)
                
                else:
                    print("Téléchargement de l'image complété\n")
            
                toc = timeit.default_timer()
                print('\nTemps écoulé :', (toc-tic)/60)
                    
                print("\nTéléchargement de l'ensemble des images S2 complété\n")
             
            
            for l in os.listdir(filePath):
                                
                for k in imgOnline:
                
                    if k in l:
                        
                        # Décompresser les images S2 téléchargées.
                        unzip(filePath+'\\'+l,filePath)
            
            # Activation de la correction atmosphérique pour les images S2
            atmCorrS2(sen2cor,filePath,locL2AFile,atmcorr_param,bands)
            
            
            if imgOffline:
                
                # Exporation du nom des images non téléchargées
                export2json(imgOffline,'imgoffline',filePath)
            
                print('\n\nDes images n\'ont pas été téléchargées. Status: offline')    
                
                print('\n\n\n*****************************************************\
                      \nInterruption du script, car l\'ensemble des images n\'est pas téléchargé. Veuillez relancer le script lorsque ces images seront disponibles.\
                          \n*****************************************************')
                
                # Quitter le script en raison d'images non téléchargées
                sys.exit()
            
            # Si toutes les images ont été téléchargées, le fichier .json contenant les noms des
            # images à télécharger est supprimé.
            if not imgOffline:
                os.remove(filePath+'\\'+i)
                
            
        else:
   
            # Téléchargement des images selon leur ID.
            count = 0
            for i in images_names_s2:
            
                # Accéder au Product_ID de l'image
                img = ee.Image(i).get('PRODUCT_ID').getInfo()
                print("\nImage %d sur %d \n" % (count,int(nb_img_s2)))
        
                # Activation de la fonction permettant de télécharger les images S2 
                # provenant des serveurs de Copernicus
                print("Téléchargement de l'image :", img)
        
                online = download_S2(img,filePath,username,password)
        
                # Ajoute le nom des images hors ligne, donc pas téléchargées pour y 
                # accéder plus tard lorsqu'elles seront en ligne
                
                if online == 'false':
                  
                    imgOffline.append(img)
                    print("Image non téléchargée, car non disponible :", img)
                
                else:
                    print("Téléchargement de l'image complété\n")
            
                toc = timeit.default_timer()
                print('\nTemps écoulé :', (toc-tic)/60)


            print("\nTéléchargement de l'ensemble des images S2 complété\n")
            
             
            # Décompresser les images S2 téléchargées.
            for f in os.listdir(filePath):
                if f.endswith('.zip'):
                    unzip(filePath+'\\'+f,filePath)
            
            toc = timeit.default_timer()
            print('\nTemps écoulé :', (toc-tic)/60)
            
            # Activation de la correction atmosphérique pour les images S2
            atmCorrS2(sen2cor,filePath,locL2AFile,atmcorr_param,bands)
                
                 
            toc = timeit.default_timer()
            print('\nTemps écoulé :', (toc-tic)/60) 
    
            
            if imgOffline:
                # Exporation du nom des images non téléchargées
                export2json(imgOffline,'imgoffline',filePath)
                
                print('\n\nDes images n\'ont pas été téléchargées. Status: offline')   
                
                print('\n\n\n*****************************************************\
                      \nInterruption du script, car l\'ensemble des images n\'est pas téléchargé. Veuillez relancer le script lorsque ces images seront disponibles.\
                          \n*****************************************************')
                # Quitter le script en raison d'images non téléchargées
                sys.exit()
                
    
    ### Clipper les images à l'étendue de la zone d'étude ###
    
    # Déclarer la résolution de l'image
    pixelSize = 10
    # Valeur de l'étendue
    minX = extent[0]
    minY = extent[2]
    maxX = extent[1]
    maxY = extent[3]
    bounds = (minX,minY,maxX,maxY)
    
    for i in os.listdir(locL2AFile):
        if i.endswith('.tif'):
            
            # Déclaration des options
            options = {
                'xRes':pixelSize,
                'yRes':pixelSize,
                'resampleAlg':None,
                'outputBounds':bounds,
                # 'cutlineDSName':inputFeature,
                # 'cropToCutline':True,
                # 'srcNodata':0,
                'dstNodata':0,
                'multithread':True
                        }
            # Appliquer le ré-échantillonage pour les image de résolution de 20m.
            if '20m' in i:
                options['resampleAlg'] = 'near'
                                
            gdal.Warp("%s/%s_clip.tif" % (locL2AClip,i[:-4]),"%s/%s" % (locL2AFile,i),**options)
         
            print('Clip de l\'image %s à la zone d\'étude' % i)
            
            
        
    # Détecter les images des mêmes dates pour les fusionner ensemble
    for i in sorted(os.listdir(locL2AClip)):
        
        if i[7:26] in name_img_dict:       
            name_img_dict[i[7:26]] += 1

        else:
            name_img_dict[i[7:26]] = 1



    # Déterminer si des images de même date existent. Si oui, extraire le nom des
    # images qui serviront à réaliser les futurs traitements. Si non, conversion des 
    # images en matrice numpy.
    for x,y in sorted(name_img_dict.items()):
            
        # Si une date image est comptée plus d'une fois, le nom des images à
        # mosaïquer est enregistré dans le dictionnaires img_to_merge.
        if y > 1:
            
            for i in sorted(os.listdir(locL2AClip)):
                
                if x in i and i.endswith('.tif'):
                    if x not in img_to_merge:
                        img_to_merge[x] = [i]

                    else:
                        img_to_merge[x].append(i)
        
        # Si une date d'image est présente une seule fois, l'image est
        # directement ajoutée au dictionnaire array_s2_clip.
        else:
            for i in sorted(os.listdir(locL2AClip)):
                
                if x in i and i.endswith('.tif'):
                    
                    # Lecture de l'image Sentinel-2 clippée
                    X, Y, geotransform,proj,array = readRaster("%s/%s" % (locL2AClip,i),[1])
                                        
                    # Ajout des différentes bandes spectrales utilisées pour
                    # chacune des dates.
                    if i[7:22] not in array_s2_clip:
                        array_s2_clip[i[7:22]] = [array/10000]
                
                    else:
                        array_s2_clip[i[7:22]].append(array/10000)                 
                        
                    del X,Y,geotransform,proj,array
 
    # Importer le script MERGE de GDAL.
    sys.path.append(OSGeoPath)
    import gdal_merge as gm
    
    # Si des images d'une même date sont détectées, appliquer gdal_merge pour
    # fusionner les tuiles de la zone d'étude.
    if img_to_merge:
        
        # Recherche des noms de fichier dans le dictionnaire.
        for x,y in img_to_merge.items():
            
            file = []
            for d in y:
                file.append("%s/%s" %(locL2AClip,d))
            
            # Activation du script pour exécuter le mosaïquage. Les images S2 ont comme nodata la valeur 0.
            gm.main(['','-o', "%s/%s_merge.tif" % (locL2AClip,y[1]), '-ot', 'Int16', '-n','0','-a_nodata',noData,*file])
            
            # Lecture des images mosaïquées
            X,Y,geotransform,proj,array = readRaster("%s/%s" % (locL2AClip,y[1]+'_merge.tif'),[1])
            
            # Stocker les images sous forme de liste
            if y[1][7:22] not in array_s2_clip:
                array_s2_clip[y[1][7:22]] = [array/10000]

            else:
                array_s2_clip[y[1][7:22]].append(array/10000)
          
            del X,Y,geotransform,proj,array
        
        print('Mosaïquage terminé')
        
    else:
        print('Aucune image n\'a été mosaïquée')
        
        
    #################################
    ### Créer la série temporelle ###
    #################################
    
    # Lire les rasters représentant le MNT et la couverture au sol
    x,y,geotransform,proj,bathym = readRaster(bathymetrie,[1])
    x,y,geotransform,proj,occ_sol = readRaster(classes_sol,[1])
    
    bathym = bathym[0]
    occ_sol = occ_sol[0]
    
    img_date = []
    
    # Contient l'index des images pour chacune des années.
    index_year = {}
    
    indice_prof = []
    
    indice_prof_mask = []
    
    limite_VeryShallow = [0.5,2.]
    limite_Shallow = [4.,6.]
    limite_Deep = [10.,12.]
    
    
    # Dictionnaire pour extraire les valeurs de l'indice de profondeur pour
    # chacun des pixels et pour chacune des zones de profondeur
    indice_VShal = {}
    indice_Shal = {}
    indice_Deep = {}

    # Liste pour extraire les valeurs de profondeur pour chacune des zones de profondeur.
    pixel_VShal = []
    pixel_Shal = []
    pixel_Deep = []
    
    # Index des bandes bleue,verte,rouge, pif et la couche de masque de nuages
    blue = 0
    green = 1
    red = 2
    nir = 3
    idx_mask_cloud = len(bands)-1
    
    
    countVShal = 0
    countShal = 0
    countDeep = 0
    
    # Compteur pour identifier l'index des images dans le dictionnaire
    count_index = 0
    
    # Masquer les zones nuageuses à l'aide de la bande 'SCL' de S2.
    for i,j in sorted(array_s2_clip.items()):
        
        for b in range(0,len(bands),1):

            # Ne prend pas en compte la dernière bande qui est le masque 'SCL'. Cette condition peut être changée en fonction de la localisation réelle de la bande de masque ou tout simplement si elle existe.
            if b < idx_mask_cloud:
                
                
                ### Appliquer un filtre gaussien aux images pour réduire le bruit.
                array_s2_clip[i][b][0] = gaussian_filter(j[b][0],2)
                
        print('Filtre gaussien appliqué aux images: ')
        
        # Localisation des informations dans le nom de l'image
        year = i[0:4]
        month = i[4:6]
        day = i[6:8]
        
        # Conversion des dates en année décimale.
        img_date.append(decimalYear(day,month,year))         
        
        # Identification des index des images pour chacune des années.
        if year not in index_year:
            index_year[year] = [count_index]
            
        else:
            index_year[year].append(count_index)
        
        count_index += 1
        
        
        print('Calcul de l\'indice de profondeur pour la date: ', year+month+day)
        
        ### Calcul de l'indice de profondeur
        DEPTH = depth(j[blue][0],j[green][0])
        
        indice_prof.append(DEPTH)
        
        ### Appliquer un filtre adaptatif Wiener ###
        DEPTH3 = wiener(DEPTH,3)
        DEPTH33 = wiener(DEPTH3,3)
        DEPTH335 = wiener(DEPTH33,5)
        
        del DEPTH,DEPTH3,DEPTH33
        
        print('Filtre adaptatif de type Wiener pour la date: ', year+month+day)
        
         
        ### Masquer les nuages présents dans les images ###
        
        mask_band = j[idx_mask_cloud][0].copy()
        
        mask_band[np.where((mask_band == 3) |
                            # mask_band == 7 |
                            (mask_band == 8) |
                            (mask_band == 9) |
                            (mask_band == 10))] = noData
        
        mask_band[np.where(mask_band != noData)] = 1
        
        mask_band[np.where(mask_band == noData)] = np.nan
        
        depth_mask = DEPTH335 * mask_band
                
        indice_prof_mask.append(depth_mask)
        
        del DEPTH335
        
        print('Masquage des nuages pour la date: ', year+month+day)
        
        
        
        for k in range(0,np.size(bathym,0),1):           
        
            for l in range(0,np.size(bathym,1),1):
                
                
                
                # Extraction des pixel de l'indice de profondeur représentant
                # les zones 'Shallow'.
                if occ_sol[k][l] == 20 and bathym[k][l] > limite_Shallow[0] and bathym[k][l] < limite_Shallow[1]:
  
                    if count_index == 1:
                        pixel_Shal.append(bathym[k][l])
                    
    
  
                    if countShal not in indice_Shal:
                        
                        indice_Shal[countShal] = [depth_mask[k][l]]
                    
                        countShal += 1
                    
                    else:
                        indice_Shal[countShal].append(depth_mask[k][l])
                        
                        countShal += 1
                    
                    
                # Extraction des pixel de l'indice de profondeur représentant
                # les zones 'Deep'.    
                if occ_sol[k][l] == 20 and bathym[k][l] > limite_Deep[0] and bathym[k][l] < limite_Deep[1]:
  
                    if count_index == 1:
                        pixel_Deep.append(bathym[k][l])
  
                    if countDeep not in indice_Deep:
                        
                        indice_Deep[countDeep] = [depth_mask[k][l]]
                    
                        countDeep += 1
                    
                    else:
                        indice_Deep[countDeep].append(depth_mask[k][l])
                        
                        countDeep += 1
                
                
                # Extraction des pixel de l'indice de profondeur représentant
                # les zones 'Very Shallow'.
                if occ_sol[k][l] == 20 and bathym[k][l] > limite_VeryShallow[0] and bathym[k][l] < limite_VeryShallow[1]:
  
                    if count_index == 1:
                        pixel_VShal.append(bathym[k][l])         
  
    
                    if countVShal not in indice_VShal:
                        
                        indice_VShal[countVShal] = [depth_mask[k][l]]
                    
                        countVShal += 1
                    
                    else:
                        indice_VShal[countVShal].append(depth_mask[k][l])
                        
                        countVShal += 1
                    

        # Remet les compteurs de pixels à 0.
        countVShal,countShal,countDeep = [0,0,0]
    
    # Conversion de la liste de dates en matrice numpy.
    img_date = np.array(img_date)   
        
    # Conversion des listes en matrice numpy.
    indice_prof = np.array(indice_prof)
    indice_prof_mask = np.array(indice_prof_mask)
    
    # Conversion des listes des pixels des différentes zones de profondeur en matrice numpy.
    pixel_Shal = np.array(pixel_Shal)
    pixel_Deep = np.array(pixel_Deep)
    pixel_VShal = np.array(pixel_VShal)
      
    # Moyenne de la valeur de l'indice de profondeur pour chaque pixel et pour chacune des classes de profondeur.
    moyPixDeep = [np.nanmean(j) for i,j in  indice_Deep.items()]
    moyPixShal = [np.nanmean(j) for i,j in  indice_Shal.items()]
    moyPixVShal = [np.nanmean(j) for i,j in indice_VShal.items()]

 
    # Valeur de l'indice de profondeur de référence pour chacune des classes de profondeur.
    refIndiceDeep = []
    refIndiceShal1 = []
    refIndiceShal2 = []
    refIndiceVShal = []
    refIndiceVShal2 = []

    # Permet de calculer le percentile des indices de profondeur pour chacune des zones.    
    for i in range(0,len(indice_prof_mask),1):
        
        valueDeep = []
        valueShal = []
        valueVShal = []

        for j in indice_Deep.values():
            
            valueDeep.append(j[i])
       
        for j in indice_Shal.values():
        
            valueShal.append(j[i])
            
        for j in indice_VShal.values():
         
            valueVShal.append(j[i])
                    
        refIndiceDeep.append(np.nanpercentile(valueDeep,25))
        refIndiceShal1.append(np.nanpercentile(valueShal,75))
        refIndiceShal2.append(np.nanpercentile(valueShal,20))
        refIndiceVShal.append(np.nanpercentile(valueVShal,60))
        refIndiceVShal2.append(np.nanpercentile(valueVShal,60))
        # del valueDeep,valueShal,valueVShal
    
    # Conversion des listes en matrice numpy
    refIndiceDeep = np.array(refIndiceDeep)
    refIndiceShal1 = np.array(refIndiceShal1)
    refIndiceShal2 = np.array(refIndiceShal2)
    refIndiceVShal = np.array(refIndiceVShal)
    refIndiceVShal2 = np.array(refIndiceVShal2)
    
    
    # Percentile des pixels convertir en valeur de profondeur
    refPixelDeep = np.percentile(pixel_Deep,75)
    refPixelShal1 = np.percentile(pixel_Shal,25)
    refPixelShal2 = np.percentile(pixel_Shal,60)
    refPixelVShal = np.percentile(pixel_VShal,20)
    refPixelVShal2 = np.percentile(pixel_VShal,20)
    
  
    ### Création de la matrice représentant les changements de bathyémtrie ###
    
    
    
    # Matrice des changements de bathymétrie. Les pixels enregistrés représentent
    # la pente des changements. Plus la pente est forte, plus il y a un changement.
   
    j = np.arange(0,37,1)
        
    pente_bathym = np.full_like(bathym,np.nan)
    
    # prof_calc1 = np.full_like(bathym,np.nan)
    
    
    n = img_date[0:len(j)]
    p = []
    
    for k in range(0,np.size(bathym,0),1):           
        
            for l in range(0,np.size(bathym,1),1):
                
                            
                # Extraction des pixel de l'indice de profondeur représentant
                # les zones 'Shallow' à 'Deep'.
                if occ_sol[k][l] == 20 and bathym[k][l] > 4. and bathym[k][l] < 12.:
                
                    # Extraction d'un pixel pour l'ensemble des dates.
                    x = indice_prof_mask[j[0]:j[-1]+1,k,l]
                    
                    # Calculer les changements pour les zones Deep.
                    norm_indice = (1-0.9)*((x-refIndiceDeep[j[0]:j[-1]+1])/(refIndiceShal1[j[0]:j[-1]+1]-refIndiceDeep[j[0]:j[-1]+1]))+0.9
                    
                    # Calculer changement pour les zones Shallow
                    # norm_indice = (1-0.9)*((x-refIndiceVShal2[j[0]:j[-1]+1])/(refIndiceShal2[j[0]:j[-1]+1]-refIndiceVShal2[j[0]:j[-1]+1]))+0.9
                    
                    
                    # Retire les données aberrantes             
                    # for z in range(0,len(norm_indice),1):
                        
                    #     if norm_indice[z] < 0.82 or norm_indice[z] > 1.1:
                    #         norm_indice[z] = np.nan
                    
                    idx = np.isfinite(n) & np.isfinite(norm_indice)
                    
                    pente_indice = np.polynomial.polynomial.Polynomial.fit(n[idx[0:len(idx)]],norm_indice[idx[0:len(idx)]],1)
                    
                    p_indice = pente_indice.convert().coef
                    
                    
                    # Conversion des indices de profondeur normalisés en profondeur réelle pour les zone Deep.
                    pente_metre = np.polynomial.polynomial.Polynomial.fit([0.9,1.0],[refPixelDeep,refPixelShal1],1)
                    
                    # Conversion des indices de profondeur normalisés en profondeur réelle pour les zone Shallow.
                    # pente_metre = np.polynomial.polynomial.Polynomial.fit([0.9,1.0],[refPixelShal2,refPixelVShal2],1)
                    
                    # Obtenir l'origine et la pente de la fonction
                    p_metre = pente_metre.convert().coef
                    p_metre = [p_metre[1],p_metre[0]]
                    
                    # Créer la fonction de conversion
                    p = np.poly1d(p_metre)
                    
                    # Liste contenant les profondeurs calculées
                    prof_calc = []
                    
                    for a in norm_indice:
                        prof_calc.append(p(a))
                        
                        
                    prof_calc = np.array(prof_calc)
                    
                    idx2 = np.isfinite(n) & np.isfinite(prof_calc)
                    
                    # Liste contenant la bathymétrie pour une date particulière
                    # prof_calc1[k,l] = p(norm_indice[3])
                    
                    
                    try:
                        pente = np.polynomial.polynomial.Polynomial.fit(n[idx2[0:len(idx2)]],prof_calc[idx2[0:len(idx2)]],1)
                        
                        pente_coef = pente.convert().coef
                        
                        pente_bathym[k,l] = pente_coef[1]
                        
                  
                    except:
                        pente_bathym[k,l] = np.nan
                        
                    
            
                    
                        
    
    writeRaster('changement_profondeur.tif',geotransform,proj,pente_bathym,noData)

    # writeRaster('profondeur.tif',geotransform,proj,prof_calc1,noData)
    
    toc = timeit.default_timer()
    
    print('\nTemps total de traitement: ', (toc-tic)/60)    
