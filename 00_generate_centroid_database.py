'''
AIM: Use African Shapefile to build up ADM1 centroid library.
OUTPUT: 'output_data/africa-centroid-database.csv'
'''
import pandas as pd
import geopandas as gpd

# %% Load in complete centroid library
afr = gpd.read_file("raw_data/Africa/africa_complete.shp")

# %% Build up ADM2 centroids (Some AD1 units have multiple entries (shapefiles), thus the need to disolution)
adm2 = afr.dissolve(by=['country','adm1','adm2']).reset_index()
# Some units have no ADM2, recover these instances
dropped = afr[~afr.set_index(['country','adm1','adm2']).index.isin(adm2.set_index(['country','adm1','adm2']).index)]

# Build out adm2 database
adm2_complete = pd.concat([adm2,dropped],sort=True)
adm2_complete['adm2_centroid_lon'] = adm2_complete.geometry.centroid.x
adm2_complete['adm2_centroid_lat'] = adm2_complete.geometry.centroid.y
main = adm2_complete.drop('geometry',axis=1)

# %% Build up ADM1 centroids
adm1 = afr[['country','adm1','geometry']].dissolve(by=['country','adm1']).reset_index()
adm1['adm1_centroid_lon'] = adm1.geometry.centroid.x
adm1['adm1_centroid_lat'] = adm1.geometry.centroid.y
ad1_sub = adm1[['country','adm1','adm1_centroid_lon','adm1_centroid_lat']]
main2 = main.merge(ad1_sub,on=['country','adm1'],how='left')

if main2.shape[0] != main.shape[0]:
    '''Test for loss'''
    raise ValueError("Issue with the ADM1 merge.")

# %% Export data
main2.to_csv("output_data/africa-centroid-database.csv",index=False)
