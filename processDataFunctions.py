import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon

from warnings import simplefilter
simplefilter('ignore')

class process_data_functions:
	def __init__(self):
		pass

	def _convert_df_to_gdf(self, df):
		"""Convert the original dataframe into geodataframe and convert the lat and long location for each geohash into a Point
		
		Args:
		df (dataframe) - the original dataframe containing information on rsrp, population, and both lat and long for each geohash
	
		Returns:
		geom_gdf (geodataframe) - the converted dataframe ready for further data cleansing
 		"""
		geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
		geom_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=4326)
		geom_gdf.drop(columns=['latitude', 'longitude'], inplace=True)

		return geom_gdf

	def _create_polygons_from_points(self, points):
		"""Function to generate geohash Polygon from each geoash Points

		Args:
		points (Points) - every Points from the geohash

		Returns:
		poly (Polygons) - the geohash's Polygons
		"""
		poly = Polygon([[p.x, p.y] for p in points])

		return poly
	
	def _clean_df(self, df):
		"""Cleans the dataframe from any missing values and geohash locations that doesn't have any population

		Args:
		df (dataframe) - the original dataframe containing information on rsrp, population, and both lat and long for each geohash

		Return:
		df (dataframe) - the cleaned dataframe
		"""
		df.dropna(inplace=True)
		df['population'] = df['population'].apply(lambda val: np.round(val))
		df = df.loc[df['population'] > 0, :]

		return df

	def _signal_strength(self, gdf):
		"""Determine signal strength categorizations based on the RSRP

		Args:
		gdf (geodataframe) - a geodataframe containing the rsrp values

		Returns:
		signal_strength (array of string) - the signal strength categorizations based on the RSRP
		"""
		signal_strength = gdf['rsrp_final'].apply(lambda val: 'Poor' if val < -120 \
									else 'Fair' if val < -106 \
									else 'Good' if val < -90 \
									else 'Excellent').values
		
		return signal_strength

	def process_df_to_gdf(self, df, operator):
		"""Process the original dataframe into geodataframe that are ready to be used for further analysis

		Args:
		df (dataframe) - the original dataframe containing information on rsrp, population, and both lat and long for each geohash
		operator (string) - the name of the operator that wanted to be analyzed

		Returns:
		gdf_polygons (geodataframe) - the processed original dataframe ready to be used for further analysis
		"""
		df = self._clean_df(df)
		gdf = self._convert_df_to_gdf(df)
		gdf_op = gdf.loc[gdf['carrier'] == operator, :]

		gdf = gdf_op.groupby('geohash')['rsrp_final'].mean().reset_index()
		gdf['population'] = gdf_op.groupby('geohash')['population'].mean().values
		
		gdf_polygons = gdf_op.groupby(['geohash', 'kecamatan'])['geometry'] \
					.agg(self._create_polygons_from_points) \
						.reset_index() \
							.sort_values(by='geohash')
 
		gdf_avg_rsrp = gdf_op.groupby(['geohash', 'kecamatan'])[['rsrp_final']] \
					.mean() \
						.reset_index() \
							.sort_values(by='geohash')

		gdf_avg_pop = gdf_op.groupby(['geohash', 'kecamatan'])[['population']] \
					.mean() \
						.reset_index() \
							.sort_values(by='geohash')
		
		gdf_polygons['rsrp'] = gdf_avg_rsrp['rsrp_final']
		gdf_polygons['signal_strength'] = self._signal_strength(gdf_polygons)
		gdf_polygons['population'] = gdf_avg_pop['population']
		gdf_polygons['kecamatan'] = gdf_polygons['kecamatan'].apply(lambda val: val.title())
		# gdf_polygons.rename(columns={'geohash':'geohash7'}, inplace=True)
		gdf_polygons.crs = "EPSG:4326"

		return gdf_polygons