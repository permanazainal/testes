import esda
import numpy as np
import pandas as pd
import libpysal as lps
from sklearn.cluster import DBSCAN

from warnings import simplefilter
simplefilter('ignore')

class analysis_functions:
    def __init__(self):
        pass
    
    def g_local_hotspot_analysis(self, gdf, column, n_neighbours):
        """Perform hotspot analysis to determine the hotspot and coldspot area using k-nearest-neigbours spatial weight

        Args:
        gdf (geodataframe) - contain informations for every geohash location
        column (string) - column in the geodataframe that used for the main metric
        n_neighbours (integer) - the number of neighbours used for the spatial weight

        Returns:
        gdf (geodataframe) - contain additional informations for the geohash categorization based on the hotspot analysis
        """
        w =  lps.weights.KNN.from_dataframe(gdf[['geometry', column]], k=n_neighbours)
        gi = esda.G_Local(gdf[column], w, permutations=999)        
        gdf['p-value'] = gi.p_sim
        gdf['z-value'] = gi.z_sim
        
        return gdf
    
    def determine_optimal_nearest_neighbours(self, gdf, column):
        """Determine the optimal number of nearest neighbours till 80% of the geohash recieve a p-value below 0.05

        Args:
        gdf (geodataframe) - contain informations for every geohash location
        column (string) - column in the geodataframe that used for the main metric

        Returns:
        gdf (geodataframe) - contain additional informations for the geohash categorization based on the hotspot analysis
        """
        store_n_neighbours = []
        store_p_val_under_005 = []
        for n in range(100, 600, 20):
            gdf_spot_n = self.g_local_hotspot_analysis(gdf, column, n)
            n_under_05 = len(gdf_spot_n.loc[gdf_spot_n['p-value'] <= 0.05, :])
            store_p_val_under_005.append(n_under_05)
            store_n_neighbours.append(n)

        df_p_value = pd.DataFrame({
            'n-neighbours':store_n_neighbours, 
            'p-value less than 0.05':store_p_val_under_005
        })
        df_p_value['p-value less than 0.05'] = df_p_value['p-value less than 0.05'] / len(gdf) * 100
        
        return df_p_value
    
    def _generate_geohash_centroid(self, gdf_hotspot):
        """Determine the center of each geohash location that are categorized as a hotspot

        Args:
        gdf_hotspot (geodataframe) - contain informations specifically for geohash locations that are categorized as a hotspot

        Returns:
        hotspot_centroid (array of Point) - the center for each geohash locations that are categorized as a hotspot
        """
        gdf_hotspot = gdf_hotspot.to_crs(epsg = 24547)
        hotspot_centroid = gdf_hotspot['geometry'].centroid.values.to_crs(epsg = 3857)
        hotspot_centroid = np.array([[point.x, point.y] for point in hotspot_centroid])      
    
        return hotspot_centroid
        
    
    def _dbscan_clustering(self, gdf_hotspot, hotspot_centroid, range_of_area, n_neighbours):
        """Perform DBSCAN clustering to determine areas within a specific range of area that has the desired number of neighbours 

        Args:
        gdf_hotspot (geodataframe) - contain informations specifically for geohash locations that are categorized as a hotspot
        hotspot_centroid (array of Point) - the center for each geohash locations that are categorized as a hotspot
        range_of_area (float) - the range of area that is desired
        n_neighbours (integer) - the number of desired neighbours within the specific desired range of area

        Returns:
        gdf_hotspot (geodataframe) - contain additional informations for the geohash locations that are the desired area
        labels (array of integer) - numbers that represents whether the geohash locations are the desired area, -1 means that the geohash
                                    location is not the desired area, 0 means that the geohash location is the desired area
        """
        clustering = DBSCAN(eps=range_of_area, min_samples=n_neighbours).fit(hotspot_centroid)
        gdf_hotspot['cluster'] = clustering.labels_
        labels = np.unique(clustering.labels_)

        return (gdf_hotspot, labels)

    def _print_process(self, steps, n_neighbours, start_neighbours, stop_n_neighbour, labels):
        """Prints the important parameters as the result of searching for the most optimal number of neighbours for the deasired range of area"""
        step_text = f'Current Steps : {steps}'
        n_neighbours_text = f'Neighbours : {n_neighbours}'
        start_text = f'Start Neighbours : {start_neighbours}'
        stop_text = f'Stop Neighbours : {stop_n_neighbour}'
        unique_text = f'Unique Desired Area : {len(labels) - 1}'
        print(f'{step_text} | {n_neighbours_text} | {start_text} | {stop_text} | {unique_text}')

        return

    def _determine_start_neighbours(self, gdf_hotspot, range_of_area):
        """Determine the nearest number of neighbours that still resulted in no desired area

        Args:
        gdf_hotspot (geodataframe) - contain informations specifically for geohash locations that are categorized as a hotspot
        range_of_area (float) - the range of area that is desired

        Returns:
        start_neighbours (integer) - the nearest number of neighbours that still resulted in no desired area        
        """
        n_sites = len(gdf_hotspot)
        hotspot_centroid = self._generate_geohash_centroid(gdf_hotspot)
        steps = int(np.round(n_sites * 0.005))
        past_len_labels = 1
        for n_neighbours in range(n_sites, 1, -steps):
            _, labels = self._dbscan_clustering(gdf_hotspot, hotspot_centroid, range_of_area, n_neighbours)
            current_len_labels = len(labels)
    
            if past_len_labels != current_len_labels:
                break
            else:
                past_len_labels = current_len_labels
                
        start_neighbours = n_neighbours + steps

        return start_neighbours

    def _determine_desired_area(self, gdf_hotspot, start_neighbours, range_of_area=0.02):
        """Determine the maximum number of neighbours within the specific range of area

        Args:
        gdf_hotspot (geodataframe) - contain informations specifically for geohash locations that are categorized as a hotspot
        start_neighbours (integer) - the nearest number of neighbours that still resulted in no desired area     
        range_of_area (float) - the range of area that is desired
        
        Returns:
        gdf_current_desired_area (geodataframe) - contain informations for geohash locations that are the desired area
        """
        hotspot_centroid = self._generate_geohash_centroid(gdf_hotspot)
        stop_n_neighbour = 1
        steps = 5

        while steps > 0:
            for n_neighbours in range(start_neighbours, stop_n_neighbour - 1, -steps):
                gdf_current_desired_area, labels = self._dbscan_clustering(gdf_hotspot, hotspot_centroid, range_of_area, n_neighbours)
                current_len_labels = len(labels)
                
                if current_len_labels == 2:
                    stop_n_neighbour = n_neighbours
                    start_neighbours = n_neighbours + steps
                    break

            if current_len_labels == 1:
                start_neighbours = n_neighbours

            _ = self._print_process(steps, n_neighbours, start_neighbours, stop_n_neighbour, labels)

            steps -= 1

        return gdf_current_desired_area
    
    def find_n_desired_areas(self, gdf_spot, n_desired_areas=1, range_of_area=0.02):
        """Determine the n number of desired areas by implementing the defined functions

        Args:
        gdf_spot (geodataframe) - contain informations for every geohash location
        n_desired_areas (integer) - number of desired areas
        range_of_area (float) - the range of area that is desired

        Returns:
        gdf_desired_area (geodataframe) - contain informations for geohash locations that are the desired area
        """
        gdf_hotspot = gdf_spot.loc[gdf_spot['spot'] == 'hotspot', :]
        desired_geohash = np.array([])
        
        for _ in range(n_desired_areas):
            start_neighbours = self._determine_start_neighbours(gdf_hotspot, range_of_area)
            gdf_hotspot_desired_area = self._determine_desired_area(gdf_hotspot, start_neighbours, range_of_area)
            
            temp_desired_geohash = gdf_hotspot_desired_area.loc[gdf_hotspot_desired_area['cluster'] != -1, 'geohash'].values
            desired_geohash = np.concatenate((desired_geohash, temp_desired_geohash))
            
            gdf_hotspot = gdf_hotspot_desired_area.loc[gdf_hotspot_desired_area['cluster'] == -1, :]
            gdf_hotspot.drop(columns=['cluster'], inplace=True)
        
        gdf_desired_area = gdf_spot.loc[gdf_spot['spot'] == 'hotspot', :]
        gdf_desired_area['desired_area'] = gdf_desired_area['geohash'].apply(lambda val: True if val in desired_geohash else False)
        
        return gdf_desired_area

    def find_all_desired_areas(self, gdf_spot, range_of_area=0.02):
        """Determine all desired areas by implementing the defined functions

        Args:
        gdf_spot (geodataframe) - contain informations for every geohash location
        range_of_area (float) - the range of area that is desired

        Returns:
        gdf_hotspot_final (geodataframe) - contain informations for geohash locations that are the desired area
        """
        gdf_hotspot = gdf_spot.loc[gdf_spot['spot'] == 'hotspot', :]
        desired_geohash = np.array([])
        geohash_rank = {}
        
        i = 1
        while len(gdf_hotspot) != 0:
            start_neighbours = self._determine_start_neighbours(gdf_hotspot, range_of_area)
            gdf_hotspot_desired_area = self._determine_desired_area(gdf_hotspot, start_neighbours, range_of_area)
            
            temp_desired_geohash = gdf_hotspot_desired_area.loc[gdf_hotspot_desired_area['cluster'] != -1, 'geohash'].values
            desired_geohash = np.concatenate((desired_geohash, temp_desired_geohash))
            
            for geo in temp_desired_geohash:
                geohash_rank[geo] = i
            
            gdf_hotspot = gdf_hotspot_desired_area.loc[gdf_hotspot_desired_area['cluster'] == -1, :]
            gdf_hotspot.drop(columns=['cluster'], inplace=True)
        
            i += 1
        
        gdf_desired_area = gdf_spot.loc[gdf_spot['spot'] == 'hotspot', :]
        
        for geo in gdf_desired_area['geohash'].unique():
            if geo not in desired_geohash:
                geohash_rank[geo] = 0
        
        gdf_desired_area['rank_desired_area'] = gdf_desired_area['geohash'].apply(lambda key: geohash_rank[key])
        
        return gdf_desired_area   