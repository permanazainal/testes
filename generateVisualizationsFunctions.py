import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from shapely.ops import unary_union

from warnings import simplefilter
simplefilter('ignore')

class generate_visualizations_functions:
    def __init__(self):
        return

    def _set_figure(self, fig, title, height, width):
        """Sets the visualization title, height, width, and overall appearence
        
        Args:
        fig (Figure) - the visualization Figure object
        title (string) - the title for the visualization
        height (float) - the height of the visualization
        width (float) - the width of the visualization

        Returs:
        fig (Figure) - the refurbished visualization
        """
        fig.update_layout(title=title, title_font_size=20)
        fig.update_layout(
            font=dict(
                family="Courier",
                size=20, 
                color="black"
            ))
        fig.update_xaxes(linewidth=2, tickfont_size=20, title_font_size=25)
        fig.update_yaxes(tickfont_size=20,title_font_size=25)
        fig.update_layout(width=width, height=height)
        
        return fig
    
    def generate_kecamatan_polygons(self, gdf):
        """Generate the Polygons for each kecamatan and calculate the average RSRP on each kecamatan

        Args:
        gdf (geodataframe) - the geodataframe used for the analysis

        Returns:
        gdf_kec (geodataframe) - a geodataframe containing information on the Polygons and the average RSRP on each kecamatan
        """
        kec_polygons = []
        kecamatan = gdf['kecamatan'].unique()

        for kec in kecamatan:
            kec_geohash_poly = gdf.loc[gdf['kecamatan'] == kec, 'geometry'].values
            kec_poly = unary_union(kec_geohash_poly)
            kec_polygons.append(kec_poly)

        gdf_kec = gpd.GeoDataFrame({
            'kecamatan':kecamatan, 
            'geometry':kec_polygons
        })
        
        kec_rsrp_mean = gdf.groupby('kecamatan')['rsrp_final'].mean().reset_index()
        gdf_kec = gdf_kec.merge(kec_rsrp_mean, on='kecamatan')
        
        return gdf_kec

    def visualize_on_map(self, gdf, index_col, color_col, fig_name=None, title=None, height=1000, width=1000):
        """Visualize the geodataframe onto a map

        Args:
        gdf (geodataframe) - the geodataframe containing information desired to visualize
        index_col (string) - the name of the column on the geodataframe used for the main attributes desired to visualize
        color_col (string) - the name of the column on the geodataframe used for the color attributes, usually the the metrics desired to visualize
        fig_name (string) - the name of the saved visualized Figure object, providing this string means that the Figure object will be saved
        title (string) - the title for the visualization
        height (float) - the height of the visualization
        width (float) - the width of the visualization

        Returns:
        fig (Figure) - the visualization Figure object
        """
        fig = px.choropleth_mapbox(
            gdf.set_index(index_col),
            geojson=gdf.geometry,
            locations=gdf.index,
            color=color_col,
            center=dict(lon=gdf.centroid.x.mean(), 
                        lat=gdf.centroid.y.mean()),
            mapbox_style='open-street-map',
            zoom=9.5
        )
        fig = self._set_figure(fig, title, height, width)
        if fig_name != None:
            fig.write_image(f"Visualizations/{fig_name}.png")
        
        return fig

    def visualize_bar(self, data, x, y, fig_name=None, color_col=None, text_col=None, title=None, category_orders=None, color_discrete_map=None, height=1000, width=1000):
        """Create a bar chart from the data

        data (dataframe/geodataframe) - a dataframe or geodataframe containing information desired to visualize
        x (string) - the name of the column on the data for the information on the x axis
        y (string) - the name of the column on the data for the information on the y axis
        fig_name (string) - the name of the saved visualized Figure object, providing this string means that the Figure object will be saved
        color_col (string) - the name of the column on the data used for the color attributes
        text_col (string) - the name of the column on the data used for the text attributes
        category_orders (dict) - the desired order for the value on the x or y axis, for example {'x':['a', 'b', 'c']}
        color_discrete_map (dict) - the desired color the the value on the x or y axis, for example {'a':'red', 'b':'blue'}
        title (string) - the title for the visualization
        height (float) - the height of the visualization
        width (float) - the width of the visualization

        Returns:
        fig (Figure) - the visualization Figure object
        """
        fig = px.bar(
            data,
            x=x,
            y=y,
            color=color_col,
            text=text_col,
            color_discrete_map=color_discrete_map,
            category_orders=category_orders
        )
        fig = self._set_figure(fig, title, height, width)
        if fig_name != None:
            fig.write_image(f'Visualizations/{fig_name}.png')
        
        return fig

    def visualize_box(self, data, x, y, fig_name=None, color_col=None, title=None, category_orders=None, color_discrete_map=None, height=1000, width=1000):
        fig = px.box(
            data,
            x=x,
            y=y,
            color=color_col,
            color_discrete_map=color_discrete_map,
            category_orders=category_orders
        )

        mean_value = data.groupby(x)[y].mean().reset_index()
        mean_trace = go.Scatter(x=mean_value[x].values, y=mean_value[y].values, mode='lines', name='Mean')
        fig.add_trace(mean_trace)

        fig = self._set_figure(fig, title, height, width)
        if fig_name != None:
            fig.write_image(f'Visualizations/{fig_name}.png')
        
        return fig