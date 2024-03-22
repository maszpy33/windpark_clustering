import os
import pandas as pd
import glob
import logging
import json
from datetime import datetime
import numpy as np
from functools import cache
from geopy.geocoders import Nominatim
import osm_manager as osmm

class WPClusterManager:
    def __init__(self):
        # Initialization parameters as previously defined
        self.SUB_DATA_PATH = "data_store/"
        self.DATA_PATH = os.path.abspath(self.SUB_DATA_PATH) + "/"
        self.SUB_DATA_PATH_MASTR = "data_store/MaStR_EDis_Erzeugungsanlagen_Data/"
        self.DATA_PATH_MASTR = os.path.abspath(self.SUB_DATA_PATH_MASTR) + "/"
        self.SUB_PLOT_SAVE_PATH = "plots/"
        self.PLOT_SAVE_PATH = os.path.abspath(self.SUB_PLOT_SAVE_PATH) + "/"
        self.SUB_EVALUATION_PATH = "data_store/cluster_eval.csv"
        self.EVALUATION_PATH = os.path.abspath(self.SUB_EVALUATION_PATH)
        self.logger = self.setup_logger()
        self.osm_manager = osmm.OSMDataManager()  # Initialize OSMDataManager
        self.columns_to_keep = ['Betriebs-Status', 'Energieträger', 'Gemeindeschlüssel', 'Inbetriebnahmedatum der Einheit',
                                'Registrierungsdatum der Einheit', 'latitude', 'longitude', 'Name des Windparks',
                                'Letzte Aktualisierung', 'Name des Anlagenbetreibers (nur Org.)',
                                'Name des Anschluss-Netzbetreibers', 'MaStR-Nr. des Anschluss-Netzbetreibers',
                                'Spannungsebene', 'Inbetriebnahmedatum der EEG-Anlage']
        self.SLICE_DATA = True
        self.PLOT_SAVE_RESOLUTION = 1200
        if self.SLICE_DATA:
            self.lat_min, self.lat_max = 52.7, 53.35  # Small slice north east of Germany
            self.lon_min, self.lon_max = 12, 12.8
        else:
            self.lat_min, self.lat_max = 47.3, 55.2  # All of Germany
            self.lon_min, self.lon_max = 5.5, 16.0

    def setup_logger(self):
        logger = logging.getLogger("info_logger")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler("info_logger.log")
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def log_string(self, string):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"{string}")

    def update_and_combine_csv_files(self):
        pattern = os.path.join(self.DATA_PATH_MASTR, '*.csv')
        csv_files = glob.glob(pattern)
        dataframes = [pd.read_csv(csv_file, delimiter=';', on_bad_lines='skip') for csv_file in csv_files]
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df

    def add_missing_coordinates(self, data):
        with open(self.DATA_PATH + 'city_data_cache.json') as f:
            new_location_data = json.load(f)

        new_location_data_cache = {}
        for index, (key, value) in enumerate(new_location_data.items()):
            new_location_data_cache[index] = {
                'Gemeindeschluessel': int(key),
                'Longitude': value['Coordinates'][1],
                'Latitude': value['Coordinates'][0],
                'CityName': value['CityName']
            }

        location_df = pd.DataFrame.from_dict(new_location_data_cache, orient='index')

        for row in location_df.itertuples():
            mask = (data['longitude'].isna() | data['latitude'].isna()) & (data['Gemeindeschlüssel'] == row.Gemeindeschluessel)
            data.loc[mask, 'longitude'] = row.Longitude
            data.loc[mask, 'latitude'] = row.Latitude

        return data

    def filter_dataframe(self, df):
        df['latitude'] = pd.to_numeric(df['latitude'].str.replace(',', '.'), errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'].str.replace(',', '.'), errors='coerce')
        df['Inbetriebnahmedatum der Einheit'] = pd.to_datetime(df['Inbetriebnahmedatum der Einheit'], dayfirst=True, errors='coerce')
        df.dropna(subset=['Inbetriebnahmedatum der Einheit'], inplace=True)
        df = self.add_missing_coordinates(df)

        df = df[self.columns_to_keep]
        df['Spannungsebene_Category'] = df['Spannungsebene'].apply(self.categorize_spannungsebene)
        df.to_csv('cleaned_data.csv', index=False)
        return df

    @cache
    def categorize_spannungsebene(self, row):
        if 'Niederspannung' in row:
            return 'Niederspannung'
        elif 'Mittelspannung' in row:
            return 'Mittelspannung'
        elif 'Hochspannung' in row:
            return 'Hochspannung'
        elif 'Höchstspannung' in row:
            return 'Höchstspannung'
        else:
            return 'Other'


    def add_missing_coordinates_to_df(self, df, city_data_cache):
        for key, value in city_data_cache.items():
            mask = df['Gemeindeschlüssel'] == key
            df.loc[mask, ['longitude', 'latitude']] = value['Coordinates']
        missing_coords_count = df[df['latitude'].isna() | df['longitude'].isna()].shape[0]
        print(f"Count of rows with missing coordinates: {missing_coords_count}")

    def generate_slice_coordinates(self):
        german_lat_min, german_lat_max = 47.3, 55.2
        german_lon_min, german_lon_max = 5.5, 16.0
        lat_min = random.uniform(german_lat_min, german_lat_max - 0.65)
        lat_max = lat_min + 0.65
        lon_min = random.uniform(german_lon_min, german_lon_max - 0.8)
        lon_max = lon_min + 0.8
        return round(lat_min, 2), round(lat_max, 2), round(lon_min, 2), round(lon_max, 2)

    def data_plot_ULevel(self, df, lat_min, lat_max, lon_min, lon_max):
        # Assuming categorize_spannungsebene method is already defined in the class

        if self.SLICE_DATA:
            df = df[(df['longitude'] >= lon_min) & (df['longitude'] <= lon_max) &
                    (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max)]

        color_map = {
            'Niederspannung': 'green',
            'Mittelspannung': 'blue',
            'Hochspannung': 'orange',
            'Höchstspannung': 'red',
            'Other': 'gray'
        }

        image_width_to_height_ratio = (lon_max - lon_min) / (lat_max - lat_min)
        fig, ax = plt.subplots(figsize=(8, 8 * image_width_to_height_ratio))
        m = Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max,
                    llcrnrlon=lon_min, urcrnrlon=lon_max, lat_ts=20, resolution='i', ax=ax)
        m.drawcoastlines()
        m.drawcountries()
        x, y = m(df.longitude, df.latitude)
        maker_size = 10 if df.shape[0] < 50 else 3

        scatter_legends = []
        for category, color in color_map.items():
            subset = df[df['Spannungsebene_Category'] == category]
            x_temp, y_temp = m(subset.longitude, subset.latitude)
            scatter = m.scatter(x_temp, y_temp, alpha=0.9, color=color, s=maker_size, label=f"[#{subset.shape[0]}] - {category}", edgecolors='black', linewidth=0.2)
            scatter_legends.append(scatter)

        plt.text(0.805, 0.015, f"Slice Coordinates:\nLat: {round(lat_min, 2)} - {round(lat_max, 2)}\nLon: {round(lon_min, 2)} - {round(lon_max, 2)}",
                 bbox=dict(facecolor='grey', alpha=0.3, pad=5), transform=plt.gca().transAxes, fontsize=8)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Wind Farm Data Slice - differentiated in voltage level [#{df.shape[0]}]')
        ax.legend(handles=scatter_legends, fontsize="x-small", loc='lower left')
        plot_id = str(uuid.uuid4())
        plt.savefig(self.PLOT_SAVE_PATH + f"wind_ms_all_data_plot_{plot_id}.png", dpi=self.PLOT_SAVE_RESOLUTION, bbox_inches='tight')
        plt.show()




if __name__ == "__main__":
    manager = WPClusterManager()
    combined_df = manager.update_and_combine_csv_files()

    num_rows_with_no_value = df["Name des Windparks"].isnull().sum()
    print(f"Number of rows with no value in the 'Name des Windparks' column: {num_rows_with_no_value}/{len(df)}")
