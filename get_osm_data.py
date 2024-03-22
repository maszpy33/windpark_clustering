import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import os
import shutil
import requests
import logging
from datetime import datetime

from OSMPythonTools.overpass import Overpass
from OSMPythonTools.overpass import Nominatim
from OSMPythonTools.overpass import overpassQueryBuilder

import logging


class OSMDataPipeline:

    def __init__(self, area_name="Hamburg"):
        self.area_name = area_name
        self.nominatim = Nominatim()
        self.overpass = Overpass()
        self.TIMEOUT = 300

    # DOWNLOAD DATA
    def download_data(self, query: str, data_type) -> dict:
        response = self.overpass.query(query, timeout=self.TIMEOUT)
        return response.toJSON()


    # WIND TURBINE DATA
    def get_germany_windturbin_data(self):
        query = f"""
        area[name="{self.area_name}"]->.a;
        (
        nwr["power"="plant"]["plant:source"="wind"];
        );
        out body;
        >;
        out skel qt;
        """

        json_data = self.download_data(query, "wind_turbine")

        node_list = [elm for elm in json_data['elements'] if elm['type'] == 'node']
        way_list = [elm for elm in json_data['elements'] if elm['type'] == 'way']

        for index, elm in enumerate(way_list):
            inline_id = 0
            for id in elm['nodes']:
                inline_id += 1
                for line_elm_dict in node_list:
                    if line_elm_dict['id'] == id:
                        try:
                            tag_keys = elm['tags'].keys()
                            line_elm_dict['way_id'] = elm['id']
                            line_elm_dict['windTurbineID'] = inline_id
                            for key in tag_keys:
                                line_elm_dict[f'{key}'] = elm['tags'][f'{key}']
                        except KeyError as err:
                            print(f"No tag: {err}")
                            # Implement error handling here (e.g., logging)
                        break  # stop iteration if id is found

        return node_list

    # WIND PARK DATA
    def get_germany_windpark_data(self):
        query = f"""
        [out:json][timeout:{self.TIMEOUT}];
        {{geocodeArea:{self.area_name}}}->.searchArea;
        (
        nwr["power"="plant"]["plant:source"="wind"](area.searchArea);
        );
        (._;>;);
        out center qt;
        """

        json_data = self.download_data(query, "wind_turbine")

        node_list = [elm for elm in json_data['elements'] if elm['type'] == 'node']
        way_list = [elm for elm in json_data['elements'] if elm['type'] == 'way']

        for index, elm in enumerate(way_list):
            inline_id = 0
            for id in elm['nodes']:
                inline_id += 1
                for line_elm_dict in node_list:
                    if line_elm_dict['id'] == id:
                        try:
                            tag_keys = elm['tags'].keys()
                            line_elm_dict['way_id'] = elm['id']
                            line_elm_dict['windTurbineID'] = inline_id
                            for key in tag_keys:
                                line_elm_dict[f'{key}'] = elm['tags'][f'{key}']
                        except KeyError as err:
                            print(f"No tag: {err}")
                            # Implement error handling here (e.g., logging)
                        break  # stop iteration if id is found

        return node_list



# Example usage
pipeline = OSMDataPipeline()
wind_turbine_data = pipeline.get_germany_windturbin_data()
print(wind_turbine_data)




# print("Init get_osm_data.py")


# nominatim = Nominatim()
# overpass = Overpass()
# # area_name = "Nordrhein-Westfalen"
# area_name = "Hamburg"
# TIMEOUT = 300


# # DOWNLOAD DATA
# def download_data(query: str, data_type) -> dict:
#     response = overpass.query(query, timeout=TIMEOUT)
#     json_data = response.toJSON()

#     return json_data


# # WIND TURBINE DATA
# def get_germany_windturbin_data():
#     query = f"""
#     [out:json][timeout:{TIMEOUT}];
#     {{geocodeArea:{self.area_name}}}->.searchArea;
#     (
#     nwr["power"="generator"]["generator:source"="wind"](area.searchArea);
#     );
#     (._;>;);
#     out qt;
#     """

#     # query by overpass-turbo string
#     json_data = download_data(query, "wind_turbine")

#     # seperate json_data in nodes with lon and lat, and ways with tag(number of cables, frequency, type, ref, voltage)
#     node_list = [elm for elm in json_data['elements'] if elm['type'] == 'node']
#     way_list = [elm for elm in json_data['elements'] if elm['type'] == 'way']

#     # get tags and add them to the matching power line (matching by id) 
#     # dataframe_list = []

#     for index, elm in enumerate(way_list):
#         inline_id = 0
#         for id in elm['nodes']:
#             inline_id += 1
#             for line_elm_dict in node_list:
#                 if line_elm_dict['id'] == id:
#                     try:
#                         tag_keys = elm['tags'].keys()
#                         line_elm_dict['way_id'] = elm['id']
#                         line_elm_dict['windTurbineID'] = inline_id
#                         for key in tag_keys:
#                             line_elm_dict[f'{key}'] = elm['tags'][f'{key}']
#                     except KeyError as err:
#                         print(f"No tag: {err}")
#                         error_logger(err, "get_germany_windturbin_data")
                    
#                     # stop iteration if id is found
#                     break
                        
#         print(f"{index+1}. {elm}")
    
#     wind_turbine_list = node_list

#     return node_list

