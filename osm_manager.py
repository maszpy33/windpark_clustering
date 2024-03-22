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

from IPython.display import clear_output


# TODO: implement function to get coordinates of wind turbines -> start with only E.DIS data -> implement cluster algo x

class OSMDataManager:
    def __init__(self):
        print("OSMDataManager init")
        # self.init_logger()

    # FIXME: outsource global path variables to env file
    data_save_path = "/Users/andreaszwikirsch/Desktop/01Uni/03BA WS22/python_scripts/overpass_api_env/data_store"
    data_store_backup_path = "/Users/andreaszwikirsch/Desktop/01Uni/03BA WS22/python_scripts/overpass_api_env/data_store/00Data_Store_TimeMachine"

    # CONTROLL FLOW VARS
    only_backup_current_data: bool = False # allows to only backup data but do not execute any other function
    save_json_file: bool = False
    download_test_area_data: bool = False
    # download_new_data is enough to disable/enable OSMDataManager funcs
    download_new_data: bool = True
    save_csv_file: bool = True
    plot_data: bool = False
    show_plot: bool = False

    # DATA PATH VARS
    # data_save_path = "/Users/andreaszwikirsch/Desktop/01Uni/03BA WS22/python_scripts/overpass_api_env/data_store"
    # data_store_backup_path = "/Users/andreaszwikirsch/Desktop/01Uni/03BA WS22/python_scripts/overpass_api_env/data_store/00Data_Store_TimeMachine"

    nominatim = Nominatim()
    overpass = Overpass()
    area_coordinates = []
    # area_name = "Nordrhein-Westfalen"
    area_name = "Hamburg"
    # area_name_list = ["Nordrhein-Westfalen", "Köln", "Wuppertal", "Duisburg", "Essen", "Oberhausen", "Gelsenkirchen", "Dortmund"]
    # area_name_list = ["Kreis Recklinghausen", "Ennepe-Ruhr-Kreis", "Kreis Unna", "Kreis Wesel", "Mülheim an der Ruhr", "Nordrhein-Westfalen", "Mettmann", "Köln", "Wuppertal", "Bottrop", "Dortmund", "Hamm", "Herne", "Gelsenkirchen", "Bochum", "Essen", "Mülheim", "Oberhausen", "Duisburg"]

    # Rheinisches Revier
    area_name_list = ["Düren", "Euskirchen", "Heinsberg", "Rhein-Erft-Kreis", "Rhein-Kreis Neuss", "Städteregion Aachen", "Mönchengladbach"]
    
    #area_name_list = ["Krefeld", "Duisburg", "Oberhausen", "Mülheim an der Ruhr", "Essen", "Bottrop", "Gelsenkirchen", "Bochum", "Herne", "Hattingen", "Witten", "Wuppertal", "Velbert", "Remscheid", "Leverkusen", "Bergisch Gladbach", "Köln", "Kerpen", "Langenfeld"]
    # area_name_list = ["Duisburg", "Düsseldorf", "Essen", "Krefeld", "Mühlheim an der Ruhr", "Oberhausen", "Wuppertal", "Neuss", "Mönchengladbach", "Heinsberg", "Düren", "Aachen", "Köln", "Bonn", "Euskirchen", "Düren"]

    scenario = "oK" # ohneKohle
    data_class_name = "power"
    timeout = 50

    # DATA VARIABLES
    data_vars = ['border_coordinates', 'tower', 'line', 'cable', 'plant']
    power_tower_list = []
    power_line_list = []
    power_cable_list = []
    power_plants_list = []
    alternative_element_list = []
    area_population_list = []
    wind_turbine_list = []
    population = 0
    population_area = 0

    # tuple with (id, lon, lat, tag1, tag2)
    area_border_coordinate_list = []

    # tuples of (name, id) both as string
    area_id_list = []

    ##############################################################################################
    # DATA MANAGEMENT + HELPER FUNCTIONS


    # DATE NOW HELPER
    def dt_now(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # dt_now = lambda self: datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    # TODO: exclude logger from both classes and create own logger class
    # LOGGER ERROR
    def error_logger(self, error_msg, func_name):
        loggerPyTool.setLevel(logging.ERROR)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        loggerPyTool.error(f'{error_msg} | in {func_name} | {now}')

    # LOGGER INFO
    def info_logger(self, info_msg, func_name):
        self.loggerPyTool.setLevel(logging.INFO)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        loggerPyTool.info(f'{info_msg} | in {func_name} | {now}')

    def init_logger(self):
        loggerPyTool = logging.getLogger("osmPythonToolLogger")
        loggerPyTool.setLevel(logging.INFO)
        osmPyTool_FileHandler = logging.FileHandler("osmPythonTool_Logger.log")
        loggerPyTool.addHandler(osmPyTool_FileHandler)
        loggerPyTool.info(f"init osmPyTool Logger | {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

        loggerTestArea = logging.getLogger("testArea_logger")
        test_logger_FileHandler = logging.FileHandler("testArea_Logger.log")
        loggerTestArea.setLevel(logging.INFO)
        loggerTestArea.addHandler(test_logger_FileHandler)


    # TIME MACHINE
    def data_store_timeMachine(self):
        # function that should copy alle files from data_store to a new folder called data_store_backup
        # this is to save the data in case of a new download
        print("Start TimeMachine....")
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        # os.makedirs(f"{self.data_store_backup_path}/{now}_{self.area_name}")
        shutil.copytree(f"{self.data_save_path}/{self.area_name}", f"{self.data_store_backup_path}/{now}_{self.area_name}")
        print("TimeMachine finished!")
    
    
    # SAVE TO CSV AND JSON
    def data_save_helper(self, json_data: dict, filterd_list: list, file_name: str):
        # save data as json
        if self.save_json_file:
            dir_exists = os.path.exists(f"{self.data_save_path}/{self.area_name.lower()}/json_raw_data_files")
            if not dir_exists:
                os.makedirs(f"{self.data_save_path}/{self.area_name.lower()}/json_raw_data_files")

            with open(f"{self.data_save_path}/{self.area_name.lower()}/json_raw_data_files/{file_name.lower()}.json", "w") as save_file:
                json.dump(json_data, save_file)
                print(f"Saved as json file successfully!  check '{file_name.lower()}.json' file.")
                self.info_logger(f"saved new dataset for {self.area_name}_{file_name} as json file", "data_save_helper")

        # save data as csv
        if self.save_csv_file:
            # check if directory already exists
            dir_exists = os.path.exists(f"{self.data_save_path}/{self.area_name.lower()}")
            if not dir_exists:
                os.makedirs(f"{self.data_save_path}/{self.area_name.lower()}")

            # Keys: ['version', 'generator', 'osm3s', 'elements']
            df = pd.DataFrame(filterd_list) # filterd is a list of dics with keys ['type', 'id', 'lat', 'lon']
            
            try:
                # sort dataframe by way_id if way_id is in dataframe (line, substation)
                if "cable" in file_name or "substation" in file_name:
                    pass
                else:
                    df = df.sort_values("way_id")
            except KeyError as err:
                print(f'no way_id in dataset of file: {file_name}')

            df.to_csv(f'{self.data_save_path}/{self.area_name.lower()}/{file_name.lower()}.csv')
            print(f"Saved as csv file successfully!  check {file_name.lower()}.csv file.")
            self.info_logger(f"saved new dataset for {self.area_name}_{file_name} as csv file", "data_save_helper")


    # DOWNLOAD DATA
    def download_data(self, query: str, data_type) -> dict:
        response = self.overpass.query(query, timeout=self.timeout)
        json_data = response.toJSON()
        # self.info_logger(f"downloaded new dataset for {data_type} of {self.area_name}", "download_data")

        return json_data

    def query_osm_data(data_name, data_class_name="power"):
        query = f"""
        area[name="{area_name}"]->.a;
        (
        node["{data_class_name}"="{data_name}"](area.a);
        way["{data_class_name}"="{data_name}"](area.a);
        relation["{data_class_name}"="{data_name}"](area.a);
        );
        out body;
        >;
        out skel qt;
        """

        json_data = download_data(query, data_name)
        
        return json_data

    # PLOT
    def plot_func(self):
        print("PLOT FUNC")
        # grep csv files to plot data
        df_line = pd.read_csv(f'{self.data_save_path}/{self.area_name.lower()}/line_{self.area_name.lower()}.csv')
        df_coordinates = pd.read_csv(f'{self.data_save_path}/{self.area_name.lower()}/border_coordinates_{self.area_name.lower()}.csv')
        df_tower = pd.read_csv(f'{self.data_save_path}/{self.area_name.lower()}/tower_{self.area_name.lower()}.csv')
        df_substation = pd.read_csv(f"{self.data_save_path}/{self.area_name.lower()}/substation_{self.area_name.lower()}.csv")
        df_plants = pd.read_csv(f'{self.data_save_path}/{self.area_name.lower()}/plant_{self.area_name.lower()}.csv')
        df_cable = pd.read_csv(f"{self.data_save_path}/{self.area_name.lower()}/cable_{self.area_name.lower()}.csv")
        df_population = pd.read_csv(f"{self.data_save_path}/{self.area_name.lower()}/population_{self.area_name.lower()}.csv")

        plt.figure(figsize=(12, 9))

        plt.scatter(df_coordinates['lon'], df_coordinates['lat'], color = "black", s=2, label=f"{self.area_name} border")
        # plt.scatter(df_tower['lon'], df_tower['lat'], color = "blue", s=3, label="Tower")
        plt.scatter(df_substation['lon'], df_substation['lat'], color = "orange", s=3, label="Substation")
        plt.scatter(df_line['lon'], df_line['lat'], color = "red", s=0.1, label="Line")
        # plt.scatter(df_cable['lon'], df_cable['lat'], color = "yellow", s=0.1, label="Cable")
        # plt.scatter(df_plants['lon'], df_plants['lat'], color = "green", s=10, label="Plant")
        # plt.scatter(x_alternative, y_alternative, color = "black", s=5, label="Alternative Line")

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(loc="lower right")
        plt.title(f"{self.area_name} Power Grid - Population: {self.calculate_population()}")

        plt.savefig(f"{self.data_save_path}/{self.area_name.lower()}/{self.area_name.lower()}_plot.png", dpi=3500)
        if self.show_plot:
            plt.show()


    ##############################################################################################
    # DATA ACQUIRING FUNCTIONS

    # AREA ID
    def get_area_id_by_name(self, area_name: str):
        # GET AREA IDs OF NRW
        nominatim = Nominatim()
        areaID = nominatim.query(f"{area_name}, Germany").areaId()
        self.area_id_list.append((area_name, areaID))


    # AREA COORDINATES
    def get_area_coordinates_by_area_name(self):
        # change target area with the var self.area_name
        query1 = f"""
        area[name="{self.area_name}"];
        (
        relation["type"="boundary"]["name"="{self.area_name}"];
        );
        (._;>;);
        out body;
        """

        # download data outsourced to reusable function
        json_data = self.download_data(query1, "coordinates")

        # data = json_data['elements']
        node_list = [elm for elm in json_data['elements'] if 'tags' not in elm.keys() and elm['type'] != 'way']
        way_list = [elm for elm in json_data['elements'] if 'tags' in elm.keys() and elm['type'] == 'way']

        for index, elm in enumerate(way_list):
            inline_id = 0
            for id in elm['nodes']:
                inline_id += 1
                for line_elm_dict in node_list:
                    if line_elm_dict['id'] == id:
                        try:
                            tag_keys = elm['tags'].keys()
                            line_elm_dict['way_id'] = elm['id']
                            line_elm_dict['lineIdKey'] = inline_id
                            for key in tag_keys:
                                line_elm_dict[f'{key}'] = elm['tags'][f'{key}']
                        except KeyError as err:
                            print(f"No tag: {err}")
                            # self.error_logger(err, "get_power_line_by_area_name")

                        # stop iteration if id is found
                        break

            print(f"{index+1}. {elm}")

        boarder_coords_list = node_list

        # filter data so only the border line of the area is left and no uneccessary 
        #filtered = [elm for elm in json_data['elements'] if 'tags' not in elm.keys() and elm['type'] != 'way']

        # self.area_coordinates = node_list

        # self.area_border_coordinate_list = node_list
        # for coords in filtered: self.area_border_coordinate_list.append(coords)

        # self.data_save_helper(json_data, node_list, f"border_coordinates_{self.area_name}")
        return node_list


    # LINE
    def get_power_line_by_area_name(self):
        query = f"""
        area[name="{self.area_name}"]->.a;
        (
        node["power"="line"](area.a);
        way["power"="line"](area.a);
        relation["power"="line"](area.a);
        );
        out body;
        >;
        out skel qt;
        """

        # query by overpass-turbo string
        json_data = self.download_data(query, "line")

        # seperate json_data in nodes with lon and lat, and ways with tag(number of cables, frequency, type, ref, voltage)
        node_list = [elm for elm in json_data['elements'] if elm['type'] == 'node']
        way_list = [elm for elm in json_data['elements'] if elm['type'] == 'way']

        # get tags and add them to the matching power line (matching by id) 
        # dataframe_list = []

        for index, elm in enumerate(way_list):
            inline_id = 0
            for id in elm['nodes']:
                inline_id += 1
                for line_elm_dict in node_list:
                    if line_elm_dict['id'] == id:
                        try:
                            tag_keys = elm['tags'].keys()
                            line_elm_dict['way_id'] = elm['id']
                            line_elm_dict['lineIdKey'] = inline_id
                            for key in tag_keys:
                                line_elm_dict[f'{key}'] = elm['tags'][f'{key}']
                        except KeyError as err:
                            print(f"No tag: {err}")
                            self.error_logger(err, "get_power_line_by_area_name")
                        
                        # stop iteration if id is found
                        break
                            
            print(f"{index+1}. {elm}")
        
        self.power_line_list = node_list

        self.data_save_helper(json_data, node_list, f"line_{self.area_name}")


    # WIND TURBINE DATA
    def get_germany_windturbin_data(self):
        query = f"""
        [out:json][timeout:300];
        {{geocodeArea:{self.area_name}}}->.searchArea;
        (
        nwr["power"="generator"]["generator:source"="wind"](area.searchArea);
        );
        (._;>;);
        out qt;
        """

        # query by overpass-turbo string
        json_data = self.download_data(query, "wind_turbine")

        # seperate json_data in nodes with lon and lat, and ways with tag(number of cables, frequency, type, ref, voltage)
        node_list = [elm for elm in json_data['elements'] if elm['type'] == 'node']
        way_list = [elm for elm in json_data['elements'] if elm['type'] == 'way']

        # get tags and add them to the matching power line (matching by id) 
        # dataframe_list = []

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
                            self.error_logger(err, "get_germany_windturbin_data")
                        
                        # stop iteration if id is found
                        break
                            
            print(f"{index+1}. {elm}")
        
        self.wind_turbine_list = node_list

        return node_list

        # self.data_save_helper(json_data, node_list, f"wind_turbine_{self.area_name}")



