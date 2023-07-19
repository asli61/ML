import csv
import re
import time

import requests
import pandas as pd
import json

floor_area = pd.read_csv("extracted_data.csv")
HMLandRegistryData = pd.read_csv("hm_land_registry_data.csv")


for _, HMLandData in HMLandRegistryData.iterrows():
    for _, floor_area_data in floor_area.iterrows():
        try:
            if (HMLandData["postcode"] == floor_area_data["Postcode"]) and (int(HMLandData["paon"]) == int(floor_area_data["House Number"])):
                HMLandRegistryData.at[HMLandData.name, "Floor Area"] = floor_area_data["Floor Area"]
                HMLandRegistryData.to_csv("hm_land_registry_data_new.csv", index=False)
                break
        except ValueError:
            # Handle the case where the value cannot be converted to an integer
            # You can choose to skip this iteration or perform alternative actions
            pass


