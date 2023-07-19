import csv
import re
import time

import requests
import pandas as pd
import json

floor_area = pd.read_csv("extracted_data.csv")
UKPostcodeLocations = pd.read_csv("ukpostcodes.csv")
HMLandRegistryData = pd.read_csv("hm_land_registry_data.csv")

for hm_data in HMLandRegistryData.iterrows():

    print(hm_data)

    postcode_data = floor_area[floor_area['Postcode'] == hm_data[1]["postcode"] and floor_area['Address'] == hm_data[1]["paon"]]

    print(postcode_data)


