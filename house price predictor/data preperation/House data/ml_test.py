import math

import pandas as pd

HMLandRegistryData = pd.read_csv("hm_land_registry_data_new.csv")



for HMData in HMLandRegistryData.iterrows():
    if math.isnan(HMData[1]["Floor Area"]):
        HMLandRegistryData = HMLandRegistryData.drop(HMData[1].name)

HMLandRegistryData.to_csv("train.csv", index=False)
