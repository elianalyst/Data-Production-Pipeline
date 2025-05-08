#Import
import logging
import pandas as pd

#Create function for main.py integration

def filter_and_clean(airport):
    #Final script for filtering including informational logging
    airport=pd.read_csv('/Users/elineiman/GIT/d602-deployment-task-2/MLProject/Data/lax_airport_data')

    lax=airport[airport['ORG_AIRPORT']=='LAX']
    logging.info("Data Filtered Succesfully To LAX")
    logging.info(f"LAX Data Frame Shape: {lax.shape}")

    #Final script for cleaning including informational logging

    # Missing Values
    lax.dropna(inplace=True)
    total_missing=lax.isnull().sum()
    logging.info(f"Missing Values: {total_missing}")

    #Change data types
    lax['DEST_AIRPORT'] = lax['DEST_AIRPORT'].astype(str)
    lax['DEPARTURE_TIME'] = lax['DEPARTURE_TIME'].astype(int)
    lax['DEPARTURE_DELAY'] = lax['DEPARTURE_DELAY'].astype(int)
    lax['ARRIVAL_TIME'] = lax['ARRIVAL_TIME'].astype(int)
    lax['ARRIVAL_DELAY'] = lax['ARRIVAL_DELAY'].astype(int)
    logging.info(f"Confirm Data Types: {lax.dtypes}")

    #lax.to_csv('lax_airport_data')

    return lax
