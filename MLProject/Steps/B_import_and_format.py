#Import
import pandas as pd
import logging

#Create function for main.py integration

def import_and_format():
    # configure logger
    logname = "polynomial_regression.txt"
    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.info("Import and Format Log")

    #Import Data
    airport=pd.read_csv('T_ONTIME_REPORTING.csv')
    logging.info(f"Data imported successfully from {'/Users/elineiman/Desktop/WGU Data Sets/T_ONTIME_REPORTING.csv'}")
    logging.info(f"Imported data frame shape: {airport.shape}")

    #Format Data

    #rename columns
    airport.rename(columns={
    'DAY_OF_MONTH': 'DAY',
    'ORIGIN': 'ORG_AIRPORT',
    'DEST_AIRPORT_ID':'DEST_AIRPORT',
    'CRS_DEP_TIME': 'SCHEDULED_DEPARTURE',
    'DEP_TIME': 'DEPARTURE_TIME',
    'DEP_DELAY': 'DEPARTURE_DELAY',
    'CRS_ARR_TIME': 'SCHEDULED_ARRIVAL',
    'ARR_TIME': 'ARRIVAL_TIME',
    'ARR_DELAY': 'ARRIVAL_DELAY'
}, inplace=True)
    logging.info(f"Renamed Columns: {airport.columns.tolist()}")

    #drop columns
    airport.drop('ORIGIN_AIRPORT_ID', axis=1, inplace=True)
    airport.drop('DEST_CITY_NAME', axis=1, inplace=True)
    logging.info(f"Columns After Drop: {airport.columns.tolist()}")

    #airport.to_csv('aiport_csv', index=False)

    return airport

#import_and_format()

#DVC Command - Run from a .ipynb to create the metafile. Located in the Metafile folder in the DVC folder.
#airport.to_csv('temp.csv', index=False)
#!dvc add temp.csv
#verfiy file creation
#!ls -la
#Commit the metafile
#!git commit -m "DVC Metafile"
#Push the metafile
#!git push origin main