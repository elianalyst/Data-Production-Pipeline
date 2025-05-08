import os
import mlflow

from Steps.B_import_and_format import import_and_format
from Steps.C_filter_and_clean import filter_and_clean
from Steps.D_ml_experiment import ml_experiment

def main():
    #Step 1: Import and Format
    airport_data=import_and_format()
    print(f"Import and Format Complete: {airport_data.shape}")

    #Step 2: Filter and Clean
    lax_airport_data=filter_and_clean(airport_data)
    print(f"Filter and Clean Complete: {lax_airport_data.shape}")

    #Step 3: MLFLow Experiment
    ml_experiment(lax_airport_data)
    print("MLFlow Experiment Complete.")

if __name__ == "__main__":
    main()

