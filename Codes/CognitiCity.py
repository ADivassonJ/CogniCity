import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from Subcodes.Initialization import Archetype_documentation_initialization, Synthetic_population_initialization
pd.set_option('mode.chained_assignment', 'raise')  # Convierte el warning en error

### Main
def main():
    # Input
    population = 450
    study_area = 'Kanaleneiland'
    
    ## Code initialization
    # Paths initialization
    main_path = Path(__file__).resolve().parent.parent
    subcodes_path = main_path / 'Subcodes'
    archetypes_path = main_path / 'Archetypes'
    data_path = main_path / 'Data'
    results_path = main_path / 'Results'
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    
    # Archetype documentation initialization
    citizen_archetypes, family_archetypes, s_archetypes, cond_archetypes = Archetype_documentation_initialization(main_path, archetypes_path)
    
    # Geodata initialization
    ######################################################################################
    
    # Synthetic population initialization
    df_citizens, df_families = Synthetic_population_initialization(results_path, citizen_archetypes, family_archetypes, population, cond_archetypes, data_path)
    
if __name__ == '__main__':
    main()

