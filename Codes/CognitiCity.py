import os
from pathlib import Path
import pandas as pd

study_area = 'Kanaleneiland'
main_path = Path(__file__).resolve().parent.parent 
subcodes_path = main_path / 'Subcodes'
archetypes_path = main_path / 'Archetypes'
data_path = main_path / 'Data'


population = 100

try:
    a_archetypes = pd.read_excel(archetypes_path / 'a_archetypes.xlsx')
except Exception as e:
    print('a_archetypes.xlsx not found')

try:
    h_archetypes = pd.read_excel(archetypes_path / 'h_archetypes.xlsx')
except Exception as e:
    print('h_archetypes.xlsx not found')
    
try:
    s_archetypes = pd.read_excel(archetypes_path / 's_archetypes.xlsx')
except Exception as e:
    print('s_archetypes.xlsx not found')
    
print(a_archetypes)
print(h_archetypes)
print(s_archetypes)