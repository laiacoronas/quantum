import h5py
from collections import Counter
import numpy as np
import csv


def atomic_numbers_to_formula(atomic_numbers):
    element_symbols = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
        11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
        
    }
    
    atomic_numbers_list = atomic_numbers.flatten().tolist()
    counter = Counter(atomic_numbers_list)
    formula = ''.join(f"{element_symbols[num]}{count if count > 1 else ''}" for num, count in sorted(counter.items()))
    return formula

file_path = '\\Users\\lclai\\Desktop\\qm7\\1000\\1000.hdf5'  
fDFT = h5py.File(file_path, 'r')


results = []


for key in fDFT.keys():
    data = fDFT[key]
    for subkey in data.keys():
        d = data[subkey]
        for subsubkey in d.keys():
            if 'atNUM' in subsubkey:
                atomic_numbers = d[subsubkey][:]
                formula = atomic_numbers_to_formula(atomic_numbers)
                results.append([key, subkey, formula])


output_csv_path = '\\Users\\lclai\\Desktop\\qm7\\chemical_formulas1000.csv'
with open(output_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Molecule_ID', 'Sub_ID', 'Chemical_Formula'])
    csvwriter.writerows(results)

print(f"Results have been saved to {output_csv_path}")
