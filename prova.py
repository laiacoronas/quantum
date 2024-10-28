import json
import numpy as np
import csv
import os

def extract_atomic_numbers(compound):
    """Extract atomic numbers from the JSON structure."""
    return compound["atoms"]["element"]

def read_json_and_write_csv(input_directory, output_directory):
    """Read JSON files from input directory, compute Coulomb matrices, and save results in output directory."""
    output_csv_path = os.path.join(output_directory, 'coulomb_matrices.csv')
    with open(output_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        for filename in os.listdir(input_directory):
            if filename.endswith('.json'):
                json_file_path = os.path.join(input_directory, filename)
                
                with open(json_file_path, 'r') as f:
                    data = json.load(f)

                compounds = data["PC_Compounds"]

                for compound in compounds:
                    atoms = compound["coords"][0]
                    x = atoms["conformers"][0]["x"]
                    y = atoms["conformers"][0]["y"]
                    print(filename)
                    z = atoms["conformers"][0]["z"]
                    atomic_numbers = extract_atomic_numbers(compound)

                    coordinates = np.array([x, y, z]).T
                    num_atoms = len(coordinates)
                    coulomb_matrix = np.zeros((num_atoms, num_atoms))

                    for i in range(num_atoms):
                        for j in range(i, num_atoms):  # Only calculate upper triangular and diagonal
                            if i == j:
                                coulomb_matrix[i][j] = 0.5 * atomic_numbers[i] ** 2.4
                            else:
                                distance = np.linalg.norm(coordinates[i] - coordinates[j])
                                coulomb_matrix[i][j] = atomic_numbers[i] * atomic_numbers[j] / distance

                    # Flatten the matrix by taking the diagonal and upper triangular part in the right order
                    upper_triangular_and_diag = [filename]  # Start with the JSON filename
                    for i in range(num_atoms):
                        upper_triangular_and_diag.append(coulomb_matrix[i][i])  # Diagonal element
                        upper_triangular_and_diag.extend(coulomb_matrix[i, i+1:num_atoms])  # Upper triangle elements

                    csv_writer.writerow(upper_triangular_and_diag)

if __name__ == "__main__":
    input_directory = "C:/Users/lclai/Desktop/datasets_corrected/json/"
    output_directory = "C:/Users/lclai/Desktop/datasets_corrected"
    read_json_and_write_csv(input_directory, output_directory)
