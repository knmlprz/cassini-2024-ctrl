#!/usr/bin/env python3

from netCDF4 import Dataset

# Otwórz plik .nc
file_path = "iwv.nc"
dataset = Dataset(file_path, mode="r")

# Wyświetl dostępne zmienne
print("Zmiennie w pliku:")
print(dataset.variables.keys())

# Odczyt konkretnej zmiennej
variable_name = "IWV"
data = dataset.variables[variable_name][:]
print(f"Dane zmiennej '{variable_name}':", data)

# Zamknij plik
dataset.close()
