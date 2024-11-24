import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Wczytaj pliki NetCDF
iwv_data = xr.open_dataset("iwv.nc")
gifapar_data = xr.open_dataset("gifapar.nc")

# Wyświetl metadane
print(gifapar_data)

# Odczyt danych konkretnej zmiennej
variable_name = "GIFAPAR"
print(gifapar_data[variable_name])

# Ekstrakcja danych
iwv = iwv_data["IWV"].values
gifapar = gifapar_data["GIFAPAR"].values

# Kryteria ryzyka powodzi
iwv_threshold = 20  # kg/m²
gifapar_threshold = 0.4

# Maski
# land_mask = gifapar > 0.1 # Wyklucza wodę
land_mask = gifapar <= 0.1 # Wyklucza wodę
water_mask = ~land_mask  # Obszary wodne
# water_mask = gifapar <= 0.1
safe_mask = (iwv < iwv_threshold) & (gifapar > gifapar_threshold) & land_mask  # Bezpieczne tereny na lądzie
unsafe_mask = ~safe_mask & land_mask  # Niebezpieczne tereny na lądzie

# Tworzenie mapy kolorów
risk_map = np.zeros(iwv.shape + (3,), dtype=np.uint8)  # Pusta mapa RGB

# Przypisanie kolorów
risk_map[land_mask] = [150,75,0]
risk_map[water_mask] = [0, 0, 255]  # Niebieski dla wody
risk_map[safe_mask] = [0, 255, 0]  # Zielony dla bezpiecznych terenów
risk_map[unsafe_mask] = [255, 0, 0]  # Czerwony dla niebezpiecznych terenów


# Wizualizacja
plt.figure(figsize=(10, 8))
plt.imshow(risk_map)
plt.title("Flood Risk Map: Water (Blue), Safe (Green), Unsafe (Red)")
plt.axis("off")
plt.show()
