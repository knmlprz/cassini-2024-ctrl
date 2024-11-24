#!/usr/bin/env python3

import osmnx as ox
import matplotlib.pyplot as plt

# Step 1: Pobierz dane o sieci ulic dla Rzeszowa
city_name = "Rzeszów, Poland"

print(f"Pobieranie danych o sieci ulic dla {city_name}...")
graph = ox.graph_from_place(city_name, network_type="all")

# Step 2: Wyświetl podstawowe informacje o grafie
print(f"Graf sieci ulic zawiera:")
print(f"- Węzły: {len(graph.nodes)}")
print(f"- Krawędzie: {len(graph.edges)}")

# Step 3: Rysuj siatkę ulic
print("Rysowanie siatki ulic...")
fig, ax = ox.plot_graph(graph, node_size=5, edge_linewidth=0.7, bgcolor="white", show=False, close=False)

# Step 4: Oblicz współrzędne graniczne
nodes, _ = ox.graph_to_gdfs(graph, nodes=True)
min_lat, max_lat = nodes['y'].min(), nodes['y'].max()
min_lon, max_lon = nodes['x'].min(), nodes['x'].max()

bounding_box_coordinates = {
    "top_left": (max_lat, min_lon),     # Lewy górny róg
    "top_right": (max_lat, max_lon),    # Prawy górny róg
    "bottom_left": (min_lat, min_lon),  # Lewy dolny róg
    "bottom_right": (min_lat, max_lon)  # Prawy dolny róg
}

# Wyświetl współrzędne rogów
print("Koordynaty rogów obserwowanego obszaru:")
for corner, coords in bounding_box_coordinates.items():
    print(f"{corner}: x={coords[0]}, y={coords[1]}")

# Step 5: Pobierz i rysuj POIs
print("Pobieranie punktów zainteresowania (POI)...")
tags = {
    "amenity": ["hospital", "fire_station", "shelter"],
    "military": ["bunker"]
}
pois = ox.geometries_from_place(city_name, tags)

# Kolory dla typów POI
colors = {
    "hospital": "red",
    "fire_station": "blue",
    "shelter": "green",
    "bunker": "orange"
}

print("Rysowanie punktów zainteresowania...")
for key, color in colors.items():
    filtered_pois = pois[pois["amenity"].eq(key) | pois["military"].eq(key)]
    if not filtered_pois.empty:
        filtered_pois.plot(ax=ax, color=color, label=key, markersize=20)

# Dodaj legendę i tytuł
plt.legend(title="POI Types")
plt.title(f"Street Network and POIs in {city_name}")
plt.grid(True)

# Wyświetl mapę
plt.show()

# Step 6: Zapisz współrzędne graniczne do pliku
output_file = "bounding_box_coordinates.txt"
with open(output_file, "w") as file:
    file.write("Koordynaty rogów obserwowanego obszaru:\n")
    for corner, coords in bounding_box_coordinates.items():
        file.write(f"{corner}: x={coords[0]}, y={coords[1]}\n")

print(f"Współrzędne zapisano pomyślnie do pliku {output_file}.")

# Step 7: Zapisz graf do pliku (opcjonalnie)
graph_output_file = "rzeszow_street_network.graphml"
print(f"Zapisywanie grafu do pliku: {graph_output_file}...")
ox.save_graphml(graph, filepath=graph_output_file)
print(f"Graf zapisano pomyślnie do pliku {graph_output_file}.")
