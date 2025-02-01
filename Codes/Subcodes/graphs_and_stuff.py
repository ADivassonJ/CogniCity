import plotly.graph_objects as go
from scipy.spatial import Voronoi
import numpy as np

def plot_voronoi_on_map(vor, area_name):
    """
    Genera un gráfico interactivo del diagrama de Voronoi sobre un mapa.
    
    Parameters:
    vor : scipy.spatial.Voronoi
        Objeto del diagrama de Voronoi calculado.
    area_name : str
        Nombre del área o territorio para el título del gráfico.
    """
    # Crear la figura de Plotly
    fig = go.Figure()
    
    # Dibujar bordes de las regiones de Voronoi
    for region in vor.regions:
        if not -1 in region and len(region) > 0:  # Filtrar regiones válidas
            polygon = [vor.vertices[i] for i in region]
            fig.add_trace(go.Scattergeo(
                lon=[p[0] for p in polygon],  # Longitudes
                lat=[p[1] for p in polygon],  # Latitudes
                mode='lines',
                line=dict(width=1, color='blue'),
                fill='toself',
                fillcolor='rgba(0,128,0,0.3)'  # Color con transparencia
            ))
    
    # Configuración del mapa base
    fig.update_geos(
        projection_type="natural earth",
        showcoastlines=True,  # Mostrar costas
        coastlinecolor="Gray",
        showland=True,
        landcolor="lightgreen",
        showcountries=True,
        countrycolor="Black"
    )
    
    # Configuración de la figura
    fig.update_layout(
        title=f"Diagrama de Voronoi sobre {area_name}",
        geo=dict(
            resolution=50,  # Detalle del mapa
            fitbounds="locations"
        )
    )
    
    # Mostrar la figura
    fig.show()