import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import geopandas as gpd
import osmnx as ox

# ============================================================
# 1. DESCARGAR GEOMETRÍAS
# ============================================================

print("Cargando datos geográficos...")

utrecht_boundary = ox.geocode_to_gdf("Utrecht, Netherlands")

oog_in_al = ox.geocode_to_gdf(
    "Oog in Al, Utrecht, Netherlands"
)

kanaleneiland = ox.geocode_to_gdf(
    "Kanaleneiland, Utrecht, Netherlands"
)

# ============================================================
# 2. REDES WALK + DRIVE
# ============================================================

print("Descargando red peatonal...")
G_walk = ox.graph_from_place(
    "Utrecht, Netherlands",
    network_type="walk",
    simplify=True
)

print("Descargando red viaria...")
G_drive = ox.graph_from_place(
    "Utrecht, Netherlands",
    network_type="drive",
    simplify=True
)

_, walk_edges = ox.graph_to_gdfs(G_walk)
_, drive_edges = ox.graph_to_gdfs(G_drive)

# ============================================================
# 3. REPROYECCIÓN
# ============================================================

crs_target = "EPSG:3857"

utrecht_boundary = utrecht_boundary.to_crs(crs_target)
oog_in_al = oog_in_al.to_crs(crs_target)
kanaleneiland = kanaleneiland.to_crs(crs_target)

walk_edges = walk_edges.to_crs(crs_target)
drive_edges = drive_edges.to_crs(crs_target)

# ============================================================
# 4. ZOOM DEL MAPA PRINCIPAL
# ============================================================

combined_bounds = gpd.GeoSeries([
    oog_in_al.geometry.iloc[0],
    kanaleneiland.geometry.iloc[0]
]).total_bounds

minx, miny, maxx, maxy = combined_bounds

padding = 600

zoom_xlim = (
    minx - padding,
    maxx + padding
)

zoom_ylim = (
    miny - padding,
    maxy + padding
)

# ============================================================
# 5. FIGURA PRINCIPAL
# ============================================================

fig, ax = plt.subplots(figsize=(10, 10))

ax.set_facecolor("#f7f7f7")

# ============================================================
# 6. RED DRIVE
# ============================================================

drive_edges.plot(
    ax=ax,
    linewidth=0.8,
    color="#b5b5b5",
    alpha=0.75,
    zorder=1
)

# ============================================================
# 7. RED WALK
# ============================================================

walk_edges.plot(
    ax=ax,
    linewidth=0.35,
    color="#707070",
    alpha=0.65,
    zorder=2
)

# ============================================================
# 8. DISTRITOS
# ============================================================

oog_in_al.plot(
    ax=ax,
    facecolor="#357abd",
    edgecolor="#1a4f85",
    alpha=0.45,
    linewidth=2,
    zorder=4,
    label="Oog in Al"
)

kanaleneiland.plot(
    ax=ax,
    facecolor="#e67e22",
    edgecolor="#8b4513",
    alpha=0.45,
    linewidth=2,
    zorder=4,
    label="Kanaleneiland"
)

# ============================================================
# 9. ZOOM
# ============================================================

ax.set_xlim(zoom_xlim)
ax.set_ylim(zoom_ylim)

# ============================================================
# 10. ETIQUETAS
# ============================================================

ax.text(
    oog_in_al.geometry.centroid.x.iloc[0],
    oog_in_al.geometry.centroid.y.iloc[0],
    "Oog in Al",
    fontsize=12,
    fontweight="bold",
    ha="center",
    color="#1a365d",
    zorder=6
)

ax.text(
    kanaleneiland.geometry.centroid.x.iloc[0],
    kanaleneiland.geometry.centroid.y.iloc[0],
    "Kanaleneiland",
    fontsize=12,
    fontweight="bold",
    ha="center",
    color="#7b241c",
    zorder=6
)

# ============================================================
# 11. TÍTULO
# ============================================================


ax.set_axis_off()

# ============================================================
# 12. INSET — PANEL INDEPENDIENTE
# ============================================================

ax_inset = ax.inset_axes(
    [0.64, 0.64, 0.32, 0.32],
    zorder=20
)

# ============================================================
# 13. FONDO SÓLIDO DEL PANEL
# ============================================================

# El propio patch del eje funciona como fondo
ax_inset.patch.set_visible(True)
ax_inset.patch.set_facecolor("white")
ax_inset.patch.set_alpha(1.0)

# ============================================================
# 14. LÍMITES DE UTRECHT
# ============================================================

utrecht_boundary.plot(
    ax=ax_inset,
    facecolor="#ffffff",
    edgecolor="black",
    linewidth=1.5,
    zorder=2
)

# ============================================================
# 15. DISTRITOS
# ============================================================

oog_in_al.plot(
    ax=ax_inset,
    facecolor="#357abd",
    edgecolor="#1a4f85",
    alpha=0.8,
    linewidth=1,
    zorder=3
)

kanaleneiland.plot(
    ax=ax_inset,
    facecolor="#e67e22",
    edgecolor="#8b4513",
    alpha=0.8,
    linewidth=1,
    zorder=3
)

# ============================================================
# 16. LÍMITES DEL INSET
# ============================================================

u_minx, u_miny, u_maxx, u_maxy = (
    utrecht_boundary.total_bounds
)

u_padding = 1000

ax_inset.set_xlim(
    u_minx - u_padding,
    u_maxx + u_padding
)

ax_inset.set_ylim(
    u_miny - u_padding,
    u_maxy + u_padding
)

# ============================================================
# 17. RECTÁNGULO DEL ÁREA DE ZOOM
# ============================================================

rect_width = zoom_xlim[1] - zoom_xlim[0]
rect_height = zoom_ylim[1] - zoom_ylim[0]

rect = Rectangle(
    (zoom_xlim[0], zoom_ylim[0]),
    rect_width,
    rect_height,
    linewidth=2,
    edgecolor="red",
    facecolor="red",
    alpha=0.18,
    zorder=5
)

ax_inset.add_patch(rect)

# ============================================================
# 18. MARCO DEL PANEL
# ============================================================

for spine in ax_inset.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_edgecolor("#222222")

# ============================================================
# 19. TÍTULO DEL PANEL
# ============================================================



ax_inset.set_axis_off()

# ============================================================
# 20. GUARDAR
# ============================================================

plt.tight_layout()

plt.savefig(
    "utrecht_distritos_walk_drive.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()