import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import voronoi_diagram
from matplotlib.patches import Patch

# --------------------------------------------------------
# A) Cargar datos y FILTRAR solo PC_electric
# --------------------------------------------------------
study_case = "Annelinn"
scenario = "s4"                 # ajusta a "S0" si tu carpeta está en mayúsculas
archetype_folder = "PC_electric"  # carpeta del arquetipo dentro del escenario

if study_case == "Kanaleneiland":
    colors = {
        "s4": "#d0e0e3",
        "q1": "#a2c4c9",
        "s2": "#76a5af",
        "s1": "#45818e",
        "max": "#134f5c",
        "base": "#434343",
    }
elif study_case == "Annelinn":
        colors = {
            "s4": "#ead1dc",
            "q1": "#d5a6bd",
            "s2": "#c27ba0",
            "s1": "#a64d79",
            "max": "#741b47",
            "base": "#434343",
        }
elif study_case == "Aradas":
        colors = {
            "s4": "#ffe599",
            "q1": "#ffd966",
            "s2": "#f1c232",
            "s1": "#bf9000",
            "max": "#7f6000",
            "base": "#434343",
        }

project_root = Path(r"C:/Users/asier.divasson/Documents/GitHub/CogniCity")

# Estructura: data/results/<scenario>/<archetype>/<archivo>
results_dir = project_root / "results" / scenario / study_case

# Nombre del archivo: <study_case>_schedule_vehicle.xlsx
excel_filename = f"{study_case}_schedule_vehicle_quantified_24.xlsx"
excel_path = results_dir / excel_filename

df = pd.read_excel(excel_path)

# 🔴 FILTRO CLAVE: solo vehículos eléctricos
df = df[df["archetype"] == "PC_electric"].copy()

# --------------------------------------------------------
# B) Seleccionar último time_slot por agent y day
# --------------------------------------------------------
ts = pd.to_datetime(df["time_slot"], format="%H:%M", errors="coerce")
df["time_slot_minutes"] = ts.dt.hour * 60 + ts.dt.minute

df_last_per_agent = (
    df.sort_values(
        by=["agent", "day", "time_slot_minutes"],
        ascending=[True, True, False]
    )
    .groupby(["agent", "day"], as_index=False)
    .first()
    .drop(columns=["time_slot_minutes"])
)

# --------------------------------------------------------
# C) MJ (campo mjkm) -> float
#    (aunque se llame mjkm, aquí son MJ)
# --------------------------------------------------------
df_last_per_agent["mjkm"] = (
    df_last_per_agent["mjkm"]
    .astype(str)
    .str.replace(",", ".", regex=False)
)
df_last_per_agent["mjkm"] = pd.to_numeric(
    df_last_per_agent["mjkm"], errors="coerce"
).fillna(0.0)

# --------------------------------------------------------
# D) Suma por node en MJ y conversión a kWh
# --------------------------------------------------------
df_mj_by_node = (
    df_last_per_agent
    .groupby("node", as_index=False)["mjkm"]
    .sum()
    .rename(columns={"mjkm": "MJ_total"})
)

# Conversión MJ -> kWh (1 kWh = 3.6 MJ)
df_mj_by_node["kWh_total"] = df_mj_by_node["MJ_total"] / 3.6

total_kwh = df_mj_by_node["kWh_total"].sum()

# Diccionario node -> kWh_total
kwh_map = dict(zip(df_mj_by_node["node"], df_mj_by_node["kWh_total"]))

# --------------------------------------------------------
# 1) Leer nodos y edges
# --------------------------------------------------------
path = f"C:/Users/asier.divasson/Documents/GitHub/CogniCity/data/{scenario}/{study_case}/population"
nodes = pd.read_csv(f"{path}/node_data_{study_case}.csv")   # lat, lon, i
edges = pd.read_csv(f"{path}/{study_case}_line_data.csv")   # i, j

# --------------------------------------------------------
# 2) Definir polígono (lat, lon) -> (lon, lat)
# --------------------------------------------------------

if study_case == "Kanaleneiland":
    #Kanaleneiland
    boundary_latlon = [  (52.07904398,	5.081736117),
                                (52.07624318,	5.08308264),
                                (52.06046958,	5.09756737),
                                (52.06021839,	5.097758556),
                                (52.06008988,	5.11164107),
                                (52.06328398,	5.113065093),
                                (52.06860149,	5.111588679),
                                (52.07642504,	5.109425399),
                                (52.07861645,	5.108711591),
                                (52.08034774,	5.107271173),
                                (52.08592257,	5.097037869),
                                (52.08498639,	5.096460351),
                                (52.08309467,	5.094751129),
                                (52.0803543, 	5.087985518),
                                (52.07904398,	5.081736117),
                                ]
elif study_case == "Annelinn":
    #Annelinn
    boundary_latlon = [(58.37779995285961, 26.737546920776367),
                        (58.38207378048632,  26.74806118011475),
                        (58.38095016884387,  26.753661632537845),
                        (58.380230379627854, 26.761858463287357),
                        (58.37991546722738,  26.76752328872681),
                        (58.37947683455617,  26.77179336547852),
                        (58.379218151193385, 26.773359775543216),
                        (58.37876826256608,  26.77522659301758),
                        (58.377767239785584, 26.77874565124512),
                        (58.377013642098476, 26.77923917770386),
                        (58.375427660052644, 26.784152984619144),
                        (58.37414532457297,  26.7826509475708),
                        (58.371591763372905, 26.782929897308353),
                        (58.37125427449941,  26.783938407897953),
                        (58.369139270733456, 26.78344488143921),
                        (58.368970514972645, 26.782200336456302),
                        (58.36281344471082,  26.78001165390015),
                        (58.359865199696884, 26.77775859832764),
                        (58.358199670019,    26.7762565612793),
                        (58.35478959051058,  26.7717719078064),
                        (58.35464327610064,  26.770827770233158),
                        (58.35328311497021,  26.765420436859134),
                        (58.35779452929785,  26.76052808761597),
                        (58.359280022547246, 26.759669780731205),
                        (58.357108030241015, 26.754348278045658),
                        (58.35555491754449,  26.746966838836673),
                        (58.355836283607296, 26.7467737197876),
                        (58.356140156436815, 26.746795177459717),
                        (58.357198063664704, 26.7476749420166),
                        (58.35798584632884,  26.749970912933353),
                        (58.35900993751445,  26.751451492309574),
                        (58.360337835698445, 26.751902103424076),
                        (58.36097926015119,  26.75168752670288),
                        (58.36248712436589,  26.7497992515564),
                        (58.36366861475146,  26.748619079589847),
                        (58.364827562136334, 26.74872636795044),
                        (58.3659752201055,   26.75001382827759),
                        (58.36697657745643,  26.750378608703613),
                        (58.368247922822114, 26.749413013458252),
                        (58.36886670266924,  26.74803972244263),
                        (58.36998047905436,  26.744391918182377),
                        (58.37151045798987,  26.741452217102054),
                        (58.37556008204928,  26.738491058349613),
                        (58.377168553744575, 26.737632751464847),
                        (58.3777421867492,   26.73758983612061),
                        ]
elif study_case == "Aradas":
    #Aradas
    boundary_latlon = [ (40.6260277,-8.6691095),
                    (40.6242125,-8.666836),
                    (40.6236329,-8.6659728),
                    (40.6234128,-8.6657849),
                    (40.6231059,-8.6656375),
                    (40.6225013,-8.6650637),
                    (40.6222507,-8.6646582),
                    (40.6220032,-8.663601),
                    (40.6216362,-8.6628016),
                    (40.6210708,-8.6622693),
                    (40.6202052,-8.6619314),
                    (40.61922,-8.6617375),
                    (40.6189017,-8.6617527),
                    (40.6186492,-8.6616989),
                    (40.6179653,-8.6614756),
                    (40.6175178,-8.661246),
                    (40.6171852,-8.6612232),
                    (40.6171653,-8.6610968),
                    (40.6173448,-8.6608471),
                    (40.6173608,-8.6607396),
                    (40.6172055,-8.6606055),
                    (40.6171618,-8.660501),
                    (40.6170468,-8.6603655),
                    (40.6140575,-8.6575236),
                    (40.613851,-8.6572765),
                    (40.6135878,-8.6568526),
                    (40.613393,-8.6566081),
                    (40.6130864,-8.6563332),
                    (40.6129204,-8.6560929),
                    (40.6127237,-8.6558836),
                    (40.6126708,-8.6557805),
                    (40.6126488,-8.6552235),
                    (40.6124499,-8.6548499),
                    (40.6123021,-8.6546783),
                    (40.6120758,-8.6545599),
                    (40.6119478,-8.6545386),
                    (40.6100666,-8.6545613),
                    (40.6096812,-8.6543448),
                    (40.6089189,-8.6535312),
                    (40.6082786,-8.652988),
                    (40.6079532,-8.6523902),
                    (40.6076241,-8.6519091),
                    (40.6072771,-8.6514912),
                    (40.60668,-8.6504698),
                    (40.6065747,-8.6504181),
                    (40.6063881,-8.6502435),
                    (40.6062332,-8.6500269),
                    (40.6054844,-8.6483856),
                    (40.6053114,-8.6480454),
                    (40.6052046,-8.6479189),
                    (40.6050178,-8.6476255),
                    (40.6045178,-8.6466887),
                    (40.6042572,-8.6463173),
                    (40.6040732,-8.6461176),
                    (40.6036906,-8.6458503),
                    (40.6035381,-8.6456551),
                    (40.6032476,-8.6454093),
                    (40.6030542,-8.6451902),
                    (40.6027228,-8.6447241),
                    (40.6025011,-8.6443065),
                    (40.6023171,-8.6440466),
                    (40.6022493,-8.6439981),
                    (40.6020254,-8.6439602),
                    (40.6013136,-8.6437384),
                    (40.6010098,-8.643593),
                    (40.6007614,-8.643566),
                    (40.6006448,-8.6435104),
                    (40.6003543,-8.6432293),
                    (40.5998495,-8.6423911),
                    (40.599316,-8.6417578),
                    (40.5991053,-8.6416322),
                    (40.5986892,-8.6416011),
                    (40.5984492,-8.6414653),
                    (40.5983617,-8.6413203),
                    (40.5983123,-8.6409809),
                    (40.5982499,-8.6408236),
                    (40.5980912,-8.640572),
                    (40.5979016,-8.6403789),
                    (40.5972284,-8.6400028),
                    (40.5969958,-8.6399705),
                    (40.5968975,-8.6399956),
                    (40.5964402,-8.6396011),
                    (40.5959139,-8.6389858),
                    (40.5954893,-8.6385972),
                    (40.5945613,-8.6379813),
                    (40.5937743,-8.6372131),
                    (40.5937678,-8.6366875),
                    (40.5937272,-8.6365929),
                    (40.5932003,-8.6359088),
                    (40.5929938,-8.6352565),
                    (40.5924366,-8.6339021),
                    (40.5923815,-8.6332999),
                    (40.5923066,-8.6329949),
                    (40.5921122,-8.6323842),
                    (40.5920556,-8.6320796),
                    (40.5918131,-8.6314779),
                    (40.591809,-8.6313249),
                    (40.5918757,-8.6309966),
                    (40.591891,-8.6305948),
                    (40.5917616,-8.6301748),
                    (40.5917761,-8.629417),
                    (40.5917162,-8.6291502),
                    (40.5916803,-8.6287324),
                    (40.5914751,-8.6282729),
                    (40.5914972,-8.6279974),
                    (40.5916615,-8.6276765),
                    (40.5916434,-8.6274363),
                    (40.5914849,-8.6269246),
                    (40.5883757,-8.6244006),
                    (40.5874733,-8.6235797),
                    (40.587295,-8.623363),
                    (40.587194,-8.6231671),
                    (40.5868211,-8.6223074),
                    (40.5866228,-8.622139),
                    (40.5864064,-8.6220873),
                    (40.5863086,-8.6217312),
                    (40.5862954,-8.6214654),
                    (40.586398,-8.6206219),
                    (40.5865712,-8.6199791),
                    (40.5867584,-8.6195026),
                    (40.5869593,-8.6191804),
                    (40.5870914,-8.6192253),
                    (40.5871382,-8.619014),
                    (40.5871432,-8.6186678),
                    (40.5871321,-8.618444),
                    (40.5871236,-8.618112),
                    (40.587257,-8.6180764),
                    (40.5895434,-8.6166483),
                    (40.5897483,-8.6166463),
                    (40.5920307,-8.6153937),
                    (40.5934133,-8.6147084),
                    (40.5939013,-8.6143535),
                    (40.5945506,-8.6140867),
                    (40.5960723,-8.6132704),
                    (40.5969792,-8.6135683),
                    (40.5977968,-8.6136263),
                    (40.6005637,-8.6142546),
                    (40.6006406,-8.6142345),
                    (40.6013907,-8.6145417),
                    (40.6019361,-8.6148515),
                    (40.6013476,-8.6157457),
                    (40.6011136,-8.6161966),
                    (40.6003201,-8.6181819),
                    (40.6052078,-8.6223089),
                    (40.6059547,-8.6230277),
                    (40.6062372,-8.6233187),
                    (40.6069917,-8.6242913),
                    (40.6089161,-8.6269292),
                    (40.6097193,-8.6277959),
                    (40.6101187,-8.6281644),
                    (40.6236293,-8.6395615),
                    (40.6238389,-8.6384898),
                    (40.6239732,-8.6383452),
                    (40.6241713,-8.6382761),
                    (40.6250741,-8.6388559),
                    (40.6249215,-8.6391849),
                    (40.6249961,-8.6394292),
                    (40.625771,-8.6411829),
                    (40.6261616,-8.6416184),
                    (40.6262771,-8.6417962),
                    (40.626503,-8.64201),
                    (40.6265785,-8.6422958),
                    (40.6266742,-8.6430671),
                    (40.6268177,-8.6434418),
                    (40.6269435,-8.6439459),
                    (40.6270621,-8.6446783),
                    (40.6274912,-8.6454454),
                    (40.6278877,-8.6464471),
                    (40.6278272,-8.6474518),
                    (40.6278976,-8.6478379),
                    (40.6279316,-8.6478432),
                    (40.6278874,-8.6485833),
                    (40.6278211,-8.648809),
                    (40.6270303,-8.6498279),
                    (40.6270053,-8.6500061),
                    (40.6269994,-8.6500623),
                    (40.6269936,-8.6501184),
                    (40.6269697,-8.650423),
                    (40.6268628,-8.6507406),
                    (40.6267409,-8.6508643),
                    (40.6265982,-8.6509256),
                    (40.6263341,-8.650965),
                    (40.6262965,-8.6511239),
                    (40.6262627,-8.6516432),
                    (40.626272,-8.6518649),
                    (40.6263288,-8.6521423),
                    (40.6265221,-8.6525869),
                    (40.6265496,-8.6526414),
                    (40.6267315,-8.6529555),
                    (40.6272151,-8.6545382),
                    (40.6272815,-8.6549749),
                    (40.6273554,-8.6561734),
                    (40.6273418,-8.6568172),
                    (40.6272684,-8.6578625),
                    (40.6271319,-8.658907),
                    (40.6268978,-8.6597033),
                    (40.6263523,-8.6621294),
                    (40.6264761,-8.6626895),
                    (40.6269026,-8.6639876),
                    (40.6270787,-8.664744),
                    (40.6271497,-8.6653192),
                    (40.6271319,-8.6657763),
                    (40.6270175,-8.6665162),
                    (40.6268942,-8.6669446),
                    (40.6267452,-8.6672412),
                    (40.6265183,-8.6674498),
                    (40.6260277,-8.6691095),
                ]

boundary_lonlat = [(lon, lat) for (lat, lon) in boundary_latlon]
boundary_poly = Polygon(boundary_lonlat)

# --------------------------------------------------------
# 3) Voronoi
# --------------------------------------------------------
points_geom = [Point(row["lon"], row["lat"]) for _, row in nodes.iterrows()]
multi_points = MultiPoint(points_geom)
voronoi_multi = voronoi_diagram(multi_points, envelope=boundary_poly, edges=False)

# --------------------------------------------------------
# 3.1) KDTree y labels Node_X
# --------------------------------------------------------
def to_node_label(x):
    s = str(x)
    return s if s.startswith("Node_") else f"Node_{s}"

nodes = nodes.copy()
nodes["node_label"] = nodes["i"].apply(to_node_label)

coords = nodes[["lon", "lat"]].to_numpy()
tree = cKDTree(coords)

# ------------------------------
# Valores de la muestra (reales)
vals_pos = df_mj_by_node.loc[df_mj_by_node["kWh_total"] > 0, "kWh_total"].to_numpy()

v_sample_min = float(vals_pos.min())       # mínimo de la muestra
v_sample_max = float(vals_pos.max())       # máximo de la muestra
vmin_sample = float(np.percentile(vals_pos, 25))  # Q1 de la muestra
vmax_sample = float(np.percentile(vals_pos, 75))  # Q3 de la muestra

# ------------------------------
# Valores manuales que metiste
vmin_manual = 33.4
vmax_manual = 314.3

def hex_to_rgb01(hex_color):
    """Convierte #RRGGBB a tupla RGB en [0,1]."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))


# Colores definidos por ti
COLOR_ZERO = hex_to_rgb01("#ffffff")   # kWh = 0

COLOR_MIN  = hex_to_rgb01("#ffffff")   # mínimo de la muestra (positivos)
COLOR_20   = hex_to_rgb01(colors.get("q1", "grey"))   # Q1 (lo llamabas 20%)
COLOR_80   = hex_to_rgb01(colors.get("max", "grey"))   # Q3 (lo llamabas 80%)
COLOR_MAX  = hex_to_rgb01("#000000")   # máximo de la muestra (positivos)


def _lerp_rgb(c1, c2, t):
    t = float(np.clip(t, 0.0, 1.0))
    return (
        c1[0] + t * (c2[0] - c1[0]),
        c1[1] + t * (c2[1] - c1[1]),
        c1[2] + t * (c2[2] - c1[2]),
    )

# --------------------------------------------------------
# Nueva función de color
COLOR_ZERO = (1, 1, 1)   # blanco
COLOR_Q3   = COLOR_80    # tu color Q3 actual
COLOR_GT_Q3 = (0, 0, 0)  # negro para > Q3

def color_from_value(v):
    """
    Gradiente 0 -> Q3_manual.
    - v <= 0 -> blanco
    - 0 < v <= Q3_manual -> gradiente blanco -> COLOR_80
    - v > Q3_manual -> negro
    """
    if (v is None) or (not np.isfinite(v)) or v <= 0:
        return COLOR_ZERO
    elif v <= vmax_manual:
        t = v / vmax_manual
        return _lerp_rgb(COLOR_ZERO, COLOR_80, t)
    else:
        return COLOR_MAX  # negro para > Q3_manual

label_positions = dict(zip(nodes["node_label"], zip(nodes["lon"], nodes["lat"])))

# --------------------------------------------------------
# 4) Dibujar
# --------------------------------------------------------
plt.figure(figsize=(8, 8))

for poly in voronoi_multi.geoms:
    clipped = poly.intersection(boundary_poly)
    if clipped.is_empty:
        continue

    c = clipped.centroid
    _, idx = tree.query([c.x, c.y], k=1)
    node_label = nodes.iloc[idx]["node_label"]

    v = kwh_map.get(node_label, 0.0)

    x, y = clipped.exterior.xy
    face = color_from_value(v)
    plt.fill(
        x, y,
        facecolor=face,
        edgecolor="none",
        zorder=1
    )

    plt.plot(x, y, linestyle="-", color="#cccccc",
             linewidth=1.2, dashes=(4, 7), zorder=2)

bx, by = boundary_poly.exterior.xy
plt.plot(bx, by, color=colors.get("max", "grey"), linewidth=1.2, zorder=5)

node_dict = {row["i"]: (row["lon"], row["lat"]) for _, row in nodes.iterrows()}
for _, row in edges.iterrows():
    n1, n2 = row["i"], row["j"]
    if n1 in node_dict and n2 in node_dict:
        x1, y1 = node_dict[n1]
        x2, y2 = node_dict[n2]
        plt.plot([x1, x2], [y1, y2], color="#000000",
                 linewidth=1.0, zorder=6)

plt.scatter(nodes["lon"], nodes["lat"], s=25, color="#000000", zorder=7)

for node_label, (x, y) in label_positions.items():
    plt.text(
        x, y,
        node_label,
        ha="center",
        va="center",
        fontsize=8,
        color="black",
        zorder=10,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.7,
            boxstyle="round,pad=0.2"
        )
    )

import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle

q1_sample = vmin_sample
q3_sample = vmax_sample

# --------------------------------------------------------
# 5) Leyenda dentro de la figura con gradiente y >Q3
# --------------------------------------------------------
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle

# --------------------------------------------------------
# Parámetros
# --------------------------------------------------------
color_zero = 'white'          # blanco
color_grad = colors.get("max", "grey")    # color real del gradiente
color_max = 'black'           # para >Q3
gradiente_height_ratio = 0.9
extra_ratio = 0.1

# Crear colormap del gradiente real
cmap = mpl.colors.LinearSegmentedColormap.from_list("grad_blanco_color", [color_zero, color_grad])

# Eje inset dentro de la figura
cax = inset_axes(plt.gca(), width="5%", height="30%", loc='lower left', borderpad=2.0)

# ScalarMappable solo para el gradiente real
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=vmax_manual))
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax, orientation='vertical')

# Ticks solo hasta Q3 manual
#ticks = [v_sample_min, q1_sample, q3_sample, v_sample_max, vmax_manual, vmax_manual*1.1]
ticks = [v_sample_min, q1_sample, q3_sample, vmax_manual*1.1]
#ticks = [v_sample_min, q1_sample, q3_sample, vmax_manual, vmax_manual*1.1]
ticklabels = [f'Min={v_sample_min:.1f} kWh',
              f'Q1={q1_sample:.1f} kWh',
              f'Q3={q3_sample:.1f} kWh',
              #f'Max={v_sample_max:.1f} kWh',
              #f'Q3(REPowerEU)={vmax_manual:.1f} kWh',
              '>Q3']

cbar.set_ticks(ticks)
cbar.set_ticklabels(ticklabels)

# -------------------------
# Franja negra superior (> Q3)
# -------------------------
cax.add_patch(
    Rectangle(
        (0, gradiente_height_ratio),
        width=1.0,
        height=extra_ratio,
        transform=cax.transAxes,
        facecolor=color_max,
        edgecolor=None,
        clip_on=False
    )
)

from collections import deque, defaultdict

# --------------------------------------------------------
# RECONSTRUIR TOPOLOGÍA DE LA RED (árbol BFS)
# --------------------------------------------------------
root = "Node_20"

# Grafo no dirigido a partir de edges
adj = defaultdict(list)
for _, row in edges.iterrows():
    a = to_node_label(row["i"])
    b = to_node_label(row["j"])
    adj[a].append(b)
    adj[b].append(a)

# BFS para obtener padre e hijos
parent = {root: None}
children = defaultdict(list)
order = []

q = deque([root])
while q:
    u = q.popleft()
    order.append(u)
    for v in adj[u]:
        if v not in parent:
            parent[v] = u
            children[u].append(v)
            q.append(v)


# ========================================================
# EXTRA (sustituye la “segunda figura GIS”):
# Esquema tipo “perfil” de la red, de izquierda (Node_43) a derecha,
# donde la ALTURA de cada nodo es el kWh acumulado.
#
# Requisitos/Asunciones:
# - Ya has calculado: kwh_cum (dict: "Node_X" -> kWh acumulado), root="Node_43"
# - Ya existen: nodes (DataFrame con lon, lat, node_label), edges (i,j),
#               to_node_label (función), children/parent/order (del BFS que hicimos)
# ========================================================

import math
from matplotlib.ticker import MaxNLocator

# ---------- 1) Utilidades: haversine (km) usando lat/lon ----------
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0  # km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ---------- 2) Mapa de coordenadas por nodo ----------
coord_map = dict(zip(nodes["node_label"], zip(nodes["lon"], nodes["lat"])))

# ---------- 3) Longitud de cada arista del árbol BFS (padre->hijo) ----------
edge_len = {}
for child, p in parent.items():
    if p is None:
        continue
    if (p in coord_map) and (child in coord_map):
        (lon1, lat1) = coord_map[p]
        (lon2, lat2) = coord_map[child]
        edge_len[(p, child)] = haversine_km(lon1, lat1, lon2, lat2)
    else:
        edge_len[(p, child)] = 0.0

# ---------- 4) Distancia acumulada desde la raíz (x-axis), siguiendo el árbol ----------
dist_from_root = {root: 0.0}
for u in order:
    if u not in dist_from_root:
        continue
    for ch in children.get(u, []):
        dist_from_root[ch] = dist_from_root[u] + edge_len.get((u, ch), 0.0)

reachable_nodes = list(dist_from_root.keys())

# --------------------------------------------------------
# 5) CONSTRUIR kWh ACUMULADO POR TOPOLOGÍA (subárbol)
# --------------------------------------------------------
kwh_map_complete = {lab: float(kwh_map.get(lab, 0.0)) for lab in parent.keys()}
kwh_cum = {lab: kwh_map_complete.get(lab, 0.0) for lab in parent.keys()}

for u in reversed(order):
    for ch in children.get(u, []):
        kwh_cum[u] += kwh_cum.get(ch, 0.0)

# ========================================================
# 6) ESQUEMA DE RED (perfil)
# ========================================================
plt.figure(figsize=(12, 5))

# 6.1 Aristas
for ch, p in parent.items():
    if p is None:
        continue
    if p not in dist_from_root or ch not in dist_from_root:
        continue

    x1 = dist_from_root[p]
    y1 = float(kwh_cum.get(p, 0.0))
    x2 = dist_from_root[ch]
    y2 = float(kwh_cum.get(ch, 0.0))

    plt.plot([x1, x2], [y1, y2], linewidth=1.5)

# 6.2 Nodos
xs = [dist_from_root[n] for n in reachable_nodes]
ys = [float(kwh_cum.get(n, 0.0)) for n in reachable_nodes]
plt.scatter(xs, ys, s=20, zorder=3)

# 6.3 Etiquetas sin solape (sin eliminar ninguna)
ax = plt.gca()

texts = []
for n in reachable_nodes:
    x = dist_from_root[n]
    y = float(kwh_cum.get(n, 0.0))
    n_short = n.replace("Node_", "")

    t = ax.text(
        x, y,
        n_short,
        fontsize=8,
        ha="center",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.15"),
        zorder=4
    )
    texts.append(t)

plt.draw()
renderer = plt.gcf().canvas.get_renderer()

def _bbox_tuple(text_obj):
    bb = text_obj.get_window_extent(renderer=renderer)
    return [bb.x0, bb.y0, bb.x1, bb.y1]

def _overlap(b1, b2, pad=2.0):
    return not (b1[2] + pad < b2[0] or b2[2] + pad < b1[0] or
                b1[3] + pad < b2[1] or b2[3] + pad < b1[1])

ref_disp = []
for t in texts:
    x_data, y_data = t.get_position()
    x_disp, y_disp = ax.transData.transform((x_data, y_data))
    ref_disp.append([x_disp, y_disp])

'''max_iter = 20000
step_y = 0.5
step_x = 0.5
pad = 1.35
pull = 0.06'''

max_iter = 200
step_y = 0.5
step_x = 0.5
pad = 1.35
pull = 0.06

pos_disp = [p.copy() for p in ref_disp]

for _ in range(max_iter):
    moved = False

    for i, t in enumerate(texts):
        x_data, y_data = ax.transData.inverted().transform(pos_disp[i])
        t.set_position((x_data, y_data))

    plt.draw()
    renderer = plt.gcf().canvas.get_renderer()
    bboxes = [_bbox_tuple(t) for t in texts]

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            b1, b2 = bboxes[i], bboxes[j]
            if not _overlap(b1, b2, pad=pad):
                continue

            c1x = (b1[0] + b1[2]) * 0.5
            c1y = (b1[1] + b1[3]) * 0.5
            c2x = (b2[0] + b2[2]) * 0.5
            c2y = (b2[1] + b2[3]) * 0.5

            sign_y = 1.0 if (c1y <= c2y) else -1.0
            sign_x = 1.0 if (c1x <= c2x) else -1.0

            pos_disp[i][1] -= sign_y * step_y
            pos_disp[j][1] += sign_y * step_y

            if step_x != 0.0:
                pos_disp[i][0] -= sign_x * step_x
                pos_disp[j][0] += sign_x * step_x

            moved = True

    for i in range(len(texts)):
        pos_disp[i][0] += (ref_disp[i][0] - pos_disp[i][0]) * pull
        pos_disp[i][1] += (ref_disp[i][1] - pos_disp[i][1]) * pull

    if not moved:
        break

for i, t in enumerate(texts):
    x_data, y_data = ax.transData.inverted().transform(pos_disp[i])
    t.set_position((x_data, y_data))

plt.draw()

plt.title(f"Esquema de red (raíz {root} a la izquierda) — Altura = kWh acumulado")
plt.xlabel("Distancia acumulada desde la raíz [km]")
plt.ylabel("kWh acumulado (subárbol) [kWh]")
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
plt.tight_layout()
plt.show()
