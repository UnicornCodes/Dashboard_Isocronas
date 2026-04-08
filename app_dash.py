import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
import numpy as np
from PIL import Image
import io
import base64
import pandas as pd
import os
import tempfile
import requests

# Directorio base = carpeta donde está este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def ruta(nombre):
    """Retorna ruta absoluta a un archivo en la carpeta del proyecto."""
    return os.path.join(BASE_DIR, nombre)

# ── Diagnóstico al inicio ─────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"BASE_DIR: {BASE_DIR}")
for f in ["AC_PN_NUMM_fix.tif", "AC_PN_NUMM_4326.tif", "Distanc_SNA_fix.tif", "DA_CAMAS_fix.tif", "HeatMap_sss_fix.tif"]:
    existe = os.path.exists(ruta(f))
    print(f"  {'✅' if existe else '❌'} {f}")
print(f"{'='*50}\n")

# ============================================
# CONSTANTES
# ============================================
COLORES_CAT = {
    "1-2 NUCLEOS":         {"color": "#3498db", "radio": 4,  "label": "1-2 Núcleos"},
    "3-5 NUCLEOS":         {"color": "#2ecc71", "radio": 6,  "label": "3-5 Núcleos"},
    "6-12 NUCLEOS":        {"color": "#f39c12", "radio": 8,  "label": "6-12 Núcleos"},
    "SERVICIOS AMPLIADOS": {"color": "#e74c3c", "radio": 10, "label": "Servicios Ampliados"},
}

ESQUEMAS_ISOCRONA = {
    "Azules": {
        'colores': ['#d4e6f1','#a2cce3','#5faed1','#2980b9','#1a5276'],
        'labels':  ['< 30 min','30-60 min','1-2 hrs','2-7.5 hrs','> 7.5 hrs']
    },
    "Rojos": {
        'colores': ['#f9e4e4','#f1a9a9','#e06666','#cc0000','#800000'],
        'labels':  ['< 30 min','30-60 min','1-2 hrs','2-7.5 hrs','> 7.5 hrs']
    },
    "Verdes": {
        'colores': ['#d5f5e3','#a9dfbf','#52be80','#27ae60','#1e8449'],
        'labels':  ['< 30 min','30-60 min','1-2 hrs','2-7.5 hrs','> 7.5 hrs']
    },
}

RANGOS_ISOCRONA  = [(0.01,30),(30,60),(60,120),(120,450),(450,50000)]
RANGOS_HOSP      = [(0.01,30),(30,60),(60,120),(120,450),(450,99999)]

ESQUEMA_SEMAFORO = {'colores': ['#2ecc71','#f39c12','#e67e22','#e74c3c','#7b0c0c'], 'labels': ['< 30 min','30-60 min','1-2 hrs','2-7.5 hrs','> 7.5 hrs']}
ESQUEMA_AZULES   = {'colores': ['#d4e6f1','#a2cce3','#5faed1','#2980b9','#1a5276'],  'labels': ['< 30 min','30-60 min','1-2 hrs','2-7.5 hrs','> 7.5 hrs']}

RANGOS_SSS  = [(1,100),(100,500),(500,3000),(3000,15000),(15000,50000),(50000,300001)]
ESQUEMAS_SSS = {
    "Calor":  {'colores': ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026','#800026'], 'labels': ['1-100','100-500','500-3k','3k-15k','15k-50k','50k-300k']},
    "Verdes": {'colores': ['#006837','#31a354','#78c679','#c2e699','#ffffcc','#ffeda0'], 'labels': ['1-100','100-500','500-3k','3k-15k','15k-50k','50k-300k']},
}

COLORES_GM = {"Muy alto":"#7b0c0c","Alto":"#d62728","Medio":"#ff7f0e","Bajo":"#bcbd22","Muy bajo":"#2ca02c","ND":"#aaaaaa"}

HF_BASE = "https://huggingface.co/datasets/UnicornCodes/dashboard-isocronas/resolve/main"
ARCHIVOS_REMOTOS = {
    "AC_PN_NUMM_4326.tif": f"{HF_BASE}/AC_PN_NUMM_4326.tif",
    "HeatMap_sss.tif":     f"{HF_BASE}/HeatMap_sss.tif",
    "Distanc_SNA.tif":     f"{HF_BASE}/Distanc_SNA.tif",
    "DA_CAMAS.tif":        f"{HF_BASE}/DA_CAMAS.tif",
}

# ============================================
# FUNCIONES DE CARGA
# ============================================
_cache = {}

def obtener_ruta_archivo(nombre):
    if os.path.exists(nombre):
        return nombre
    url = ARCHIVOS_REMOTOS.get(nombre)
    if not url:
        raise FileNotFoundError(f"'{nombre}' no encontrado.")
    tmp_path = os.path.join(tempfile.gettempdir(), nombre)
    if not os.path.exists(tmp_path):
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        part = tmp_path + ".part"
        with open(part, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8*1024*1024):
                if chunk: f.write(chunk)
        os.rename(part, tmp_path)
    return tmp_path


def _bounds_from_gpkg():
    if 'mexico_bounds' in _cache:
        return _cache['mexico_bounds']
    fix = ruta("AC_PN_NUMM_fix.tif")
    if os.path.exists(fix):
        with rasterio.open(fix) as src:
            b = src.bounds
            bounds = (b.bottom, b.left, b.top, b.right)
    else:
        bounds = (14.5340, -117.1274, 32.7240, -86.7174)
    _cache['mexico_bounds'] = bounds
    return bounds


def cargar_raster_isocrona():
    if 'iso' in _cache: return _cache['iso']
    src_path = ruta("AC_PN_NUMM_fix.tif")
    if not os.path.exists(src_path):
        src_path = obtener_ruta_archivo("AC_PN_NUMM_4326.tif")
    print(f"  Cargando isocrona: {src_path}")
    with rasterio.open(src_path) as src:
        data = src.read(1, out_shape=(src.height//4, src.width//4),
                        resampling=rasterio.enums.Resampling.average).astype(float)
        b = src.bounds
        bounds = (b.bottom, b.left, b.top, b.right)  # (S, W, N, E)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
    data[data <= 0] = np.nan
    data[data >= 1e30] = np.nan
    print(f"  bounds={bounds}  shape={data.shape}  med={np.nanmedian(data[~np.isnan(data)]):.1f}")
    _cache['iso'] = (data, bounds)
    return data, bounds


def cargar_raster_sss():
    if 'sss' in _cache: return _cache['sss']
    archivo = "HeatMap_sss_fix.tif" if os.path.exists(ruta("HeatMap_sss_fix.tif")) else "HeatMap_sss.tif"
    print(f"  Cargando SSS: {archivo}")
    src_path = ruta(archivo) if os.path.exists(ruta(archivo)) else obtener_ruta_archivo("HeatMap_sss.tif")
    with rasterio.open(src_path) as src:
        if src.crs and src.crs.to_epsg() != 4326:
            transform, width, height = calculate_default_transform(
                src.crs, 'EPSG:4326', src.width//4, src.height//4, *src.bounds)
            data = np.empty((height, width), dtype=float)
            reproject(source=rasterio.band(src, 1), destination=data,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs='EPSG:4326',
                      resampling=Resampling.average)
            left_b, bottom_b, right_b, top_b = array_bounds(height, width, transform)
            bounds = (bottom_b, left_b, top_b, right_b)
        else:
            data = src.read(1, out_shape=(src.height//4, src.width//4),
                            resampling=rasterio.enums.Resampling.average).astype(float)
            b = src.bounds
            bounds = (b.bottom, b.left, b.top, b.right)
    data[data==-9999]=np.nan; data[data>=1e+30]=np.nan; data[data<=0]=np.nan
    _cache['sss'] = (data, bounds)
    return data, bounds


def cargar_raster_hospital(nombre):
    if nombre in _cache: return _cache[nombre]

    nombre_fix = nombre.replace(".tif", "_fix.tif")
    ruta_fix = ruta(nombre_fix)
    ruta_orig = ruta(nombre) if os.path.exists(ruta(nombre)) else obtener_ruta_archivo(nombre)
    src_path = ruta_fix if os.path.exists(ruta_fix) else ruta_orig
    print(f"  Cargando {nombre}: {src_path}")

    with rasterio.open(src_path) as src:
        if src.crs and src.crs.to_epsg() == 4326:
            # _fix ya está en 4326 — leer directo
            data = src.read(1, out_shape=(src.height//4, src.width//4),
                            resampling=Resampling.average).astype(float)
            b = src.bounds
            bounds = (b.bottom, b.left, b.top, b.right)
        else:
            # reproyectar al grid de la isocrona
            _, iso_bounds = cargar_raster_isocrona()
            S, W, N, E = iso_bounds
            from rasterio.transform import from_bounds as fb
            DST_W, DST_H = 6375//4, 3815//4
            dst_t = fb(W, S, E, N, DST_W, DST_H)
            crs_src = src.crs or rasterio.crs.CRS.from_epsg(6370)
            data = np.full((DST_H, DST_W), np.nan, dtype=float)
            reproject(source=rasterio.band(src, 1), destination=data,
                      src_transform=src.transform, src_crs=crs_src,
                      dst_transform=dst_t, dst_crs='EPSG:4326',
                      resampling=Resampling.average,
                      src_nodata=src.nodata, dst_nodata=np.nan)
            bounds = (S, W, N, E)

    data[data == -9999] = np.nan
    data[data >= 1e30]  = np.nan
    data[data <= 0]     = np.nan

    v = data[~np.isnan(data)]
    if len(v) > 0 and np.nanmedian(v) < 30 and np.nanmax(v) > 300:
        print(f"  Convirtiendo s->min")
        data = data / 60.0

    v = data[~np.isnan(data)]
    if len(v) > 0:
        print(f"  OK {len(v):,} px  med={np.nanmedian(v):.1f} min")
    _cache[nombre] = (data, bounds)
    return data, bounds


def cargar_pna():
    if 'pna' in _cache: return _cache['pna']
    gdf = gpd.read_file("PNA-IMSSB.gpkg")
    gdf = gdf[['clues_imb','nombre_de_la_unidad','categoria_gerencial',
               'nombre_de_tipologia','entidad','municipio','latitud','longitud']]
    _cache['pna'] = gdf
    return gdf


def cargar_capas_poligonos():
    if 'polys' in _cache: return _cache['polys']
    TOL = 0.01
    entidad    = gpd.read_file("Entidad.gpkg").to_crs(4326)
    entidad['geometry'] = entidad['geometry'].simplify(TOL)
    edos_nofed = gpd.read_file("EDOS_NOFED.gpkg").to_crs(4326)
    edos_nofed['geometry'] = edos_nofed['geometry'].simplify(TOL)
    ro         = gpd.read_file("Regiones_Operativas.gpkg").to_crs(4326)
    ro['geometry'] = ro['geometry'].simplify(TOL)
    _cache['polys'] = (entidad, edos_nofed, ro)
    return entidad, edos_nofed, ro


def cargar_municipios():
    if 'mun' in _cache: return _cache['mun']
    gdf = gpd.read_file("MUNICIPAL_CARACTERISTICAS.gpkg").to_crs(4326)
    gdf['geometry'] = gdf['geometry'].simplify(0.005)
    if 'CVEGEO' in gdf.columns:
        gdf['CVE_ENT'] = gdf['CVEGEO'].astype(str).str[:2].str.zfill(2)
    elif 'clmun' in gdf.columns:
        gdf['CVE_ENT'] = gdf['clmun'].astype(str).str.zfill(4).str[:2]
    _cache['mun'] = gdf
    return gdf


def cargar_agebs():
    """Carga anexo1 con tiempos a los 3 rasters y población por AGEB.
    Hace join con MUNICIPAL_CARACTERISTICAS para agregar Grado de Marginación."""
    if 'agebs' in _cache: return _cache['agebs']
    gdf = gpd.read_file("anexo1_desde_excel_tiempos.gpkg")

    # Join con municipios para obtener grado de marginación
    try:
        gdf_mun = gpd.read_file("MUNICIPAL_CARACTERISTICAS.gpkg")
        # Asegurar clave de join compatible
        if 'CVEGEO' in gdf_mun.columns:
            gdf_mun['CVEGEO_MUN'] = gdf_mun['CVEGEO'].astype(str).str.zfill(5)
        elif 'clmun' in gdf_mun.columns:
            gdf_mun['CVEGEO_MUN'] = gdf_mun['clmun'].astype(str).str.zfill(5)

        gdf['CVEGEO_MUN'] = gdf['CVEGEO_MUN'].astype(str).str.zfill(5)

        # Seleccionar solo columna de GM para el join
        col_gm = None
        for c in ['gm', 'GM', 'grado_marginacion']:
            if c in gdf_mun.columns:
                col_gm = c
                break

        if col_gm and 'CVEGEO_MUN' in gdf_mun.columns:
            mun_gm = gdf_mun[['CVEGEO_MUN', col_gm]].drop_duplicates('CVEGEO_MUN')
            gdf = gdf.merge(mun_gm, on='CVEGEO_MUN', how='left')
            gdf = gdf.rename(columns={col_gm: 'gm'})
            print(f"  AGEBs: join grado marginación OK — valores únicos: {gdf['gm'].dropna().unique()[:6]}")
        else:
            print("  AGEBs: columna GM no encontrada en MUNICIPAL_CARACTERISTICAS")
    except Exception as e:
        print(f"  AGEBs: no se pudo hacer join con municipios: {e}")

    _cache['agebs'] = gdf
    return gdf


def tabla_pob_por_categoria(gdf, col_cat, col_pob, titulo, color_accent):
    """Genera tabla de población por categoría, ordenada numéricamente ascendente"""
    import re

    PALETA = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#7b0c0c']

    if col_cat not in gdf.columns or col_pob not in gdf.columns:
        return html.P(f"Columna {col_cat} no encontrada.",
                      style={"color":"#555","fontSize":"11px","fontFamily":"DM Mono"})

    resumen = (gdf.groupby(col_cat)[col_pob]
               .sum().reset_index()
               .rename(columns={col_cat:'Categoría', col_pob:'Población'}))
    total = resumen['Población'].sum()
    resumen['%'] = resumen['Población'] / total * 100 if total > 0 else 0

    # Extraer primer número de la categoría para ordenar numéricamente
    def primer_numero(s):
        m = re.search(r'[\d.]+', str(s))
        return float(m.group()) if m else 9999

    resumen['_ord'] = resumen['Categoría'].apply(primer_numero)
    resumen = resumen.sort_values('_ord').reset_index(drop=True)

    filas = []
    for i, row in resumen.iterrows():
        c = PALETA[min(i, len(PALETA) - 1)]
        filas.append(html.Tr([
            html.Td([
                html.Span(style={"display":"inline-block","width":"8px","height":"8px",
                                 "borderRadius":"50%","backgroundColor":c,
                                 "marginRight":"6px","flexShrink":"0","verticalAlign":"middle"}),
                str(row['Categoría'])
            ], style={"fontSize":"11px","color":"#c8d0e7","padding":"5px 8px","fontFamily":"DM Mono"}),
            html.Td(f"{int(row['Población']):,}",
                    style={"fontSize":"11px","color":"#9ba8c0","padding":"5px 8px",
                           "textAlign":"right","fontFamily":"DM Mono"}),
            html.Td(f"{row['%']:.1f}%",
                    style={"fontSize":"11px","color":c,"padding":"5px 8px",
                           "textAlign":"right","fontFamily":"DM Mono","fontWeight":"600"}),
        ], style={"borderBottom":"1px solid #1a2035"}))

    return html.Div([
        html.P(titulo, style={"color":color_accent,"fontSize":"12px","fontWeight":"600",
                              "fontFamily":"Syne","marginBottom":"6px"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Categoría", style={"color":"#3d4f6e","fontSize":"10px","padding":"4px 8px","fontFamily":"DM Mono","fontWeight":"500","borderBottom":"1px solid #1e2438"}),
                html.Th("Población", style={"color":"#3d4f6e","fontSize":"10px","padding":"4px 8px","textAlign":"right","fontFamily":"DM Mono","fontWeight":"500","borderBottom":"1px solid #1e2438"}),
                html.Th("%",         style={"color":"#3d4f6e","fontSize":"10px","padding":"4px 8px","textAlign":"right","fontFamily":"DM Mono","fontWeight":"500","borderBottom":"1px solid #1e2438"}),
            ])),
            html.Tbody(filas),
            html.Tfoot(html.Tr([
                html.Td("Total", style={"fontSize":"11px","color":"#e8eaf6","padding":"5px 8px","fontFamily":"DM Mono","fontWeight":"600","borderTop":"1px solid #2d3555"}),
                html.Td(f"{int(total):,}", style={"fontSize":"11px","color":"#e8eaf6","padding":"5px 8px","textAlign":"right","fontFamily":"DM Mono","fontWeight":"600","borderTop":"1px solid #2d3555"}),
                html.Td("100%", style={"fontSize":"11px","color":"#e8eaf6","padding":"5px 8px","textAlign":"right","fontFamily":"DM Mono","fontWeight":"600","borderTop":"1px solid #2d3555"}),
            ])),
        ], style={"width":"100%","borderCollapse":"collapse",
                  "backgroundColor":"#0f1420","borderRadius":"6px","overflow":"hidden"}),
    ], style={"backgroundColor":"#0f1420","borderRadius":"8px","padding":"12px",
              "border":"1px solid #1e2438"})


def crear_imagen_raster(data, rangos, colores, aplicar_mercator=False, lat_s=14.5321, lat_n=32.7187):
    """Convierte array numpy a PNG base64. Si aplicar_mercator=True, estira verticalmente
    para compensar la distorsión de Web Mercator (Leaflet)."""
    import math
    h, w = data.shape
    colored = np.zeros((h, w, 4), dtype=np.uint8)
    for i, (vmin, vmax) in enumerate(rangos):
        mask = (data >= vmin) & (data < vmax)
        hx = colores[i].lstrip('#')
        r, g, b = int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)
        colored[mask] = [r, g, b, 210]
    colored[np.isnan(data)] = [0,0,0,0]

    if aplicar_mercator:
        # Reproyectar filas de WGS84 a Web Mercator
        def merc(lat):
            return math.log(math.tan(math.pi/4 + math.radians(lat)/2))
        merc_s = merc(lat_s)
        merc_n = merc(lat_n)
        # Para cada fila de salida, calcular de qué fila de entrada viene
        new_rows = np.zeros(h, dtype=int)
        for row_out in range(h):
            # fila 0 = norte, fila h-1 = sur
            frac = row_out / (h - 1)  # 0=norte, 1=sur
            lat_merc = merc_n - frac * (merc_n - merc_s)
            lat_deg = math.degrees(2 * math.atan(math.exp(lat_merc)) - math.pi/2)
            # Convertir lat_deg a fila en el array WGS84
            row_in = int((lat_n - lat_deg) / (lat_n - lat_s) * (h - 1))
            new_rows[row_out] = max(0, min(h-1, row_in))
        colored = colored[new_rows, :, :]

    img = Image.fromarray(colored, mode='RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def calcular_estadisticas(data):
    valid = data[~np.isnan(data)]
    rangos_labels = [
        ('< 30 min',    0,   30,   '🟢'),
        ('30 min-1 hr', 30,  60,   '🟡'),
        ('1 - 2 hrs',   60,  120,  '🟠'),
        ('2 - 7.5 hrs', 120, 450,  '🔴'),
        ('> 7.5 hrs',   450, 99999,'⚫'),
    ]
    dist = []
    for label, vmin, vmax, emoji in rangos_labels:
        count = int(np.sum((valid >= vmin) & (valid < vmax)))
        dist.append({'Rango': label, 'Emoji': emoji,
                     'Pixeles': count, 'Porcentaje': count/len(valid)*100})
    return {
        'media':       float(np.mean(valid)),
        'mediana':     float(np.median(valid)),
        'pct_30':      float(np.sum(valid<30)/len(valid)*100),
        'pct_60':      float(np.sum(valid<60)/len(valid)*100),
        'pct_120':     float(np.sum(valid<120)/len(valid)*100),
        'distribucion': dist,
    }


def construir_mapa(capas):
    basemap  = capas.get('basemap', 'CartoDB positron')
    m = folium.Map(location=[23.6, -102.5], zoom_start=5, tiles=basemap)

    # Bounds exactos desde el raster fix
    MEX_S, MEX_W, MEX_N, MEX_E = _bounds_from_gpkg()

    # ── Rasters (van primero, debajo de todo) ─────────────────────────────────
    if capas.get('iso_on') and capas.get('iso_data') is not None:
        data, _ = capas['iso_data']
        esquema = ESQUEMAS_ISOCRONA.get(capas.get('iso_esquema','Azules'))
        img = crear_imagen_raster(data, RANGOS_ISOCRONA, esquema['colores'],
                                  aplicar_mercator=True, lat_s=MEX_S, lat_n=MEX_N)
        folium.raster_layers.ImageOverlay(
            image=img, bounds=[[MEX_S, MEX_W], [MEX_N, MEX_E]],
            opacity=capas.get('iso_opacity', 0.8), name="⏱️ Isocronas"
        ).add_to(m)

    if capas.get('sss_on') and capas.get('sss_data') is not None:
        data, _ = capas['sss_data']
        esquema = ESQUEMAS_SSS.get(capas.get('sss_esquema','Calor'))
        img = crear_imagen_raster(data, RANGOS_SSS, esquema['colores'],
                                  aplicar_mercator=True, lat_s=MEX_S, lat_n=MEX_N)
        folium.raster_layers.ImageOverlay(
            image=img, bounds=[[MEX_S, MEX_W], [MEX_N, MEX_E]],
            opacity=capas.get('sss_opacity', 0.75), name="👥 HeatMap SSS"
        ).add_to(m)

    if capas.get('sna_on') and capas.get('sna_data') is not None:
        data, _ = capas['sna_data']
        esquema = ESQUEMA_SEMAFORO if capas.get('sna_esquema','Semaforo') == 'Semaforo' else ESQUEMA_AZULES
        img = crear_imagen_raster(data, RANGOS_HOSP, esquema['colores'],
                                  aplicar_mercator=True, lat_s=MEX_S, lat_n=MEX_N)
        folium.raster_layers.ImageOverlay(
            image=img, bounds=[[MEX_S, MEX_W], [MEX_N, MEX_E]],
            opacity=capas.get('sna_opacity', 0.75), name="🏥 Todos los hospitales"
        ).add_to(m)

    if capas.get('camas_on') and capas.get('camas_data') is not None:
        data, _ = capas['camas_data']
        esquema = ESQUEMA_SEMAFORO if capas.get('camas_esquema','Semaforo') == 'Semaforo' else ESQUEMA_AZULES
        img = crear_imagen_raster(data, RANGOS_HOSP, esquema['colores'],
                                  aplicar_mercator=True, lat_s=MEX_S, lat_n=MEX_N)
        folium.raster_layers.ImageOverlay(
            image=img, bounds=[[MEX_S, MEX_W], [MEX_N, MEX_E]],
            opacity=capas.get('camas_opacity', 0.75), name="🛏️ H. no especializados"
        ).add_to(m)

    # ── EDOS_NOFED — encima de rasters, tapa estados sin cobertura ────────────
    try:
        _, edos_nofed, _ = cargar_capas_poligonos()
        capa_nofed = folium.FeatureGroup(name="🔲 Estados no federalizados", show=True)
        for _, row in edos_nofed.iterrows():
            folium.GeoJson(
                row['geometry'].__geo_interface__,
                style_function=lambda f: {
                    'fillColor': '#6b7280',
                    'color':     '#4b5563',
                    'weight':    1,
                    'fillOpacity': 0.92,
                },
                tooltip=row.get('NOMGEO', '')
            ).add_to(capa_nofed)
        capa_nofed.add_to(m)
    except Exception:
        pass

    # ── Estados federalizados (contornos) ─────────────────────────────────────
    if capas.get('estados_on'):
        entidad, edos_nofed, _ = cargar_capas_poligonos()
        nofed_claves = set(edos_nofed['CVE_ENT'].astype(str).tolist())
        capa_est = folium.FeatureGroup(name="🗺️ Estados", show=True)
        for _, row in entidad.iterrows():
            es_nofed = str(row.get('CVE_ENT','')).zfill(2) in nofed_claves
            if not es_nofed:   # solo dibujar contorno de estados federalizados
                folium.GeoJson(
                    row['geometry'].__geo_interface__,
                    style_function=lambda f: {
                        'fillColor': 'transparent',
                        'color': '#4a6fa5',
                        'weight': 1.2,
                        'fillOpacity': 0,
                    },
                    tooltip=row['NOMGEO']
                ).add_to(capa_est)
        capa_est.add_to(m)

    # ── Regiones Operativas ───────────────────────────────────────────────────
    if capas.get('ro_on'):
        _, _, ro = cargar_capas_poligonos()
        capa_ro = folium.FeatureGroup(name="🔶 Regiones Operativas", show=True)
        for _, row in ro.iterrows():
            folium.GeoJson(
                row['geometry'].__geo_interface__,
                style_function=lambda f: {'fillColor':'#8e44ad','color':'#6c3483','weight':1.5,'fillOpacity':0.08},
                tooltip=row.get('nueva_regionalizacion','')
            ).add_to(capa_ro)
        capa_ro.add_to(m)

    # ── Municipios ────────────────────────────────────────────────────────────
    estado_mun = capas.get('estado_mun')
    if estado_mun and estado_mun != '(ninguno)':
        try:
            gdf_mun = cargar_municipios()
            entidad_gdf, _, _ = cargar_capas_poligonos()
            dict_estados = dict(zip(entidad_gdf['NOMGEO'], entidad_gdf['CVE_ENT'].astype(str).str.zfill(2)))
            cve = dict_estados.get(estado_mun)
            if cve:
                var = capas.get('var_mun','gm')
                gdf_fil = gdf_mun[gdf_mun['CVE_ENT']==cve]
                capa_mun = folium.FeatureGroup(name=f"🏘️ Municipios — {estado_mun}", show=True)
                for _, row in gdf_fil.iterrows():
                    color = COLORES_GM.get(str(row.get('gm','ND')).strip(),'#aaaaaa') if var == 'gm' \
                            else ('#e74c3c' if str(row.get('PJS','')).strip()=='PJS' else '#3498db')
                    folium.GeoJson(
                        row['geometry'].__geo_interface__,
                        style_function=lambda f, c=color: {'fillColor':c,'color':'#333','weight':0.6,'fillOpacity':0.55},
                        tooltip=f"{row.get('municip', row.get('NOMGEO',''))} | {row.get('gm','')}"
                    ).add_to(capa_mun)
                capa_mun.add_to(m)
        except Exception:
            pass

    # ── PNA ───────────────────────────────────────────────────────────────────
    if capas.get('pna_on'):
        gdf_pna = cargar_pna()
        filtro_ent = capas.get('filtro_ent', [])
        filtro_cat = capas.get('filtro_cat', [])
        gdf_fil = gdf_pna.copy()
        if filtro_ent: gdf_fil = gdf_fil[gdf_fil['entidad'].isin(filtro_ent)]
        if filtro_cat: gdf_fil = gdf_fil[gdf_fil['categoria_gerencial'].isin(filtro_cat)]
        capa = MarkerCluster(name="📍 Unidades PNA") if capas.get('cluster', True) \
               else folium.FeatureGroup(name="📍 Unidades PNA", show=True)
        capa.add_to(m)
        for _, row in gdf_fil.iterrows():
            cfg = COLORES_CAT.get(row['categoria_gerencial'], {"color":"#95a5a6","radio":4})
            popup_html = (f'<div style="font-family:Arial;min-width:200px;">'
                          f'<b>{row["nombre_de_la_unidad"]}</b><br>'
                          f'<span style="color:#555;">📍 {row["municipio"]}, {row["entidad"]}</span><br>'
                          f'<span style="color:#555;">🔑 {row["clues_imb"]}</span></div>')
            folium.CircleMarker(
                location=[row['latitud'], row['longitud']],
                radius=cfg['radio'], color=cfg['color'],
                fill=True, fill_color=cfg['color'], fill_opacity=0.75, weight=1,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=row['nombre_de_la_unidad'],
            ).add_to(capa)

    folium.TileLayer('OpenStreetMap', name='OSM').add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()


# ============================================
# CARGAR DATOS LIGEROS AL INICIO
# ============================================
print("Cargando capas vectoriales...")
gdf_pna_global        = cargar_pna()
gdf_entidad_g, _, _   = cargar_capas_poligonos()
estados_lista         = ['(ninguno)'] + sorted(gdf_entidad_g['NOMGEO'].tolist())
cats_lista            = sorted(gdf_pna_global['categoria_gerencial'].unique().tolist())
print("Listo.")

# ============================================
# APP DASH
# ============================================
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.CYBORG,
    "https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&display=swap"
], suppress_callback_exceptions=True)
app.title = "Accesibilidad Centros de Salud"

# CSS global — fuerza tema oscuro en todos los dropdowns de Dash
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        * { box-sizing: border-box; }

        body {
            background-color: #0a0e1a !important;
            font-family: 'DM Mono', monospace;
        }

        h1, h2, h3, h4, h5 {
            font-family: 'Syne', sans-serif !important;
        }

        /* Scrollbar sidebar */
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0f1117; }
        ::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 4px; }

        /* ---- Dropdowns Dash oscuros ---- */
        .Select-control,
        .Select--single > .Select-control,
        .Select--multi > .Select-control {
            background-color: #161b2e !important;
            border: 1px solid #2d3555 !important;
            border-radius: 6px !important;
            color: #c8d0e7 !important;
            min-height: 32px !important;
        }

        .Select-value-label,
        .Select-placeholder,
        .Select-input input {
            color: #c8d0e7 !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 11px !important;
        }

        .Select-menu-outer {
            background-color: #161b2e !important;
            border: 1px solid #2d3555 !important;
            border-radius: 6px !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6) !important;
            z-index: 9999 !important;
        }

        .Select-option {
            background-color: #161b2e !important;
            color: #c8d0e7 !important;
            font-size: 11px !important;
            padding: 8px 12px !important;
            font-family: 'DM Mono', monospace !important;
        }

        .Select-option:hover,
        .Select-option.is-focused {
            background-color: #1e2640 !important;
            color: #7eb8f7 !important;
        }

        .Select-option.is-selected {
            background-color: #1a3a6e !important;
            color: #7eb8f7 !important;
        }

        .Select-arrow { border-color: #555 transparent transparent !important; }
        .Select-clear { color: #555 !important; }

        /* Multi-select tags */
        .Select-value {
            background-color: #1a3a6e !important;
            border: 1px solid #2d5299 !important;
            border-radius: 4px !important;
            color: #7eb8f7 !important;
        }
        .Select-value-icon { border-right: 1px solid #2d5299 !important; color: #7eb8f7 !important; }
        .Select-value-icon:hover { background-color: #e74c3c !important; color: #fff !important; }

        /* Cards métricas */
        .metric-card {
            background: linear-gradient(135deg, #161b2e 0%, #1a2035 100%);
            border: 1px solid #2d3555;
            border-radius: 10px;
            padding: 14px 16px;
            transition: border-color 0.2s;
        }
        .metric-card:hover { border-color: #4a6fa5; }

        /* Insight cards */
        .insight-verde  { background: rgba(39,174,96,0.08);  border-left: 3px solid #27ae60; border-radius: 6px; padding: 14px; }
        .insight-ambar  { background: rgba(243,156,18,0.08); border-left: 3px solid #f39c12; border-radius: 6px; padding: 14px; }
        .insight-rojo   { background: rgba(231,76,60,0.08);  border-left: 3px solid #e74c3c; border-radius: 6px; padding: 14px; }

        /* Switch labels */
        .form-check-label { color: #9ba8c0 !important; font-size: 12px !important; }

        /* Slider rail */
        .rc-slider-rail { background-color: #2a2a3a !important; }
        .rc-slider-track { background-color: #2980b9 !important; }
        .rc-slider-handle { border-color: #2980b9 !important; background-color: #2980b9 !important; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

SIDEBAR_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0,
    "width": "280px", "padding": "20px 16px", "overflowY": "auto",
    "backgroundColor": "#0f1117", "borderRight": "1px solid #2a2a3a", "zIndex": 1000,
}
CONTENT_STYLE = {"marginLeft": "290px", "padding": "20px"}

def make_legend_dots(items):
    return [html.Div([
        html.Span(style={"display":"inline-block","width":"14px","height":"14px",
                         "borderRadius":"3px","backgroundColor":c,"marginRight":"8px",
                         "border":"1px solid rgba(255,255,255,0.2)","flexShrink":"0"}),
        html.Span(label, style={"fontSize":"12px","color":"#ccc"})
    ], style={"display":"flex","alignItems":"center","margin":"4px 0"}) for c, label in items]

sidebar = html.Div([
    html.H5("🏥 Panel de Control", style={"color":"#fff","marginBottom":"20px","fontWeight":"700"}),

    # ── Unidades médicas (puntos) ─────────────────────────────────────────────
    html.Div([
        html.P("📍 Unidades médicas", style={"color":"#7eb8f7","fontWeight":"700","fontSize":"13px","marginBottom":"8px","letterSpacing":"0.5px"}),
        dbc.Switch(id="pna-toggle", label="Centros de salud (CS)", value=True, style={"color":"#ccc"}),
        dbc.Switch(id="sna-toggle", label="Hospitales totales (H1)", value=False, style={"color":"#ccc"}),
        dbc.Switch(id="camas-toggle", label="H. sin monotemáticos (H2)", value=False, style={"color":"#ccc"}),
        dbc.Switch(id="cluster-toggle", label="Agrupar (cluster)", value=True, style={"color":"#ccc","marginTop":"4px"}),
        html.Label("Filtrar por estado", style={"color":"#aaa","fontSize":"11px","marginTop":"8px"}),
        dcc.Dropdown(id="filtro-ent", options=sorted(gdf_pna_global['entidad'].unique()),
                     multi=True, placeholder="Todos...",
                     style={"backgroundColor":"#1e1e2e","color":"#fff","fontSize":"12px"}),
    ], style={"borderBottom":"1px solid #2a2a3a","paddingBottom":"14px","marginBottom":"14px"}),

    # ── Isócronas (rasters, radio — solo una a la vez) ────────────────────────
    html.Div([
        html.P("⏱️ Isócronas", style={"color":"#7eb8f7","fontWeight":"700","fontSize":"13px","marginBottom":"8px","letterSpacing":"0.5px"}),
        dcc.RadioItems(
            id="iso-selector",
            options=[
                {"label": " Acceso CS",  "value": "cs"},
                {"label": " Acceso H1",  "value": "h1"},
                {"label": " Acceso H2",  "value": "h2"},
                {"label": " Ninguna",    "value": "none"},
            ],
            value="cs",
            labelStyle={"display":"block","color":"#ccc","fontSize":"12px","marginBottom":"4px","cursor":"pointer"},
            inputStyle={"marginRight":"8px","accentColor":"#7eb8f7"},
        ),
        html.Label("Opacidad", style={"color":"#aaa","fontSize":"11px","marginTop":"8px"}),
        dcc.Slider(id="iso-opacity", min=0.1, max=1.0, step=0.1, value=0.8,
                   marks={0.1:"0.1",1.0:"1"}, tooltip={"always_visible":False}),
        html.Label("Esquema", style={"color":"#aaa","fontSize":"11px","marginTop":"6px"}),
        dcc.Dropdown(id="iso-esquema", options=list(ESQUEMAS_ISOCRONA.keys()),
                     value="Azules", clearable=False,
                     style={"backgroundColor":"#1e1e2e","color":"#fff","fontSize":"12px"}),
    ], style={"borderBottom":"1px solid #2a2a3a","paddingBottom":"14px","marginBottom":"14px"}),

    # ── Población SSS HeatMap ─────────────────────────────────────────────────
    html.Div([
        html.P("👥 Población SSS (HeatMap)", style={"color":"#aaa","fontWeight":"600","fontSize":"13px","marginBottom":"6px"}),
        dbc.Switch(id="sss-toggle", label="Mostrar", value=False, style={"color":"#ccc"}),
        html.Div([
            html.Label("Opacidad", style={"color":"#aaa","fontSize":"11px"}),
            dcc.Slider(id="sss-opacity", min=0.1, max=1.0, step=0.1, value=0.75,
                       marks={0.1:"0.1",1.0:"1"}, tooltip={"always_visible":False}),
            html.Label("Esquema", style={"color":"#aaa","fontSize":"11px","marginTop":"6px"}),
            dcc.Dropdown(id="sss-esquema", options=list(ESQUEMAS_SSS.keys()),
                         value="Calor", clearable=False,
                         style={"backgroundColor":"#1e1e2e","color":"#fff","fontSize":"12px"}),
        ], id="sss-controls"),
    ], style={"borderBottom":"1px solid #2a2a3a","paddingBottom":"14px","marginBottom":"14px"}),

    # ── Capas poligonales ─────────────────────────────────────────────────────
    html.Div([
        html.P("🗂️ Capas poligonales", style={"color":"#aaa","fontWeight":"600","fontSize":"13px","marginBottom":"6px"}),
        dbc.Switch(id="estados-toggle", label="Límites de estados", value=True, style={"color":"#ccc"}),
        dbc.Switch(id="ro-toggle", label="Regiones Operativas", value=False, style={"color":"#ccc"}),
        html.Label("Municipios de...", style={"color":"#aaa","fontSize":"11px","marginTop":"8px"}),
        dcc.Dropdown(id="estado-mun", options=estados_lista, value="(ninguno)",
                     clearable=False,
                     style={"backgroundColor":"#1e1e2e","color":"#fff","fontSize":"12px"}),
        html.Div([
            html.Label("Colorear por", style={"color":"#aaa","fontSize":"11px","marginTop":"6px"}),
            dcc.Dropdown(id="var-mun",
                         options=[{"label":"Grado de marginación","value":"gm"},
                                  {"label":"Jurisdicción PJS/NPJS","value":"PJS"}],
                         value="gm", clearable=False,
                         style={"backgroundColor":"#1e1e2e","color":"#fff","fontSize":"12px"}),
        ], id="var-mun-div"),
    ], style={"borderBottom":"1px solid #2a2a3a","paddingBottom":"14px","marginBottom":"14px"}),

    # ── Mapa base ─────────────────────────────────────────────────────────────
    html.Div([
        html.P("🗺️ Mapa base", style={"color":"#aaa","fontWeight":"600","fontSize":"13px","marginBottom":"6px"}),
        dcc.Dropdown(id="basemap",
                     options=["CartoDB positron","OpenStreetMap","CartoDB dark_matter"],
                     value="CartoDB positron", clearable=False,
                     style={"backgroundColor":"#1e1e2e","color":"#fff","fontSize":"12px"}),
    ]),

    # Stores necesarios (sin UI)
    dcc.Store(id="iso-toggle", data=True),
    dcc.Store(id="sna-esquema", data="Semaforo"),
    dcc.Store(id="camas-esquema", data="Semaforo"),

], style=SIDEBAR_STYLE)

content = html.Div([

    # Header
    html.Div([
        html.Div([
            html.H2("Accesibilidad geográfica a unidades médicas",
                    style={"color":"#e8eaf6","fontFamily":"Syne, sans-serif","fontWeight":"800",
                           "fontSize":"24px","margin":"0","letterSpacing":"-0.5px"}),
            html.P("Detección de desiertos de atención · IMSS-Bienestar · Fuente: Isocronas AC_PN_NUMM + AGEBs",
                   style={"color":"#6b7a99","fontSize":"12px","margin":"4px 0 0 0","fontFamily":"DM Mono, monospace"}),
        ]),
    ], style={"marginBottom":"16px","paddingBottom":"16px","borderBottom":"1px solid #1e2438"}),

    # Métricas — reactivas al raster activo y al filtro de estado
    dbc.Row([
        dbc.Col(html.Div([
            html.P(id="label-media", children="⏱️ Tiempo promedio",
                   style={"color":"#6b7a99","fontSize":"11px","margin":"0","fontFamily":"DM Mono"}),
            html.H4(id="metric-media", style={"color":"#e8eaf6","margin":"4px 0 0 0","fontFamily":"Syne"}),
            html.P(id="label-scope", children="territorio nacional",
                   style={"color":"#3d4f6e","fontSize":"10px","margin":"2px 0 0 0","fontFamily":"DM Mono"}),
        ], className="metric-card"), width=2),
        dbc.Col(html.Div([
            html.P("📍 Mediana de acceso", style={"color":"#6b7a99","fontSize":"11px","margin":"0","fontFamily":"DM Mono"}),
            html.H4(id="metric-mediana", style={"color":"#e8eaf6","margin":"4px 0 0 0","fontFamily":"Syne"}),
            html.P("50% de la población", style={"color":"#3d4f6e","fontSize":"10px","margin":"2px 0 0 0","fontFamily":"DM Mono"}),
        ], className="metric-card"), width=2),
        dbc.Col(html.Div([
            html.P("🟢 Buena accesibilidad", style={"color":"#6b7a99","fontSize":"11px","margin":"0","fontFamily":"DM Mono"}),
            html.H4(id="metric-30", style={"color":"#2ecc71","margin":"4px 0 0 0","fontFamily":"Syne"}),
            html.P("llegan en < 30 min", style={"color":"#3d4f6e","fontSize":"10px","margin":"2px 0 0 0","fontFamily":"DM Mono"}),
        ], className="metric-card"), width=2),
        dbc.Col(html.Div([
            html.P("🟡 Acceso aceptable", style={"color":"#6b7a99","fontSize":"11px","margin":"0","fontFamily":"DM Mono"}),
            html.H4(id="metric-60", style={"color":"#f39c12","margin":"4px 0 0 0","fontFamily":"Syne"}),
            html.P("llegan en < 1 hora", style={"color":"#3d4f6e","fontSize":"10px","margin":"2px 0 0 0","fontFamily":"DM Mono"}),
        ], className="metric-card"), width=2),
        dbc.Col(html.Div([
            html.P(id="label-pna", children="📍 Unidades médicas activas",
                   style={"color":"#6b7a99","fontSize":"11px","margin":"0","fontFamily":"DM Mono"}),
            html.H4(id="metric-pna", style={"color":"#7eb8f7","margin":"4px 0 2px 0","fontFamily":"Syne"}),
            html.Div(id="metric-pna-detalle",
                     style={"color":"#4a6080","fontSize":"10px","fontFamily":"DM Mono","lineHeight":"1.5"}),
        ], className="metric-card"), width=2),
        dbc.Col(html.Div([
            html.P("🔴 Desierto de atención", style={"color":"#6b7a99","fontSize":"11px","margin":"0","fontFamily":"DM Mono"}),
            html.H4(id="metric-120", style={"color":"#e74c3c","margin":"4px 0 0 0","fontFamily":"Syne"}),
            html.P("requieren > 2 horas", style={"color":"#3d4f6e","fontSize":"10px","margin":"2px 0 0 0","fontFamily":"DM Mono"}),
        ], className="metric-card", style={"borderColor":"rgba(231,76,60,0.3)"}), width=2),
    ], className="mb-3 g-2"),

    # Mapa
    dbc.Spinner(
        html.Iframe(id="mapa", style={
            "width":"100%","height":"600px","border":"none",
            "borderRadius":"10px","boxShadow":"0 4px 24px rgba(0,0,0,0.5)"
        }),
        color="primary", spinner_style={"width":"3rem","height":"3rem"}
    ),

    html.Div(id="status-msg", style={"color":"#f39c12","fontSize":"12px","marginTop":"6px","fontFamily":"DM Mono"}),

    # ---- ANÁLISIS ----
    html.Div([
        html.Hr(style={"borderColor":"#1e2438","margin":"28px 0 20px 0"}),

        dbc.Row([
            dbc.Col([
                html.H5("📊 Análisis de accesibilidad", style={
                    "color":"#e8eaf6","fontFamily":"Syne, sans-serif","fontWeight":"700",
                    "fontSize":"16px","marginBottom":"0","display":"inline-block"
                }),
            ], width=6, style={"display":"flex","alignItems":"center"}),
            dbc.Col([
                html.Label("Filtrar por estado:", style={"color":"#9ba8c0","fontSize":"11px",
                           "fontFamily":"DM Mono","marginBottom":"4px","display":"block"}),
                dcc.Dropdown(
                    id="filtro-estado-tablas",
                    options=[{"label": e, "value": e} for e in sorted(estados_lista[1:])],
                    value=None,
                    placeholder="Todos los estados...",
                    multi=True,
                    clearable=True,
                    style={"fontSize":"12px"},
                ),
            ], width=6),
        ], className="mb-4", style={"alignItems":"center"}),

        # Panel 1 — Banner cobertura
        html.Div(id="banner-desierto", className="mb-4"),

        # Panel 2 — Tabla población por categorías (una sola, reactiva a isocrona)
        html.Div(id="tabla-pob-pna", className="mb-4"),

        # Panel 3 — Población SIN acceso por grado de marginación
        html.Div(id="tabla-pob-sna", className="mb-4"),

        # Insight cards
        dbc.Row([
            dbc.Col(html.Div(id="insight-verde"), width=4),
            dbc.Col(html.Div(id="insight-ambar"), width=4),
            dbc.Col(html.Div(id="insight-rojo"),  width=4),
        ], className="mb-4"),

        # Elementos ocultos para mantener compatibilidad con outputs del callback
        html.Div(id="tabla-dist",    style={"display":"none"}),
        html.Div(id="tabla-detalle", style={"display":"none"}),
        html.Div(id="tabla-pob-camas", style={"display":"none"}),

    ], id="seccion-analisis"),

    html.Hr(style={"borderColor":"#1e2438","margin":"16px 0"}),
    html.P("🏥 Dashboard de Accesibilidad · IMSS-Bienestar · AC_PN_NUMM · PNA IMSS-Bienestar",
           style={"color":"#2d3555","fontSize":"11px","textAlign":"center","fontFamily":"DM Mono"}),

], style=CONTENT_STYLE)

app.layout = html.Div([sidebar, content], style={"backgroundColor":"#0f1117","minHeight":"100vh"})


# ============================================
# CALLBACK PRINCIPAL
# ============================================
@app.callback(
    Output("mapa",           "srcDoc"),
    Output("metric-media",   "children"),
    Output("metric-mediana", "children"),
    Output("metric-30",      "children"),
    Output("metric-60",      "children"),
    Output("metric-pna",     "children"),
    Output("metric-120",     "children"),
    Output("status-msg",     "children"),
    Output("tabla-dist",     "children"),
    Output("tabla-detalle",  "children"),
    Output("insight-verde",    "children"),
    Output("insight-ambar",    "children"),
    Output("insight-rojo",     "children"),
    Output("tabla-pob-pna",    "children"),
    Output("tabla-pob-camas",  "children"),
    Output("tabla-pob-sna",    "children"),
    Output("banner-desierto",  "children"),
    Output("label-scope",      "children"),
    Output("metric-pna-detalle", "children"),
    Input("iso-selector",    "value"),
    Input("iso-opacity",     "value"),
    Input("iso-esquema",     "value"),
    Input("sss-toggle",      "value"),
    Input("sss-opacity",     "value"),
    Input("sss-esquema",     "value"),
    Input("sna-esquema",     "data"),
    Input("camas-esquema",   "data"),
    Input("estados-toggle",  "value"),
    Input("ro-toggle",       "value"),
    Input("estado-mun",      "value"),
    Input("var-mun",         "value"),
    Input("pna-toggle",      "value"),
    Input("cluster-toggle",  "value"),
    Input("filtro-ent",      "value"),
    Input("basemap",         "value"),
    Input("filtro-estado-tablas", "value"),
)
def actualizar_mapa(
    iso_sel, iso_op, iso_esq,
    sss_on, sss_op, sss_esq,
    sna_esq, camas_esq,
    estados_on, ro_on, estado_mun, var_mun,
    pna_on, cluster, filtro_ent,
    basemap, filtro_estado_tablas
):
    status = ""

    # Derivar toggles desde iso-selector
    iso_on    = iso_sel == "cs"
    iso_h1_on = iso_sel == "h1"
    iso_h2_on = iso_sel == "h2"

    # Umbral según isocrona activa
    umbral_min = 30 if iso_sel == "cs" else 60

    # Columna AGEBs según isocrona
    col_cat = {"cs": "cat_PNA", "h1": "cat_sna", "h2": "cat_camas"}.get(iso_sel, "cat_PNA")

    filtro_cat = []  # ya no se usa filtro de categoría en el sidebar

    capas = {
        'basemap': basemap,
        'iso_on': iso_on, 'iso_opacity': iso_op, 'iso_esquema': iso_esq, 'iso_data': None,
        'sss_on': sss_on, 'sss_opacity': sss_op, 'sss_esquema': sss_esq, 'sss_data': None,
        'sna_on': iso_h1_on, 'sna_opacity': iso_op, 'sna_esquema': sna_esq, 'sna_data': None,
        'camas_on': iso_h2_on, 'camas_opacity': iso_op, 'camas_esquema': camas_esq, 'camas_data': None,
        'estados_on': estados_on, 'ro_on': ro_on,
        'estado_mun': estado_mun, 'var_mun': var_mun,
        'pna_on': pna_on, 'cluster': cluster,
        'filtro_ent': filtro_ent or [], 'filtro_cat': filtro_cat,
    }

    stats = {'media':0,'mediana':0,'pct_30':0,'pct_60':0,'pct_120':0,'distribucion':[]}

    try:
        if iso_sel == "cs":
            data, bounds = cargar_raster_isocrona()
            capas['iso_data'] = (data, bounds)
            stats = calcular_estadisticas(data)
        elif iso_sel == "h1":
            data, bounds = cargar_raster_hospital("Distanc_SNA.tif")
            capas['sna_data'] = (data, bounds)
            stats = calcular_estadisticas(data)
        elif iso_sel == "h2":
            data, bounds = cargar_raster_hospital("DA_CAMAS.tif")
            capas['camas_data'] = (data, bounds)
            stats = calcular_estadisticas(data)
    except Exception as e:
        status += f"⚠️ Isocrona: {e} "

    try:
        if sss_on:
            capas['sss_data'] = cargar_raster_sss()
    except Exception as e:
        status += f"⚠️ SSS: {e} "

    mapa_html = construir_mapa(capas)

    # Contar PNA filtrado
    gdf_p = cargar_pna()
    if filtro_ent: gdf_p = gdf_p[gdf_p['entidad'].isin(filtro_ent)]
    n_pna = len(gdf_p) if pna_on else 0

    # ── Desglose de unidades según capa activa ────────────────────────────────
    # Cargar Excel de categorías (una sola vez, en cache)
    if 'clues_cats' not in _cache:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(ruta("CLUES_IMB_Categorías_UAS.xlsx"), read_only=True)
            ws = wb.active
            rows = list(ws.iter_rows(min_row=2, values_only=True))
            _cache['clues_cats'] = rows
        except Exception:
            _cache['clues_cats'] = []

    clues_rows = _cache.get('clues_cats', [])

    def contar_cats(nivel_filtro, excluir_cats=None):
        """Cuenta unidades EN OPERACION por categoria_gerencial."""
        c = {}
        for row in clues_rows:
            if row[10] != 'EN OPERACION': continue
            if row[11] not in nivel_filtro: continue
            cat = row[7] or 'Otros'
            if excluir_cats and cat in excluir_cats: continue
            c[cat] = c.get(cat, 0) + 1
        return c

    pna_detalle = ""
    if pna_on or iso_sel in ("h1", "h2"):
        if iso_sel == "cs" and pna_on:
            cats_cs = contar_cats(['PRIMER NIVEL'])
            nucleos = cats_cs.get('Núcleos', 0)
            cessa   = cats_cs.get('Servicios ampliados', 0)
            pna_detalle = f"Núcleos: {nucleos:,} · CESSA: {cessa:,}"
        elif iso_sel == "h1":
            cats_h = contar_cats(['SEGUNDO NIVEL', 'TERCER NIVEL'])
            lineas = [
                f"HBC: {cats_h.get('Basico comunitario',0):,}",
                f"HG: {cats_h.get('Generales',0):,}",
                f"HRAE: {cats_h.get('HRAES',0):,}",
                f"Mat.Inf: {cats_h.get('Materno infantil',0):,}",
                f"Psiq: {cats_h.get('Psiquiátrico',0):,}",
                f"Ped: {cats_h.get('Pediátrico',0):,}",
            ]
            pna_detalle = " · ".join(lineas)
        elif iso_sel == "h2":
            cats_h = contar_cats(['SEGUNDO NIVEL', 'TERCER NIVEL'],
                                  excluir_cats={'Psiquiátrico', 'Pediátrico'})
            lineas = [
                f"HBC: {cats_h.get('Basico comunitario',0):,}",
                f"HG: {cats_h.get('Generales',0):,}",
                f"HRAE: {cats_h.get('HRAES',0):,}",
                f"Mat.Inf: {cats_h.get('Materno infantil',0):,}",
            ]
            pna_detalle = " · ".join(lineas)

    # ---- Scope label para tarjeta de tiempo promedio ----
    scope_label = f"{', '.join(filtro_estado_tablas)}" if filtro_estado_tablas else "territorio nacional"

    # ---- Distribución raster (barras compactas, sin tabla redundante) ----
    COLORES_RANGOS = ['#2ecc71','#f39c12','#e67e22','#e74c3c','#7b0c0c']
    dist = stats.get('distribucion', [])

    if dist:
        max_pct = max(d['Porcentaje'] for d in dist) or 1
        barras = html.Div([
            html.P("Distribución por tiempo de viaje · isocronas raster",
                   style={"color":"#6b7a99","fontSize":"11px","marginBottom":"10px","fontFamily":"DM Mono"}),
            *[html.Div([
                html.Div(d['Emoji'] + " " + d['Rango'],
                         style={"fontSize":"11px","color":"#9ba8c0","width":"110px","flexShrink":"0","fontFamily":"DM Mono"}),
                html.Div(style={
                    "height":"16px","borderRadius":"3px","flexGrow":"1",
                    "backgroundColor": COLORES_RANGOS[i],
                    "width": f"{d['Porcentaje']/max_pct*100:.0f}%",
                    "minWidth":"2px","transition":"width 0.4s ease"
                }),
                html.Div(f"{d['Porcentaje']:.1f}%",
                         style={"fontSize":"11px","color": COLORES_RANGOS[i],"width":"40px",
                                "textAlign":"right","flexShrink":"0","fontFamily":"DM Mono","fontWeight":"600"}),
            ], style={"display":"flex","alignItems":"center","gap":"10px","marginBottom":"5px"})
            for i, d in enumerate(dist)]
        ], style={"backgroundColor":"#0f1420","borderRadius":"8px","padding":"14px",
                  "border":"1px solid #1e2438"})
    else:
        barras = html.P("Activa las isocronas para ver la distribución.",
                        style={"color":"#3d4f6e","fontSize":"12px","fontFamily":"DM Mono",
                               "padding":"14px","backgroundColor":"#0f1420","borderRadius":"8px",
                               "border":"1px dashed #1e2438"})

    tabla_detalle_html = html.Div()

    # ── Configuración según isocrona activa ───────────────────────────────────
    COL_POB = 'POBLACIÓN NO DERECHOHABIENTE'
    ISO_CONFIG = {
        "cs":   {"col": "cat_PNA",   "umbral": 30,  "label": "centros de salud",    "color": "#7eb8f7"},
        "h1":   {"col": "cat_sna",   "umbral": 60,  "label": "hospitales totales",  "color": "#55efc4"},
        "h2":   {"col": "cat_camas", "umbral": 60,  "label": "H. sin monotemáticos","color": "#a29bfe"},
        "none": {"col": "cat_PNA",   "umbral": 30,  "label": "centros de salud",    "color": "#7eb8f7"},
    }
    cfg = ISO_CONFIG.get(iso_sel, ISO_CONFIG["cs"])
    col_cat   = cfg["col"]
    umbral    = cfg["umbral"]
    iso_label = cfg["label"]
    iso_color = cfg["color"]

    pob_desierto = 0
    pob_total    = 0
    pob_con_acceso = 0
    pct_buena  = stats['pct_60']
    pct_media  = stats.get('pct_120', 0) - stats['pct_60']
    pct_baja   = 100 - stats.get('pct_120', 0)
    panel2 = html.Div()
    panel3 = html.Div()

    try:
        gdf_ageb = cargar_agebs()
        gdf_f = gdf_ageb.copy()
        if filtro_estado_tablas and 'ENTIDAD' in gdf_f.columns:
            gdf_f = gdf_f[gdf_f['ENTIDAD'].isin(filtro_estado_tablas)]

        estado_label = f" · {', '.join(filtro_estado_tablas)}" if filtro_estado_tablas else ""

        if COL_POB in gdf_f.columns and col_cat in gdf_f.columns:
            pob_total = int(gdf_f[COL_POB].sum())

            # Categorías con acceso en rango adecuado
            cat_ok = ['0 a 30 min'] if umbral == 30 else ['0 a 30 min', '30.1 a 60']
            cat_sin = [c for c in gdf_f[col_cat].unique() if c not in cat_ok and c is not None]

            pob_con_acceso = int(gdf_f.loc[gdf_f[col_cat].isin(cat_ok), COL_POB].sum())
            pob_desierto   = int(gdf_f.loc[gdf_f[col_cat].isin(cat_sin), COL_POB].sum())

            pct_con_acceso = pob_con_acceso / pob_total * 100 if pob_total else 0

            # ── Panel 2: tabla población por categorías ────────────────────
            panel2 = tabla_pob_por_categoria(
                gdf_f, col_cat, COL_POB,
                f"👥 Población no derechohabiente · {iso_label}{estado_label}",
                iso_color
            )

            # ── Panel 3: sin acceso agrupado por Grado de Marginación ──────
            COL_GM = 'gm' if 'gm' in gdf_f.columns else None

            if COL_GM:
                gdf_sin = gdf_f[gdf_f[col_cat].isin(cat_sin)].copy()
                gm_pob = (gdf_sin.groupby(COL_GM)[COL_POB]
                          .sum().sort_values(ascending=False).reset_index())
                total_sin = gm_pob[COL_POB].sum()

                GM_COLORS = {
                    "Muy alto": "#e74c3c", "Alto": "#e67e22",
                    "Medio": "#f1c40f",    "Bajo": "#2ecc71",
                    "Muy bajo": "#3498db", "No aplica": "#95a5a6",
                }
                filas = []
                for _, row in gm_pob.iterrows():
                    gm  = str(row[COL_GM])
                    pob = int(row[COL_POB])
                    pct = pob / total_sin * 100 if total_sin else 0
                    color = GM_COLORS.get(gm, "#7f8c8d")
                    filas.append(html.Tr([
                        html.Td([
                            html.Span(style={"display":"inline-block","width":"10px","height":"10px",
                                            "borderRadius":"2px","backgroundColor":color,
                                            "marginRight":"8px","flexShrink":"0"}),
                            html.Span(gm, style={"color":"#c8d0e7","fontSize":"12px","fontFamily":"DM Mono"}),
                        ], style={"padding":"6px 10px","display":"flex","alignItems":"center"}),
                        html.Td(f"{pob:,.0f}",
                                style={"padding":"6px 10px","color":"#e8eaf6","fontFamily":"DM Mono","fontSize":"12px","textAlign":"right"}),
                        html.Td(f"{pct:.1f}%",
                                style={"padding":"6px 10px","color":color,"fontFamily":"DM Mono","fontSize":"12px","textAlign":"right","fontWeight":"600"}),
                    ]))

                panel3 = html.Div([
                    html.H6(f"🔴 Sin acceso en rango adecuado · por Grado de Marginación{estado_label}",
                            style={"color":"#9ba8c0","fontFamily":"Syne","fontSize":"13px",
                                   "fontWeight":"600","marginBottom":"12px"}),
                    html.Table(
                        [html.Thead(html.Tr([
                            html.Th("Grado de Marginación", style={"padding":"6px 10px","color":"#6b7a99","fontSize":"11px","fontFamily":"DM Mono","textAlign":"left"}),
                            html.Th("Población",            style={"padding":"6px 10px","color":"#6b7a99","fontSize":"11px","fontFamily":"DM Mono","textAlign":"right"}),
                            html.Th("%",                    style={"padding":"6px 10px","color":"#6b7a99","fontSize":"11px","fontFamily":"DM Mono","textAlign":"right"}),
                        ]))] + [html.Tbody(filas)],
                        style={"width":"100%","borderCollapse":"collapse",
                               "backgroundColor":"#0f1420","borderRadius":"8px",
                               "border":"1px solid #1e2438","overflow":"hidden"}
                    ),
                ], style={"backgroundColor":"#0f1420","borderRadius":"8px",
                          "padding":"16px","border":"1px solid rgba(231,76,60,0.2)"})
            else:
                panel3 = html.P("Columna de Grado de Marginación no encontrada en AGEBs.",
                                style={"color":"#6b7a99","fontSize":"11px","fontFamily":"DM Mono"})

            if filtro_estado_tablas:
                total_v = len(gdf_f)
                pct_buena = float(gdf_f[col_cat].isin(cat_ok).sum() / total_v * 100) if total_v else 0
                pct_baja  = float(gdf_f[col_cat].isin(cat_sin).sum() / total_v * 100) if total_v else 0
                pct_media = 100 - pct_buena - pct_baja

    except FileNotFoundError:
        panel2 = panel3 = html.P("⚠️ anexo1_desde_excel_tiempos.gpkg no encontrado.",
                                  style={"color":"#f39c12","fontSize":"11px","fontFamily":"DM Mono"})
    except Exception as e:
        panel2 = panel3 = html.P(f"⚠️ Error AGEBs: {e}",
                                  style={"color":"#e74c3c","fontSize":"11px","fontFamily":"DM Mono"})

    # ── Panel 1: Banner cobertura ─────────────────────────────────────────────
    pct_con_acceso_val = (pob_con_acceso / pob_total * 100) if pob_total > 0 else 0
    estado_txt = f" en {', '.join(filtro_estado_tablas)}" if filtro_estado_tablas else " a nivel nacional"
    rango_txt  = f"0 a {umbral}" if umbral == 30 else f"0 a {umbral}"

    banner = html.Div([
        html.Div([
            html.Div([
                html.P(f"✅ Personas con acceso{estado_txt}",
                       style={"color":"#2ecc71","fontSize":"12px","fontWeight":"700",
                              "margin":"0","fontFamily":"Syne"}),
                html.H2(f"{pob_con_acceso:,.0f}",
                        style={"color":"#2ecc71","margin":"4px 0","fontFamily":"Syne",
                               "fontSize":"32px","fontWeight":"800"}),
                html.P(f"personas con acceso en un rango de {rango_txt} minutos de desplazamiento hacia {iso_label}",
                       style={"color":"#9ba8c0","fontSize":"12px","margin":"0","fontFamily":"DM Mono"}),
            ], style={"flex":"1"}),
            html.Div([
                html.P(f"{pct_con_acceso_val:.1f}%",
                       style={"color":"#2ecc71","fontSize":"48px","fontWeight":"800",
                              "fontFamily":"Syne","margin":"0","lineHeight":"1"}),
                html.P("de la población no derechohabiente",
                       style={"color":"#6b7a99","fontSize":"11px","margin":"4px 0 0 0","fontFamily":"DM Mono"}),
            ], style={"textAlign":"right","flexShrink":"0"}),
        ], style={"display":"flex","justifyContent":"space-between","alignItems":"center"}),
    ], style={
        "backgroundColor":"rgba(46,204,113,0.07)",
        "border":"1px solid rgba(46,204,113,0.3)",
        "borderLeft":"4px solid #2ecc71",
        "borderRadius":"10px","padding":"20px 24px",
    }) if pob_total > 0 else html.Div()

    # ── Insight cards ─────────────────────────────────────────────────────────
    ins_verde = html.Div([
        html.H6("🟢 Buena accesibilidad", style={"color":"#27ae60","margin":"0 0 6px 0","fontFamily":"Syne","fontSize":"13px"}),
        html.H3(f"{pct_buena:.1f}%", style={"color":"#2ecc71","margin":"0 0 4px 0","fontFamily":"Syne"}),
        html.P(f"llega en menos de {umbral} min", style={"color":"#9ba8c0","fontSize":"12px","margin":"0","fontFamily":"DM Mono"}),
    ], className="insight-verde")

    ins_ambar = html.Div([
        html.H6("🟡 Accesibilidad media", style={"color":"#d68910","margin":"0 0 6px 0","fontFamily":"Syne","fontSize":"13px"}),
        html.H3(f"{pct_media:.1f}%", style={"color":"#f39c12","margin":"0 0 4px 0","fontFamily":"Syne"}),
        html.P("acceso en rango intermedio", style={"color":"#9ba8c0","fontSize":"12px","margin":"0","fontFamily":"DM Mono"}),
    ], className="insight-ambar")

    ins_rojo = html.Div([
        html.H6("🔴 Sin acceso adecuado", style={"color":"#c0392b","margin":"0 0 6px 0","fontFamily":"Syne","fontSize":"13px"}),
        html.H3(f"{pct_baja:.1f}%", style={"color":"#e74c3c","margin":"0 0 4px 0","fontFamily":"Syne"}),
        html.P(f"fuera del rango de {umbral} min", style={"color":"#9ba8c0","fontSize":"12px","margin":"0","fontFamily":"DM Mono"}),
    ], className="insight-rojo")

    barras = html.Div()  # ya no se usa

    return (
        mapa_html,
        f"{stats['media']:.0f} min",
        f"{stats['mediana']:.0f} min",
        f"{stats['pct_30']:.1f}%",
        f"{stats['pct_60']:.1f}%",
        f"{n_pna:,}",
        f"{pct_baja:.1f}%",
        status,
        barras,
        tabla_detalle_html,
        ins_verde,
        ins_ambar,
        ins_rojo,
        panel2,
        html.Div(),   # tabla-pob-camas oculta
        panel3,
        banner,
        scope_label,
        pna_detalle,
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)