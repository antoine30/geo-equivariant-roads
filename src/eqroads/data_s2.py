"""Real data: Sentinel-2 L2A from a public STAC catalogue, road labels from OpenStreetMap.

* Scenes are found with a STAC search (Microsoft Planetary Computer) filtered on cloud cover.
* Bands are read directly from Cloud-Optimised GeoTIFFs with windowed HTTP range requests:
  only the area of interest is downloaded, never the full 110 km scene.
* The SCL scene-classification layer flags clouds and shadows; those pixels are excluded
  from training and evaluation through a `valid` mask.
* OpenStreetMap roads are reprojected to the scene CRS (UTM), buffered by road class in
  metres, and rasterised on the exact Sentinel-2 10 m grid.

Requires the optional dependencies: pip install -e ".[real]"
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BANDS = ("B02", "B03", "B04", "B08")          # 10 m bands: blue, green, red, near-infrared
SCL_INVALID = (0, 1, 3, 8, 9, 10)              # no data, saturated, cloud shadow, clouds, cirrus

# Half-width of the road buffer in metres, by OSM highway class.
ROAD_HALF_WIDTH_M = {
    "motorway": 12, "trunk": 10, "primary": 8, "secondary": 6,
    "tertiary": 5, "motorway_link": 6, "trunk_link": 6, "primary_link": 5,
}

# Default area of interest: Abu Dhabi mainland, desert and urban roads (lon/lat, WGS84).
ABU_DHABI_BBOX = (54.50, 24.35, 54.65, 24.47)


@dataclass
class Scene:
    image: np.ndarray        # (4, H, W) float32 surface reflectance
    valid: np.ndarray        # (H, W) bool, False on clouds / shadows / no data
    transform: tuple         # affine (a, b, c, d, e, f) in `crs`
    crs: str                 # e.g. "EPSG:32640"
    item_id: str             # STAC item id, kept for provenance


def search_scenes(bbox, date_range: str, max_cloud: float = 5.0, limit: int = 5):
    import planetary_computer
    import pystac_client

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=["sentinel-2-l2a"], bbox=bbox, datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}}, sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
        max_items=limit,
    )
    return list(search.items())


def _boa_offset(item) -> float:
    """Processing baseline >= 04.00 (from 25 Jan 2022) adds a -1000 DN offset to L2A reflectance."""
    baseline = str(item.properties.get("s2:processing_baseline", "00.00"))
    return -1000.0 if baseline >= "04.00" else 0.0


def read_scene(item, bbox) -> Scene:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    offset = _boa_offset(item)
    bands, transform, crs, shape = [], None, None, None
    for name in BANDS:
        with rasterio.open(item.assets[name].href) as src:
            bounds = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
            window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            data = src.read(1, window=window).astype(np.float32)
            if transform is None:
                transform, crs, shape = src.window_transform(window), src.crs.to_string(), data.shape
            bands.append((data + offset) / 10000.0)

    with rasterio.open(item.assets["SCL"].href) as src:  # 20 m -> resample to the 10 m grid
        bounds = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        window = from_bounds(*bounds, transform=src.transform)
        scl = src.read(1, window=window, out_shape=shape, resampling=Resampling.nearest)

    image = np.clip(np.stack(bands), 0, 1)
    nodata = np.all(np.stack(bands) <= 0, axis=0)
    valid = ~np.isin(scl, SCL_INVALID) & ~nodata
    return Scene(image, valid, tuple(transform)[:6], crs, item.id)


def osm_road_mask(bbox, scene: Scene) -> np.ndarray:
    """Rasterise OpenStreetMap roads onto the scene grid (1 = road)."""
    import osmnx as ox
    from rasterio import Affine
    from rasterio.features import rasterize

    tags = {"highway": list(ROAD_HALF_WIDTH_M)}
    gdf = ox.features_from_bbox(bbox, tags=tags)  # osmnx >= 2.0: bbox = (left, bottom, right, top)
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].to_crs(scene.crs)
    if gdf.empty:
        return np.zeros(scene.image.shape[1:], dtype=np.uint8)
    half_width = gdf["highway"].map(lambda h: ROAD_HALF_WIDTH_M.get(h if isinstance(h, str) else h[0], 5))
    shapes = ((geom.buffer(w), 1) for geom, w in zip(gdf.geometry, half_width))
    return rasterize(shapes, out_shape=scene.image.shape[1:], transform=Affine(*scene.transform),
                     fill=0, all_touched=True, dtype=np.uint8)


def osm_building_fraction(bbox, scene: Scene, oversample: int = 5) -> np.ndarray:
    """Fraction of each 10 m pixel covered by OpenStreetMap building footprints (0..1).

    Footprints are rasterised on a grid `oversample` times finer (2 m by default), then averaged
    back to 10 m: a 12 x 15 m villa contributes its true partial coverage instead of being
    either lost or inflated to whole pixels.
    """
    import osmnx as ox
    from rasterio import Affine
    from rasterio.features import rasterize

    gdf = ox.features_from_bbox(bbox, tags={"building": True})
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(scene.crs)
    H, W = scene.image.shape[1:]
    if gdf.empty:
        return np.zeros((H, W), dtype=np.float32)
    fine = rasterize(((g, 1) for g in gdf.geometry), out_shape=(H * oversample, W * oversample),
                     transform=Affine(*scene.transform) * Affine.scale(1 / oversample), fill=0, dtype=np.uint8)
    return fine.reshape(H, oversample, W, oversample).mean(axis=(1, 3)).astype(np.float32)


def class_mask(road_mask: np.ndarray, building_fraction: np.ndarray, built_up_threshold: float = 0.3) -> np.ndarray:
    """0 background, 1 road, 2 built-up (pixel mostly covered by roofs). Roads take precedence."""
    classes = np.zeros(road_mask.shape, dtype=np.uint8)
    classes[building_fraction >= built_up_threshold] = 2
    classes[road_mask > 0] = 1
    return classes


def save_tile(path, scene: Scene, mask: np.ndarray, bbox, classes: np.ndarray | None = None,
              building_fraction: np.ndarray | None = None) -> None:
    extra = {}
    if classes is not None:
        extra = {"classes": classes, "building_fraction": building_fraction.astype(np.float16)}
    np.savez_compressed(path, image=scene.image, mask=mask, valid=scene.valid,
                        transform=np.array(scene.transform), crs=scene.crs, item_id=scene.item_id,
                        bbox=np.array(bbox), **extra)


def load_tile(path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}
