"""Export the road graph as a map-ready GeoJSON layer.

Each road segment becomes a LineString with:
  * geometry in WGS84 (EPSG:4326, as required by RFC 7946), reprojected from the raster CRS;
  * `status`: "detected" (from the segmentation) or "inferred" (gap closed by the GNN);
  * `confidence`: mean road probability along the segment, or the GNN probability for links;
  * `length_m` computed in the raster's projected CRS;
  * provenance: source scene id, model version, creation time -> audit trail.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import ndimage


def pixel_to_map(rc: np.ndarray, transform) -> np.ndarray:
    """(row, col) pixel coordinates -> (x, y) in the raster CRS, at pixel centres.

    `transform` follows the GDAL/rasterio affine convention (a, b, c, d, e, f):
        x = a*col + b*row + c ;  y = d*col + e*row + f
    """
    a, b, c, d, e, f = transform[:6]
    row, col = rc[:, 0] + 0.5, rc[:, 1] + 0.5
    return np.stack([a * col + b * row + c, d * col + e * row + f], axis=1)


def graph_to_geojson(G: nx.Graph, prob: np.ndarray, transform, crs: str, provenance: dict,
                     link_confidence: dict | None = None) -> dict:
    from pyproj import Transformer

    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    link_confidence = link_confidence or {}
    created = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    features = []
    for u, v, data in G.edges(data=True):
        xy = pixel_to_map(data["path"], transform)
        lon, lat = to_wgs84.transform(xy[:, 0], xy[:, 1])
        inferred = bool(data.get("inferred", False))
        if inferred:
            conf = link_confidence.get((u, v), link_confidence.get((v, u)))
        else:
            rc = np.clip(np.rint(data["path"]).astype(int), 0, np.array(prob.shape) - 1)
            conf = float(prob[rc[:, 0], rc[:, 1]].mean())
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": np.round(np.stack([lon, lat], 1), 7).tolist()},
            "properties": {
                "status": "inferred" if inferred else "detected",
                "confidence": None if conf is None else round(float(conf), 3),
                "length_m": round(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), 1),
                "source_crs": crs,
                "created_utc": created,
                **provenance,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def write_geojson(fc: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(fc))


def class_polygons_to_geojson(classes: np.ndarray, probs: np.ndarray, cls: int, label: str, transform,
                              crs: str, provenance: dict, min_area_px: int = 3, merge_gap_px: int = 2) -> dict:
    """Vectorise one class of a class map into WGS84 polygons (e.g. built-up areas).

    Blocks separated by gaps of up to `merge_gap_px` pixels (streets, 20 m by default) are merged into one
    district polygon by morphological closing. Each polygon carries its envelope area
    (`area_m2`), the area actually classified as `label` inside it (`class_area_m2`), its mean
    class probability and the provenance fields. Regions smaller than `min_area_px` are dropped.
    """
    from pyproj import Transformer
    from rasterio import Affine
    from rasterio.features import shapes
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shp_transform

    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    affine = Affine(*transform[:6])
    pixel_area = abs(affine.a * affine.e - affine.b * affine.d)
    created = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    detected = classes == cls
    region_mask = detected
    if merge_gap_px > 0:
        closed = ndimage.binary_closing(np.pad(detected, merge_gap_px), iterations=merge_gap_px)
        region_mask = closed[merge_gap_px:-merge_gap_px, merge_gap_px:-merge_gap_px] | detected
    labels, _ = ndimage.label(region_mask, structure=np.ones((3, 3)))
    features = []
    for geom, value in shapes(labels.astype(np.int32), mask=region_mask, transform=affine, connectivity=8):
        region = labels == int(value)
        inside = region & detected
        if inside.sum() < min_area_px:
            continue
        poly = shape(geom)
        features.append({
            "type": "Feature",
            "geometry": mapping(shp_transform(to_wgs84, poly)),
            "properties": {
                "class": label,
                "area_m2": round(float(region.sum() * pixel_area), 1),
                "class_area_m2": round(float(inside.sum() * pixel_area), 1),
                "confidence": round(float(probs[cls][inside].mean()), 3),
                "source_crs": crs,
                "created_utc": created,
                **provenance,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def write_class_geotiff(path: str | Path, classes: np.ndarray, transform, crs: str, class_names) -> None:
    """Single-band uint8 GeoTIFF with a colour table, ready for QGIS."""
    import rasterio
    from rasterio import Affine

    colours = {0: (0, 0, 0, 0), 1: (255, 214, 0, 255), 2: (230, 57, 70, 255)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=classes.shape[0], width=classes.shape[1], count=1,
                       dtype="uint8", crs=crs, transform=Affine(*transform[:6]), nodata=255,
                       compress="deflate") as dst:
        dst.write(classes.astype(np.uint8), 1)
        dst.write_colormap(1, colours)
        dst.update_tags(1, **{f"class_{i}": n for i, n in enumerate(class_names)})
