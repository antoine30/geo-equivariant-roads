"""Download a Sentinel-2 L2A area of interest (windowed COG reads) and its OpenStreetMap road mask.

    python scripts/fetch_s2.py --out data/abu_dhabi.npz
    python scripts/fetch_s2.py --buildings --out data/abu_dhabi.npz     # roads + built-up labels
    python scripts/fetch_s2.py --bbox 55.20 25.05 55.35 25.20 --dates 2024-01-01/2024-03-31 --out data/dubai.npz

Requires: pip install -e ".[real]"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eqroads.data_s2 import (
    ABU_DHABI_BBOX,
    class_mask,
    osm_building_fraction,
    osm_road_mask,
    read_scene,
    save_tile,
    search_scenes,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, default=ABU_DHABI_BBOX,
                    metavar=("WEST", "SOUTH", "EAST", "NORTH"), help="WGS84 lon/lat")
    ap.add_argument("--dates", default="2024-01-01/2024-04-30", help="STAC datetime range")
    ap.add_argument("--max-cloud", type=float, default=5.0)
    ap.add_argument("--buildings", action="store_true", help="also label built-up areas from OSM footprints")
    ap.add_argument("--built-up-threshold", type=float, default=0.3,
                    help="min fraction of a 10 m pixel covered by roofs to label it built-up")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    items = search_scenes(args.bbox, args.dates, args.max_cloud)
    if not items:
        raise SystemExit("no scene found: widen --dates or raise --max-cloud")
    item = items[0]
    print(f"scene {item.id}  cloud {item.properties.get('eo:cloud_cover')}%  "
          f"baseline {item.properties.get('s2:processing_baseline')}")

    scene = read_scene(item, args.bbox)
    print(f"read {scene.image.shape} in {scene.crs}, valid pixels {scene.valid.mean():.1%}")
    mask = osm_road_mask(args.bbox, scene)
    print(f"OSM road pixels {mask.mean():.2%}")

    classes = fraction = None
    if args.buildings:
        fraction = osm_building_fraction(args.bbox, scene)
        classes = class_mask(mask, fraction, args.built_up_threshold)
        print(f"OSM building coverage {fraction.mean():.2%}, built-up pixels {(classes == 2).mean():.2%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_tile(args.out, scene, mask, args.bbox, classes, fraction)
    print("saved", args.out)


if __name__ == "__main__":
    main()
