import json
from pathlib import Path

import geojson
import numpy as np
from turfpy.measurement import points_within_polygon

BACKARC_JSON_FFP = Path(__file__).parent.parent.parent / "data/NSHM2022" / "backarc.json" 

def get_backarc_mask(locs: np.ndarray):
    """
    Computes a mask identifying each location
    that requires the backarc flag based on
    wether it is inside the backarc polygon or not

    locs: array of floats
        [lon, lat]
    """

    # Determine if backarc needs to be enabled for each loc
    points = geojson.FeatureCollection(
        [
            geojson.Feature(geometry=geojson.Point(tuple(cur_loc[::-1]), id=ix))
            for ix, cur_loc in enumerate(locs)
        ]
    )
    with BACKARC_JSON_FFP.open("r") as f:
        poly_coords = np.flip(json.load(f)["geometry"]["coordinates"][0], axis=1)

    polygon = geojson.Polygon([poly_coords.tolist()])
    backarc_ind = (
        [
            cur_point["geometry"]["id"]
            for cur_point in points_within_polygon(points, polygon)["features"]
        ],
    )
    backarc_mask = np.zeros(shape=locs.shape[0], dtype=bool)
    backarc_mask[backarc_ind] = True

    return backarc_mask