"""Deed-polygon vs predicted-plat-polygon metrics, in a projected CRS with metre units (e.g. EPSG:26915).

contains     plat contains the whole deed polygon ("fall within", the PLSS criterion the 2025 code used)
intersects   plat and deed share area > 0 ("or partially overlap", the paper's wording for PLSS)
centroid_in  plat contains the deed centroid
covered      area(deed ∩ plat) / area(deed)
iou          area(deed ∩ plat) / area(deed ∪ plat)
centroid_m   distance from the deed centroid to the plat (0 when inside)
"""


def geom_metrics(plat, deed):
    inter = plat.intersection(deed).area if plat.intersects(deed) else 0.0
    union = plat.union(deed).area
    c = deed.centroid
    return {
        "contains": bool(plat.contains(deed)),
        "intersects": inter > 0,
        "centroid_in": bool(plat.contains(c)),
        "covered": min(inter / deed.area, 1.0) if deed.area > 0 else 0.0,
        "iou": inter / union if union > 0 else 0.0,
        "centroid_m": float(plat.distance(c)),
    }
