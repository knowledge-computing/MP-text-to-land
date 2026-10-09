"""
Print structure / aggregates of spatial & tabular files.
  ref mode: also prints 3 sample rows (strings truncated)  -> for reference data
  out mode: NO values at all, only schema + non-null counts -> for deed-derived outputs
Usage: python inspect_data.py {ref|out} <file_or_folder>
"""
import sys
import pathlib
import pandas as pd
import geopandas as gpd

try:
    import pyogrio
    list_layers = lambda p: [l[0] for l in pyogrio.list_layers(p)]
except ImportError:
    import fiona
    list_layers = fiona.listlayers

EXTS = {".shp", ".geojson", ".json", ".gpkg", ".csv", ".parquet"}
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def find_sources(root: pathlib.Path):
    if root.is_file():
        yield root, None
        return
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() == ".gdb" and p.is_dir():
            for layer in list_layers(p):
                yield p, layer
        elif p.is_file() and p.suffix.lower() in EXTS and ".gdb" not in p.parts[-2]:
            yield p, None


def load(path, layer):
    s = path.suffix.lower()
    if s == ".csv":
        return pd.read_csv(path, low_memory=False)
    if s == ".parquet":
        try:
            return gpd.read_parquet(path)
        except Exception:
            return pd.read_parquet(path)
    return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)


def report(df, mode):
    print(f"  rows: {len(df)}")
    if isinstance(df, gpd.GeoDataFrame) and df.geometry is not None:
        print(f"  CRS: {df.crs}")
        print(f"  geometry types: {df.geom_type.value_counts().to_dict()}")
    cols = [c for c in df.columns if c != getattr(df, '_geometry_column_name', None)]
    summary = pd.DataFrame({
        "dtype": [str(df[c].dtype) for c in cols],
        "non_null": [int(df[c].notna().sum()) for c in cols],
        "n_unique": [int(df[c].nunique()) for c in cols] if mode == "ref" else ["-"] * len(cols),
    }, index=cols)
    print(summary.to_string())
    if mode == "ref":
        sample = df[cols].head(3).copy()
        for c in sample.columns:
            if sample[c].dtype == object:
                sample[c] = sample[c].astype(str).str.slice(0, 60)
        print("  sample rows:")
        print(sample.to_string())


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in {"ref", "out"}:
        sys.exit(__doc__)
    mode, root = sys.argv[1], pathlib.Path(sys.argv[2]).expanduser()
    for path, layer in find_sources(root):
        name = f"{path.relative_to(root.parent)}" + (f" :: {layer}" if layer else "")
        print(f"\n=== {name} ===")
        try:
            report(load(path, layer), mode)
        except Exception as e:
            print(f"  [could not read] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
