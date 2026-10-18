import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd, collect as c

co = c.priority_order(pd.read_csv(c.RAW / "district_coords.csv"))
c.NDVI_DIR.mkdir(exist_ok=True)
todo = [r for r in co.itertuples() if not c._ndvi_file(r.state, r.district).exists()]

def job(r):
    t = time.time()
    df = c.fetch_ndvi_one(r.lat, r.lon)
    if df is not None:
        df.to_csv(c._ndvi_file(r.state, r.district), index=False)
    return r.district, df is not None, round(time.time() - t)

with ThreadPoolExecutor(8) as ex:
    futs = [ex.submit(job, r) for r in todo]
    for f in as_completed(futs):
        try: print(*f.result(), flush=True)
        except Exception as e: print("ERROR", repr(e), flush=True)
