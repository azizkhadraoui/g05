"""Convert LaWAM's merged LIBERO set to G0.5's LIBERO gripper-action convention.

LaWAM's merged set: action[6] in {0, 1} (found by job 427089; NOT raw LIBERO's +/-1).
G0.5 (released g05-libero): gripper action in [0, 1], 1 = open, 0 = close, after G0.5's processor maps the raw value monotonically.
Mapping: a -> 1 - a (inverts the semantics; equals negation in G0.5's normalised [-1,1] space). Verified by job 427085: frozen g05-libero FM loss
2.443 as-is vs 0.072 with the gripper inverted (images: as-is is already the right orientation).
Videos are symlinked, meta copied, data parquet rewritten. Nothing else changes.
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC = "/export/home/kaziz/g05/data/libero_merged_no_noops_20hz"
DST = "/export/home/kaziz/g05/data/libero_merged_g05grip"
os.makedirs(DST, exist_ok=True)
if not os.path.exists(f"{DST}/videos"):
    os.symlink(f"{SRC}/videos", f"{DST}/videos")
if not os.path.exists(f"{DST}/meta"):
    shutil.copytree(f"{SRC}/meta", f"{DST}/meta")
before, after = [], []
for f in sorted(glob.glob(f"{SRC}/data/*/*.parquet")):
    rel = os.path.relpath(f, SRC)
    os.makedirs(os.path.dirname(f"{DST}/{rel}"), exist_ok=True)
    t = pq.read_table(f)
    col = t.column("action")
    a = np.stack(col.to_numpy(zero_copy_only=False)).astype(np.float32)
    before.append(np.unique(np.round(a[:, 6], 3)))
    assert set(np.unique(a[:, 6]).tolist()) <= {0.0, 1.0}, np.unique(a[:, 6])[:10]
    a[:, 6] = 1.0 - a[:, 6]
    after.append(np.unique(np.round(a[:, 6], 3)))
    arr = pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1), type=pa.float32()), a.shape[1])
    if pa.types.is_list(col.type):
        arr = pa.array(a.tolist(), type=col.type)
    t = t.set_column(t.schema.get_field_index("action"), t.schema.field("action"), arr.cast(col.type))
    tmp = f"{DST}/{rel}.tmp"
    pq.write_table(t, tmp)
    os.replace(tmp, f"{DST}/{rel}")  # atomic: a running reader keeps the old inode
    print(rel, len(t))
print("gripper values before:", np.unique(np.concatenate(before))[:10], "after:", np.unique(np.concatenate(after))[:10])
