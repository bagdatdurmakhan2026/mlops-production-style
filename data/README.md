# Local datasets

The CSV files are intentionally kept out of Git because the Bangalore dataset
is large and dataset redistribution terms have not been verified. Obtain and
store the files locally with these exact names:

- `bangalore-restaurants.csv` — Bangalore restaurant records used for rating
  regression, recommendations, cuisine classification, and the Bangalore map.
- `zomato_astana_dirty_full.csv` — Astana records used for the Astana map.

The project does not currently have verified source URLs or license metadata
for these two local copies. Before redistributing a dataset, record its source,
download date, license, and any required attribution here.
The SHA-256 values for the exact local copies used during development are
recorded in `checksums.sha256`. Training verifies the Bangalore CSV against
that manifest before parsing it. A checksum proves file identity, not dataset
provenance or licensing; replacing a dataset requires reviewing and updating
the manifest.

Expected common columns include `name`, `location`, `rate`, `votes`, and
`approx_cost(for two people)`. For actual geospatial maps, provide `latitude`
and `longitude` (or equivalent `lat`/`lon` columns). When coordinates are not
present, the map script creates clearly marked synthetic demonstration points.

Verify local files against the recorded checksums:

```powershell
python -m scripts.verify_data
```
