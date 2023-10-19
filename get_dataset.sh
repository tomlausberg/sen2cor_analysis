#!/usr/bin/bash

python /home/pf/pfstud/nimbus/download_data/sentinel_download_1C.py --outdir ./scratch/toml/sn7/sentinel2 --data ./credentials.json --geojson ./test.geojson


# for ATTEMPT in {1..48}
# do
#     echo "[Iteration $ATTEMPT]"
#     ./get_dataset.py
#     echo "Sleeping for 30 minutes..."
#     sleep 30m
# done
