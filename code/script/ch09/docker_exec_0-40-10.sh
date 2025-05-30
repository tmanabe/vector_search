#!/usr/bin/env bash

cd `dirname $0`/../..
docker compose exec workspace /code/ch09_clustering_0_faiss.py \
    --number-of-centroids 40 --number-of-probes 10
