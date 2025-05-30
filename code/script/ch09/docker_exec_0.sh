#!/usr/bin/env bash

cd `dirname $0`/../..
docker compose exec workspace /code/ch09_clustering_0_faiss.py
