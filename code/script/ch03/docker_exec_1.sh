#!/usr/bin/env bash

cd `dirname $0`/../..
docker compose exec workspace /code/ch03_vector_search_engines_1_faiss.py
