#!/usr/bin/env bash

cd `dirname $0`/../..
docker compose exec workspace /code/ch06_fast_vectorization.py --sample-rate 1.0
