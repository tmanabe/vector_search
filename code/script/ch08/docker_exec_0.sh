#!/usr/bin/env bash

cd `dirname $0`/../..
docker compose exec workspace /code/ch08_dim_reduce_and_hash_0_numpy.py
