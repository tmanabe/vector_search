#!/usr/bin/env bash

cd `dirname $0`/../..
docker compose exec workspace /code/ch07_vector_compression_0_data.py
