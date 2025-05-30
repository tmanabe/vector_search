#!/usr/bin/env bash

cd `dirname $0`/../..
docker compose exec workspace /code/ch05_advanced_vectorization_0_tune.py
