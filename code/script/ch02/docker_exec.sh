#!/usr/bin/env bash

cd `dirname $0`/../..
docker compose exec workspace /code/ch02_basic_vectorization.py
