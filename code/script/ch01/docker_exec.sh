#!/usr/bin/env bash

cd `dirname $0`/../..
docker compose exec workspace /code/ch01_data_preparation.py
