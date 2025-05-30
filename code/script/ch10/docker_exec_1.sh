#!/usr/bin/env bash

cd `dirname $0`/../..
docker compose exec workspace /code/ch10_graph_1_os.py
