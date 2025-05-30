#!/usr/bin/env bash

cd `dirname $0`/../..
docker compose exec workspace /code/ch11_integration_1.py
