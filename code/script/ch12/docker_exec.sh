#!/usr/bin/env bash

cd `dirname $0`/../..
docker compose exec workspace /code/ch12_integration_2.py
