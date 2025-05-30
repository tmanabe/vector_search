#!/usr/bin/env bash

cd `dirname $0`/../..
docker compose exec workspace /code/chaa_image_0_prec.py
