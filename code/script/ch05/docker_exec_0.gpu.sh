#!/usr/bin/env bash

cd `dirname $0`/../..
docker compose exec --tty workspace \
    /code/ch05_advanced_vectorization_0_tune.py --sample-rate 1.0
