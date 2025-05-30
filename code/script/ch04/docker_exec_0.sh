#!/usr/bin/env bash

cd `dirname $0`/../..
docker compose exec workspace /code/ch04_evaluate_search_results_0_ndcg.py
