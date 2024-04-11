#!/bin/bash
cp .env docker_worker/code/
cp .env docker_flask/code/
docker compose up -d

