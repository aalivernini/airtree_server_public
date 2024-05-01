#!/bin/bash
cp .env docker_worker/code/
cp .env docker_fastapi/code/
docker compose up -d

