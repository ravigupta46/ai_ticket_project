#!/bin/bash
set -o errexit

pip install --upgrade pip setuptools wheel
pip install --no-cache-dir -r requirements.txt
