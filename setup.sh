#!/usr/bin/bash
python -m venv ./env
source ./env/bin/activate
python -m pip install --upgrade pip
python -m pip install wheel
python -m pip install numpy
python -m pip install -r requirements.txt
