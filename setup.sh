#!/usr/bin/bash
sudo apt update python3
sudo apt install python3-pip
sudo apt install python3-venv
python -m venv ./env
source ./env/bin/activate
python -m pip install --upgrade pip
python -m pip install wheel
python -m pip install numpy
python -m pip install -r requirements.txt
