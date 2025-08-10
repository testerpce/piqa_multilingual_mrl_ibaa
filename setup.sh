#!/bin/bash

# Create virtual env if not exists
if [ ! -d "env_mrl" ]; then
  python3 -m venv env_mrl
fi

# Activate it
source env_mrl/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment activated and requirements installed."