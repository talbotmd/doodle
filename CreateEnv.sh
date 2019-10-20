#!/bin/bash
export PATH=/Users/Manu/anaconda3/bin:$PATH
conda init bash
exec bash
conda create -n DoodleProjectEnv python=3.7 anaconda
conda activate DoodleProjectEnv
pip install -r requirements.txt
