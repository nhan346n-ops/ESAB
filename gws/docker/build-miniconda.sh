#!/bin/bash
set -x
. ~/miniconda/bin/activate
conda init bash
. ~/.bashrc

python create_anaconda_environments.py -target runtime
conda clean -a -y
conda pack -n gws_runtime -o ./miniconda.tar.gz
