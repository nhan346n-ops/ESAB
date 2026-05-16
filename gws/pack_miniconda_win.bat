call python create_anaconda_environments.py -target runtime
call conda clean -a -y
call conda activate gws_runtime
call conda pack -n gws_runtime -o miniconda.zip
