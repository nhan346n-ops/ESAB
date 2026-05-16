#How to generate MINICONDA for Globe

### Windows

* 1) Download & install miniconda : https://docs.conda.io/en/latest/miniconda.html
* 2) Configure miniconda environment with Gws requirements :

   - python create_anaconda_environments.py -target runtime

* 3) Use conda pack zip the environment

   - conda activate gws_runtime
   - conda pack -n gws_runtime -o miniconda.zip

* 4) Unzip it in 'fr.ifremer.globe.app/resources/miniconda'.

### Windows with mamba
mamba can be installed using to speed up environment solver: conda install mamba -n base -c conda-forge
Then use the following command in step 2:
python create_anaconda_environments.py -target runtime -runner mamba

# Linux
Launch:
 - docker-compose -f docker/docker-compose.yml run --rm build-miniconda
