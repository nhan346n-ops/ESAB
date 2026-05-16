# -*- coding: utf-8 -*-
"""
This module create all environments defined for the project
return 0 if everything went OK or the error code value otherwise
"""

# environment names as defined in the yml files
import os
#install required libraries
os.system("pip install conda_mgr --index-url https://gitlab.ifremer.fr/api/v4/projects/fleet%2Facoustic%2Fconda_mgr/packages/pypi/simple")


#we should have import errors, since conda_mgr is installed with the previous line
from conda_mgr import conda_env

env = conda_env.DevEnv(
    environment_runtime_name="pyat_runtime", environment_test_name="pyat_test", environment_dev_name="pyat_dev"
)

env()
