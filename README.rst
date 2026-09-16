Deforisk Jupyter Notebooks
==========================

A comprehensive toolkit for deforestation/degradation risk modeling and spatial analysis. This project provides Jupyter notebooks and Python scripts for processing geospatial data, building predictive models, and analyzing forest degradation patterns using machine learning and statistical methods.

📚 **Documentation**: https://deforisk-notebooks.readthedocs.io/en/latest/

Installation
------------

This project uses `micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_
(conda) for package management, because the geospatial stack (GDAL) and the RAPIDS GPU
stack (``cudf``/``cuml``/``dask-cuda``) are best installed from conda channels.

.. code-block:: bash

    # Clone repo
    git clone https://github.com/sepal-contrib/spatial-risk-module.git
    cd spatial-risk-module

    # Install micromamba (to ~/.local/bin)
    mkdir -p ~/.local/bin
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C ~/.local --strip-components=1 bin/micromamba
    ~/.local/bin/micromamba shell init -s bash -r ~/micromamba
    exec bash

    # Strict channel priority is required so RAPIDS resolves correctly
    micromamba config set channel_priority strict

    # Create the environment from environment.yml
    micromamba create -f environment.yml -y

    # Install the fork-preserving / CIRAD pip packages with --no-deps
    micromamba run -n spatial-risk pip install --no-deps \
        "geemap==0.36.0" \
        "forestatrisk>=1.3.2" "riskmapjnr>=1.3.2" "geefcc>=0.1.6" "pywdpa>=0.1.6"

    # Register the Jupyter kernel
    micromamba run -n spatial-risk python -m ipykernel install --user \
        --name spatial-risk --display-name "Python (spatial-risk)"

In VS Code / Jupyter, select the **"Python (spatial-risk)"** kernel for the notebooks.







