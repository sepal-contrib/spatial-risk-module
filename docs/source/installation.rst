Installation
============

This guide covers the installation of the Deforisk Analysis Framework in SEPAL.

What is SEPAL?
--------------

SEPAL (System for Earth Observation Data Access, Processing and Analysis for Land Monitoring) is a cloud-based platform that provides access to powerful computing resources and satellite imagery. It's designed to make geospatial analysis accessible without requiring your own infrastructure.

The Deforisk framework is designed to work seamlessly within SEPAL, taking advantage of its pre-configured environment and Google Earth Engine integration.

Installation Steps
------------------

Follow these simple steps to set up the framework in your SEPAL environment:

Open SEPAL Terminal and cloning the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Log into your SEPAL account and open a terminal window. This is where you'll run all the installation commands.
Clone the Repository code to your SEPAL workspace:

.. code-block:: bash

   git clone https://github.com/sepal-contrib/spatial-risk-module.git
   cd spatial-risk-module

This creates a folder with all the framework code and notebooks.

Step 3: Install UV Package Manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UV is a fast Python package manager that will help install all dependencies:

.. code-block:: bash

   pip install uv

Step 4: Set Up the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the main setup script to create a virtual environment and install dependencies:

.. code-block:: bash

   uv run main.py

This command automatically creates an isolated Python environment and installs all required packages.

Step 5: Install GDAL
~~~~~~~~~~~~~~~~~~~~~

GDAL is a geospatial data library needed for working with raster and vector data:

.. code-block:: bash

   uv pip install gdal[numpy]==3.8.4 --no-build-isolation --no-cache-dir --force-reinstall

Step 6: Activate the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Activate the virtual environment that was created:

.. code-block:: bash

   source .venv/bin/activate

You'll notice your terminal prompt changes to show ``(.venv)`` at the beginning - this means the environment is active.

Step 7: Install Jupyter Kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, register the environment as a Jupyter kernel so you can use it in notebooks:

.. code-block:: bash

   python -m ipykernel install --user --name deforisk-deg --display-name "deforisk-deg"

This makes the environment available in Jupyter with the name "deforisk-deg".

Using the Framework
-------------------

Once installation is complete, you can:

**Option 1: Use Jupyter Notebooks**

1. Open JupyterLab in SEPAL
2. Navigate to the ``spatial-risk-module/notebooks/`` folder
3. Select the "deforisk-deg" kernel when opening a notebook
4. Start with ``_1.variables_factory.ipynb`` to learn the basics

**Option 2: Use from Terminal**

With the environment activated, you can run Python scripts directly:

.. code-block:: bash

   source .venv/bin/activate
   python your_script.py

Google Earth Engine
-------------------

The framework uses Google Earth Engine to access satellite imagery and global datasets. In SEPAL, Earth Engine is already configured and ready to use - no additional authentication is needed!

When you start working with the notebooks, the framework will automatically connect to Earth Engine using SEPAL's credentials.

What's Next?
------------

Now that you have the framework installed, you're ready to start analyzing deforestation risk:

- :doc:`quickstart` - Understand what the framework does and see the basic workflow
- :doc:`tutorials/variable_factory` - Learn how to work with geospatial variables

Need Help?
----------

If you encounter any issues during installation:

1. Make sure you're running commands in the SEPAL terminal
2. Check that you're in the correct directory (``spatial-risk-module``)
3. Verify the environment is activated (you should see ``(.venv)`` in your prompt)
4. Try running the commands one at a time to identify where the issue occurs
