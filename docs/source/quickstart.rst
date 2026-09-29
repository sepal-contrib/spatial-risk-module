Quick Start Guide
=================

Understanding the Deforisk Framework
-------------------------------------

What Does This Framework Do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Deforisk Analysis Framework helps you analyze deforestation and forest degradation risk in any region of the world. Think of it as a toolkit that combines:

- **Satellite imagery** from Google Earth Engine (like forest cover maps)
- **Your own data** (local maps, boundaries, infrastructure)
- **Processing tools** to analyze how these factors relate to forest loss

The framework is particularly useful for:

- Identifying areas at high risk of deforestation
- Understanding what drives forest loss (roads, agriculture, settlements, etc.)
- Creating maps and visualizations for reports and decision-making
- Analyzing historical patterns and predicting future trends

Who Is This For?
~~~~~~~~~~~~~~~~

This framework is designed for forest analysts, conservation practitioners, and researchers who need to work with geospatial data but may not be coding experts. While some Python knowledge is helpful, the framework simplifies many complex operations into straightforward commands.

The Core Concept: Projects
---------------------------

Everything in the framework revolves around a **Project**. A project is like a workspace that keeps everything organized:

What Is a Project?
~~~~~~~~~~~~~~~~~~

A project contains:

- **A name**: To identify your analysis (e.g., "amazon_analysis", "mtq_deforestation")
- **Time period**: The years you're analyzing (e.g., 2015-2024)
- **Variables**: All your data layers (forest cover, roads, protected areas, etc.)
- **Settings**: How your data should be aligned and processed

When you create a project, the framework automatically creates an organized folder structure to store your raw data, processed results, and visualizations.

Creating Your First Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating a project is simple:

.. code-block:: python

   from spatialrisk.project import Project
   
   project = Project(
       project_name="my_forest_analysis",
       years=[2015, 2020, 2024]
   )

This creates a project called "my_forest_analysis" that will analyze data across three time periods: 2015, 2020, and 2024.

The Project Folder Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once created, your project has this structure:

.. code-block:: text

   data/my_forest_analysis/
   ├── data/              # Your processed, analysis-ready data goes here
   ├── data_raw/          # Original downloaded data is stored here
   ├── plots/             # Maps and visualizations
   └── my_forest_analysis_project.json  # Project settings and metadata

You don't need to create these folders manually - the framework does it for you!

Defining Your Area of Interest
-------------------------------

Before you can analyze deforestation, you need to define **where** you want to analyze. This is called your Area of Interest (AOI).

Why Is This Important?
~~~~~~~~~~~~~~~~~~~~~~~

Think of your AOI as drawing a boundary on a map and saying "I want to study what's happening inside this area." The AOI determines:

- Which satellite imagery gets downloaded
- The extent of your analysis
- The size of your final maps and datasets

A well-defined AOI helps you focus your analysis and keeps file sizes manageable.

Using Administrative Boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to define an AOI is using existing administrative boundaries like countries, states, or provinces. The framework can automatically fetch these from FAO GAUL (a global database of administrative areas).

For example, to analyze Martinique (a Caribbean island):

.. code-block:: python

   import ee
   from spatialrisk.gee.ee_fao_gaul import get_fao_gaul_features
   from spatialrisk.variables import GEEVar
   
   # Initialize Google Earth Engine
   ee.Initialize()
   
   # Get Martinique's boundary using its ISO code
   iso_code = "MTQ"
   aoi_image = get_fao_gaul_features(level=0, code=iso_code)

What just happened?
^^^^^^^^^^^^^^^^^^^^

1. We connected to Google Earth Engine (SEPAL does this automatically)
2. We requested the boundary of Martinique using its 3-letter ISO code
3. The framework retrieved the official administrative boundary

Every country and territory has an ISO code - you just need to know the code for your area.

Creating an AOI Variable
~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have the boundary, you turn it into a "variable" that your project can use:

.. code-block:: python

   aoi_var = GEEVar(
       name="aoi",
       data_type="vector",  # It's a boundary (lines/polygons)
       gee_images=[aoi_image],
       project=project,
       aoi=aoi_image.geometry(),
   )
   
   # Download it to your computer
   aoi_local = aoi_var.to_local_vector()

This downloads the boundary as a shapefile that you can view in GIS software.

Using Custom Boundaries
~~~~~~~~~~~~~~~~~~~~~~~~

You can also use your own boundaries! If you have a shapefile or KML of your study area, you can load it directly:

.. code-block:: python

   from spatialrisk.variables import LocalVectorVar
   
   # Use your own shapefile
   aoi_var = LocalVectorVar(
       name="my_custom_aoi",
       path="/path/to/your/boundary.shp",
       project=project
   )

This is useful when you're studying:

- A specific forest reserve
- A watershed or ecological region
- A custom-defined area that doesn't match political boundaries

Understanding the Workflow
---------------------------

Now that you understand projects and areas of interest, here's the typical workflow:

1. **Create a Project**
   
   Set up your workspace with a name and time period.

2. **Define Your Area of Interest**
   
   Choose the geographic area you want to study.

3. **Add Variables** (covered in Variable Factory tutorial)
   
   Bring in data: forest cover, roads, protected areas, topography, etc.

4. **Process the Data** (covered in Process Factory tutorial)
   
   Align all your data layers so they match perfectly and calculate distances, forest loss, etc.

5. **Analyze and Visualize**
   
   Create maps, run statistical models, generate reports.

What's Next?
------------

This guide covered the foundational concepts and first steps. To continue learning:

**Variable Factory Tutorial**

Learn how to add different types of data to your project - from satellite imagery to local datasets. This is where you build your complete data collection.

- :doc:`tutorials/variable_factory` - Working with variables

**Working with Notebooks**

The framework includes Jupyter notebooks that walk you through complete examples:

- ``_1.variables_factory.ipynb`` - Creating and managing variables
- ``_2.process_factory.ipynb`` - Processing and analyzing data

These notebooks contain detailed explanations and working code you can run step-by-step.
