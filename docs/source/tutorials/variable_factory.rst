Variable Factory Tutorial
=========================

Understanding Variables
-----------------------

What Are Variables?
~~~~~~~~~~~~~~~~~~~

In the context of deforestation analysis, a **variable** is any data layer that
might influence or describe forest loss. Think of variables as ingredients
you're collecting for your analysis recipe.

Examples of variables include:

- **Forest cover maps**: Showing where forests exist
- **Roads**: Infrastructure that provides access to forests
- **Protected areas**: National parks, reserves, and conservation zones
- **Topography**: Elevation, slope - forests on steep slopes might be safer
- **Settlements**: Cities, towns, villages - sources of pressure on forests
- **Agriculture**: Crop areas that might expand into forests

The "factory" pattern in this framework means you have a standardized way to
create, organize, and manage all these different data sources.

Types of Variables
~~~~~~~~~~~~~~~~~~

The framework works with three main types of variables:

**1. GEE Variables (GEEVar)**

These come from Google Earth Engine - a massive catalog of satellite imagery
and global datasets. The advantage? You don't need to download gigabytes of
data manually. You just request what you need for your area.

Examples: Hansen forest cover, population density, rainfall data, protected
areas databases.

**2. Local Raster Variables (LocalRasterVar)**

These are grid-based datasets you already have on your computer, like GeoTIFF
files. Each cell in the grid has a value (elevation, temperature, etc.).

Examples: High-resolution forest maps from national agencies, digital elevation
models, climate data.

**3. Local Vector Variables (LocalVectorVar)**

These are feature-based datasets with points, lines, or polygons. They define
specific locations or boundaries.

Examples: Shapefiles of roads, boundaries of protected areas, locations of
logging concessions.

Building Your Variable Collection
----------------------------------

The Process
~~~~~~~~~~~

Think of building your variable collection like assembling a toolkit. You're
gathering all the data layers you think might help explain deforestation
patterns in your area.

The typical process looks like this:

1. **Start with your Area of Interest** - Define where you're working
2. **Add forest layers** - Your main outcome variable
3. **Add driver variables** - Things that might cause deforestation
4. **Add contextual variables** - Environmental and geographic context
5. **Organize with tags** - Group related variables together

Each variable gets a meaningful name, metadata about what it represents, and
optional tags for organization.

Working with Google Earth Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Google Earth Engine gives you access to incredible datasets without downloading
terabytes of data. Here's what you can access:

**Global Forest Change (Hansen)**

The most widely used global forest dataset, updated annually. Shows tree cover,
loss, and gain from 2000 to present.

.. code-block:: python

   # Example: Getting forest cover for 2015
   gfcImage = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
   forest2000 = gfcImage.select(["treecover2000"])

**Protected Areas (WDPA)**

Global database of protected areas, parks, and reserves.

.. code-block:: python

   # Get protected areas
   wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons")

**Topography (SRTM)**

Elevation data covering most of the world at 30-meter resolution.

.. code-block:: python

   # Get elevation
   srtm = ee.Image("CGIAR/SRTM90_V4")

The framework handles the complexity of downloading these datasets just for
your area of interest.

Working with Local Data
~~~~~~~~~~~~~~~~~~~~~~~

Often, you'll have data from local sources that's more accurate or recent than
global datasets. The framework makes it easy to integrate:

**Local Rasters**

If you have GeoTIFF or other raster files:

.. code-block:: python

   from spatialrisk.variables import LocalRasterVar

   local_dem = LocalRasterVar(
       name="high_res_dem",
       path="/path/to/elevation.tif",
       project=project,
       tags=["topography"]
   )

**Local Vectors**

If you have shapefiles or other vector formats:

.. code-block:: python

   from spatialrisk.variables import LocalVectorVar

   local_roads = LocalVectorVar(
       name="national_roads",
       path="/path/to/roads.shp",
       project=project,
       tags=["infrastructure"]
   )

The framework will handle converting vectors to rasters when needed for
analysis.

Temporal Variables
------------------

Understanding Time in Your Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deforestation analysis often needs to look at how things change over time. The
framework supports **temporal variables** - data that varies across different
years.

For example:

- Forest cover in 2015, 2020, and 2024
- Population growth over time
- Expansion of agricultural areas

When you create your project with multiple years, you can create variables for
each year:

.. code-block:: python

   project = Project(
       project_name="temporal_analysis",
       years=[2015, 2020, 2024]
   )

   # Create forest variable for each year
   for year in project.years:
       forest_var = GEEVar(
           name=f"forest_{year}",
           year=year,  # Associate with specific year
           # ... other parameters
       )

This allows you to analyze:

- How much forest was lost between periods
- Whether deforestation is accelerating or slowing
- How driver variables changed over time

Organizing with Tags
--------------------

Why Use Tags?
~~~~~~~~~~~~~

As your variable collection grows, tags help you stay organized. Tags are
simple labels you attach to variables so you can group and filter them later.

Example tagging strategy:

- **forest**: All forest-related layers
- **infrastructure**: Roads, settlements, airports
- **protected**: Protected areas and conservation zones
- **topography**: Elevation, slope, aspect
- **drivers**: Variables that might cause deforestation

Using tags:

.. code-block:: python

   # Create variable with tags
   forest_var = GEEVar(
       name="forest_2020",
       tags=["forest", "temporal"],
       # ... other parameters
   )

   # Later, find all forest variables
   forest_vars = project.get_variables_by_tag("forest")

The Factory Notebooks
---------------------

Detailed Examples
~~~~~~~~~~~~~~~~~

The ``_1.variables_factory.ipynb`` notebook provides complete, working examples
of:

- Setting up a project from scratch
- Connecting to Google Earth Engine
- Creating variables from different sources
- Organizing variables with tags
- Saving and loading projects

The notebook includes real examples for different regions and use cases. Rather
than copying all that code here, we encourage you to:

1. Open the notebook in Jupyter
2. Read through the explanations
3. Run each cell step by step
4. Modify the examples for your own area

Key Concepts from the Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The notebook demonstrates:

**Project Workflow**

How to create, save, and reload projects so you can work across multiple
sessions.

**AOI Definition**

Multiple ways to define your study area - from country boundaries to custom
polygons.

**Variable Creation Patterns**

Standard approaches for creating different types of variables consistently.

**Data Organization**

How the framework organizes raw versus processed data, and how to keep track of
everything.

**Quality Checks**

Simple ways to verify your variables were created correctly before moving to
processing.

Best Practices
--------------

Tips for Success
~~~~~~~~~~~~~~~~

**1. Start Small**

Begin with a small area and a few variables. Make sure everything works before
scaling up.

**2. Use Meaningful Names**

Name variables descriptively: ``forest_gfc_30_2015`` is better than ``var1``.

**3. Document Your Sources**

Keep notes about where each variable came from and any processing applied.

**4. Check Your AOI**

Visualize your area of interest to make sure it's correct before downloading
data.

**5. Use Tags Consistently**

Develop a tagging scheme and stick to it throughout your project.

**6. Save Regularly**

Use ``project.save()`` frequently to preserve your work.

Common Pitfalls to Avoid
~~~~~~~~~~~~~~~~~~~~~~~~

❌ **Creating too many variables at once** - Start with essential variables only

❌ **Inconsistent naming** - Use a clear naming convention from the start

❌ **Forgetting to specify years** - For temporal analysis, always include the
year

❌ **Not checking data quality** - Visualize variables to catch issues early

❌ **Mixing coordinate systems** - Let the framework handle reprojection
(covered in Process Factory)

Next Steps
----------

Once you've created your variable collection:

- :doc:`process_factory` - Learn how to process and analyze your variables

The process factory tutorial covers aligning all your variables, calculating
distances, analyzing forest loss, and preparing data for modeling.

For detailed code examples, open the notebooks:


Next Steps
----------

Once you've created your variable collection, the next step is processing and
analysis.

For detailed code examples, open the notebooks:

- ``_1.variables_factory.ipynb`` - Complete variable creation examples
- ``_2.process_factory.ipynb`` - Processing and analysis workflows
