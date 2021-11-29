.. highlight:: python


Quick Start
============

This page will show some examples to load a sourced dataset or to make quicklook plots. For more details, please refer to
:ref:`the user manual <usermanual>`.

More examples can be found here_.

.. _here: https://github.com/JouleCai/geospacelab/tree/master/examples

Use the Datahub and Dock a Dataset
----------------------------------

The example below shows how to dock a sourced dataset (Madrigal/EISCAT data) to a datahub. Please refer to
:ref:`the list of the data sources <datasources>` for loading other geospace data.

.. literalinclude:: ../examples/dock_a_dataset.py
    :language: python
    :caption: examples/dock_a_dataset.py


Use the Time-Series Viewer and Create a Figure
-----------------------------------------------


Add Indicators
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the Geomap Viewer and Create a Map
-------------------------------------------------------

Use the Express Viewer and Create a Figure
--------------------------------------------------

The quicklook plots are produced by the method "quicklook" of a viewer,
which is custom-designed for a data source. Those specialized viewer can be imported
from ``geospacelab.express``. The two examples below show the solar wind and geomagnetic
indices, as well as the EISCAT data, respectively.

Solar Wind and Geomagnetic Indices from OMNI and WDC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Solar wind and geomagnetic indices data

    .. literalinclude:: ../examples/demo_omni_data.py
        :language: python
        :caption: examples/demo_omni_data.py

Output:

    .. image:: ../examples/OMNI_1min_20160314-060000-20160320-060000.png
        :alt: IMF and solar wind from the OMNI database and the geomagnetic indices from WDC and GFZ

EISCAT from Madrigal with Marking Tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   .. literalinclude:: ../examples/demo_eiscat_quicklook.py
        :language: python
        :linenos:
        :caption: examples/demo_eiscat_quicklook.py

Output:

    .. image:: ../examples/EISCAT_UHF_beata_cp1_60_20201209-180000-20201210-060000.png
        :alt: EISCAT quicklook

DMSP/SSUSI auroral images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   .. literalinclude:: ../examples/demo_dmsp_ssusi_single_panel.py
        :language: python
        :linenos:
        :caption: examples/demo_dmsp_ssusi_single_panel.py

Output:

    .. image:: ../examples/DMSP_SSUSI_20150908-2021_LBHS_F18_N.png
        :alt: DMSP SSUSI image