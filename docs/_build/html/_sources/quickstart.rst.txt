.. highlight:: python


Quick start
===========
This page will show some examples to load a sourced dataset or to make quicklook plots. For more details, please see
:ref:`userguide`.

More examples can be found here_.

.. _here: https://github.com/JouleCai/geospacelab/tree/master/examples

Dock a dataset
--------------

The example below shows how to dock a sourced dataset (Madrigal/EISCAT data) to the datahub.
See also the code in ``dock_a_dataset.py``.

    .. literalinclude:: ../examples/dock_a_dataset.py
        :language: python
        :linenos:


View the quicklook plots
------------------------------------------------

The quicklook plots are produced by the method "quicklook" of a viewer,
which is custom-designed for a data source. Those specialized viewer can be imported
from ``geospacelab.express``. The two examples below show the solar wind and geomagnetic
indices, as well as the EISCAT data, respectively.

Solar wind and geomagnetic indices from OMNI and WDC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Source code: ``quicklook_omni_wdc.py``

    .. literalinclude:: examples/quicklook_omni_wdc.py
        :language: python
        :linenos:

Output:

    .. image:: examples/OMNI_1min_20160314-060000-20160320-060000.png
        :alt: OMNI solar wind and WDC geomagnetic indices

EISCAT from Madrigal with marking tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Source code: ``quicklook_eiscat.py``

   .. literalinclude:: examples/quicklook_eiscat.py
        :language: python
        :linenos:

Output:

    .. image:: examples/EISCAT_UHF_beata_cp1_2.1u_CP_20201209-180000-20201210-060000.png
        :alt: EISCAT quicklook