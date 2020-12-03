================
Plots in Pharmpy
================

Plotting in Python is done using the Altair package, which is based on Vega and Vega-Lite.

~~~~~~~~~~~~~~
Setting themes
~~~~~~~~~~~~~~

It is possible to change the theme of the plots from the default to for example ggplot.

1. Open the plot in the Vega editor
2. Choose the Config tab
3. Select theme

~~~~~~~~~~~~~~~~~~~~~~
Changing axis of plots
~~~~~~~~~~~~~~~~~~~~~~

To modify the scaling of the x-axis add ``"domain": [-100, 100], "clamp": true`` in vconcat -> spec -> encoding ->
x -> scale.

.. code-block::

    "vconcat": [
      {
        ...,
        "spec": {
            "layer": [
                {
                ...,
                "encoding": {
                    "x": {
                        ...,
                        "scale": {"zero": false, "domain": [-100, 100], "clamp": true},
    ...
    ],

To have different limits for the plots, set ``"resolve": {"scale": {"x": "independent"}``

.. code-block::

    "vconcat": [
    ...
    ],
    "resolve": {"scale": {"x": "shared"}},
    ...

