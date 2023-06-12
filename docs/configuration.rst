.. _config_page:

=============
Configuration
=============

You can set up certain global settings via the Pharmpy configuration file. This can be useful if you for example need
to specify the path to NONMEN.

Each module in Pharmpy can have their own configuration. Each configuration item is available through a configuration
object in the module attribute 'conf'.

~~~~~~~~~~~~~~~~~~~~~~
The configuration file
~~~~~~~~~~~~~~~~~~~~~~

Pharmpy can use a configuration file called :code:`pharmpy.conf` that should be placed in the Pharmpy configuration
directory. The placement of this directory is dependent on the operating system. The function
:py:func:`pharmpy.modeling.get_config_path` can be used to see the path to the configuration file path, and if none
exists you can see which path Pharmpy expects the file to be at. It is possible to create an empty template
configuration file using :py:func:`pharmpy.modeling.create_config_template`. To override the default path you can use
the ``PHARMPYCONFIGPATH`` environment variable (see below).

The format of the configuration file is a plain .ini file where each section is the name of the module being
configured. For example:

.. code-block::

   [pharmpy.plugins.nonmem]
   default_nonmem_path=/opt/nmfe751


~~~~~~~~~~~~~~~~~~~~~~~~
Available configurations
~~~~~~~~~~~~~~~~~~~~~~~~

pharmpy.plugins.nonmem
----------------------

+-------------------------+---------------------------------------------------------------+
| Setting                 | Description                                                   |
+=========================+===============================================================+
| ``default_nonmem_path`` | Full path to the default NONMEM installation directory        |
+-------------------------+---------------------------------------------------------------+
| ``licfile``             | Path to the NONMEM license file                               |
+-------------------------+---------------------------------------------------------------+


pharmpy.plugins.nlmixr
----------------------

+-------------------------+---------------------------------------------------------------+
| Setting                 | Description                                                   |
+=========================+===============================================================+
| ``rpath``               | Path to R installation directory                              |
+-------------------------+---------------------------------------------------------------+

pharmpy.plugins.rxode
---------------------

+-------------------------+---------------------------------------------------------------+
| Setting                 | Description                                                   |
+=========================+===============================================================+
| ``rpath``               | Path to R installation directory                              |
+-------------------------+---------------------------------------------------------------+

~~~~~~~~~~~~~~~~~~~~~
Environment variables
~~~~~~~~~~~~~~~~~~~~~

It is also possible to temporarily disable the config file ``PHARMPYCONFIGFILE`` or set a different path to the
Pharmpy config file ``PHARMPYCONFIGPATH``.

+-------------------------+---------------------------------------------------------------+
| Environment variable    | Description                                                   |
+=========================+===============================================================+
| ``PHARMPYCONFIGFILE``   | Set to 1 to not read the configuration file even if it exists |
+-------------------------+---------------------------------------------------------------+
| ``PHARMPYCONFIGPATH``   | Set to path to override the default paths                     |
+-------------------------+---------------------------------------------------------------+

