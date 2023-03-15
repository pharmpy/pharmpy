.. _config_page:

=============
Configuration
=============

Each module in Pharmpy can have its own configuration. Each configuration item is available through a configuration
object in the module attribute 'conf'.

~~~~~~~~~~~~~~~~~~~~~~
The configuration file
~~~~~~~~~~~~~~~~~~~~~~

Pharmpy can use a configuration file called :code:`pharmpy.conf` that should be placed in the `Pharmpy` configuration
directory. The placement of this directory is dependent on the operating system. Use the `get_config_path` in `modeling` to
find out the configuration file path. It is also possible to override the default path by using the `PHARMPYCONFIGPATH` environment
variable (see below).

The format of the configuration file is a plain .ini file where each section is the name of the module being
configured. For example:

.. code-block::

   [pharmpy.plugins.nonmem]
   default_nonmem_path=/opt/nmfe751

will set the path to nonmem.

It is possible to create an empty template configuration file using the `create_default_config` function.

~~~~~~~~~~~~~~~~~~~~~
Environment variables
~~~~~~~~~~~~~~~~~~~~~

+------------------------+---------------------------------------------------------------+
| Environment variable   | Description                                                   |
+========================+===============================================================+
| PHARMPYCONFIGFILE      | Set to 1 to not read the configuration file even if it exists |
+------------------------+---------------------------------------------------------------+
| PHARMPYCONFIGPATH      | Set to path to override the default paths                     |
+------------------------+---------------------------------------------------------------+
