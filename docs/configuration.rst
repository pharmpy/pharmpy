=============
Configuration
=============

Each module in Pharmpy can have its own configuration. Each configuration item is available through a configuration object in the module attribute 'conf'.




~~~~~~~~~~~~~~~~~~~~~~
The configuration file
~~~~~~~~~~~~~~~~~~~~~~

Pharmpy can use a configuration file called :code:`pharmpy.conf` that should be placed in the `Pharmpy` configuration directory. The placement of this directory is dependent on the operating system. As Pharmpy is using the appdirs package to find the placement of configuration files in your system please check the appdirs package web page: https://pypi.org/project/appdirs/ for more information. 

The format of the configuration file is a plain ini file where each section is the name of the module being configured. For example

.. code-block::

   [pharmpy.plugins.nonmem]
   parameter_names=comment

will set the configuration item `parameter_names` to `comment` in the nonmem module.

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
