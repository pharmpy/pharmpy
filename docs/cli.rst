==================================
The Pharmpy command line interface
==================================

Pharmpy has a command line interface for some of its functionality. The command line tool is an integrated part of the Pharmpy python package.

The main command is ```pharmpy``` and the functionality is divided into subcommands with two levels. The first and top level most often represents a type of object to perform on operation on. For example ```model``` or ```data```.  (The built in help system will give an overview of available subcommands with ```pharmpy -h```. The second level is most often an operation or verb to perform. After the main command and the two subcommands follows the input and options of the particular command.


.. autoprogram:: pharmpy.cli:parser
   :prog: pharmpy
