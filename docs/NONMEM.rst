.. _NONMEM-section:

======
NONMEM
======

.. note:: This section is for NONMEM documentation. The information is intended for non-documented or hard-to-find information that is important for pharmpy developers. 


$PK and $PRED

NM-TRAN allows multiple $PK and $PRED in a model. They simple get merged into one big record in the order they have in the control stream. As long as the first $PK/$PRED follows the regular placement rules the others can be placed at any position after that.

Abbreviated and verbatim code

NM-TRAN sets
.. code-block:: fortran

    IMPLICIT REAL(KIND=DPSIZE) (A-Z)

for the main functions in FSUBS (checked $PK and $ERROR) so variables starting with letters from A-Z doesn't need to be declared. Not even in abbreviated code.
