.. _NONMEM-section:

======
NONMEM
======

.. note:: This section is for NONMEM documentation. The information is intended for non-documented or hard-to-find information that is important for pharmpy developers. 


Code records
------------

NM-TRAN allows multiple $PK, $PRED and $ERROR in a model. They simple get merged into one big record in the order they have in the control stream. As long as the first $PK/$PRED/$ERROR follows the regular placement rules the others can be placed at any position after that.


Derivatives
-----------

In verbatim code in $ERROR HH(n,j*neps+i) will be the second partial derivative of F(n) with respect to EPS(i) and ETA(j)


Abbreviated and verbatim code
-----------------------------

NM-TRAN sets

.. code-block:: fortran

    IMPLICIT REAL(KIND=DPSIZE) (A-Z)

for the main functions in FSUBS (checked $PK and $ERROR) so variables starting with letters from A-Z doesn't need to be declared, not even in abbreviated code. However if a variable only used in verbatim code is to be used in $TABLE it must be assigned a value in abbreviated code.
