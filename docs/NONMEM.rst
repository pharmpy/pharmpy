.. _NONMEM-section:

======
NONMEM
======

.. note:: This section is for NONMEM documentation. The information is intended for non-documented or hard-to-find information that is important for pharmpy developers. 


NULL items in datasets
----------------------

A NULL item is an item in a dataset that is either a ., surrounded by two TABS or two commas. By default NM-TRAN translates a NULL item to a blank field in FDATA which will be interpreted by NONMEM as 0. Using the NULL option to $DATA the default can be changed. However the NULL option is limited to one character and the only legal ones are [0-9-+] since these are numbers. All other values on $DATA will be met with ERROR from NM-TRAN. A double comma or tab at the end of a row will insert a value at the end after giving a big warning. If the number of columns in $INPUT is larger than the length of some row, NM-TRAN will warn and pad with NULLs. NULL items are inserted after the IGNORE filtering so comment characters cannot be inserted using NULL. 


Code records
------------

NM-TRAN allows multiple $PK, $PRED and $ERROR in a model. They simple get merged into one big record in the order they have in the control stream. As long as the first $PK/$PRED/$ERROR follows the regular placement rules the others can be placed at any position after that.


Derivatives
-----------

In verbatim code in $ERROR HH(n,j*neps+i) (in $PRED it is called H) will be the second partial derivative of F(n) with respect to EPS(i) and ETA(j)


Abbreviated and verbatim code
-----------------------------

NM-TRAN sets

.. code-block:: fortran

    IMPLICIT REAL(KIND=DPSIZE) (A-Z)

for the main functions in FSUBS (checked $PK and $ERROR) so variables starting with letters from A-Z doesn't need to be declared, not even in abbreviated code. However if a variable only used in verbatim code is to be used in $TABLE it must be assigned a value in abbreviated code.
