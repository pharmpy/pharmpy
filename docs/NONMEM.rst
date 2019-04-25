.. _NONMEM-section:

======
NONMEM
======

.. note:: This section is for NONMEM documentation. The information is intended for non-documented or hard-to-find information that is important for pharmpy developers. 


Dataset
-------

Overview of parsing and translation steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Removing commented rows
- Splitting a row into data items
- IGNORE/ACCEPT of rows
- Dropping of columns
- Error handling of non valid items
- Translating TIMEs and DATEs

NM-TRAN dataset parsing
~~~~~~~~~~~~~~~~~~~~~~~

Some rules for the parsing of the dataset by NM-TRAN. These were tested with NONMEM 7.4.3

- Delimeter between items is comma, space or TAB.
- Spaces before or after a comma are ignored
- Spaces after a TAB are ignored
- Spaces before a TAB gives ERROR (sic!)
- Spaces in the beginning or of a row are ignored
- Comma in the end or beginning of a row will insert NULL after or before the comma
- Each item can only be numeric i.e. no other characters than Ee+-0123456789 are allowed except for TIME, II, DATE, DATx columns
- The fortran short form for exponential notation is allowed, i.e. 2-1 means 2e-1 and 2+1 means 2e1
- A lone + or - in an item means 0
- A . (dot) in an item means NULL
- An item can be at most 24 characters long, not counting delimiters and spaces that was eaten by other delimiter
- Empty lines in a dataset will give an error if not BLANKOK is set in $DATA then NULLs are inserted
- As empty lines are counted empty lines and lines only containing spaces and TABs.
- Columns that are DROPed in $INPUT can contain any characters and there is no limit to length of items in such a column
- If any line has more columns than $INPUT all extra columns are considered to be DROPed

NULL items in datasets
~~~~~~~~~~~~~~~~~~~~~~

A NULL item is an item in a dataset that is either a ., surrounded by two TABS or two commas. By default NM-TRAN translates a NULL item to a blank field in FDATA which will be interpreted by NONMEM as 0. Using the NULL option to $DATA the default can be changed. However the NULL option is limited to one character and the only legal ones are [0-9-+] since these are numbers. All other values on $DATA will be met with ERROR from NM-TRAN. A double comma or tab at the end of a row will insert a value at the end after giving a big warning. If the number of columns in $INPUT is larger than the length of some row, NM-TRAN will warn and pad with NULLs. NULL items are inserted after the IGNORE filtering so comment characterscannot be inserted using NULL neither can the IGNORE/ACCEPT filter on NULL.

IGNORE/ACCEPT
~~~~~~~~~~~~~

Some rules for the IGNORE/ACCEPT option in $DATA:

- It is possible to IGNORE on a dropped column
- IGNORE is done before the error check, i.e. columns with text can be ignored
- Text IGNORE (i.e. .EQ. and .NE.) can contain letter + alphanum/underscore or a real number (no special fortran format), + or - (meaning 0), no digit at start, no ., = has to be enclosed in ' or ". Other special characters are ok with or w/o "'.
- With others .EQN, .GE. etc can only use numbers
- A text/number to IGNORE can be enclosed in ' or ".
- A value for comparison can maximum have 12 characters (not counting spaces or '," at start or end)
- IGNORE is performed before translating TIME and DATE


phi files
---------

File format
~~~~~~~~~~~

The default format (could this be changed?) of the table part (except the TABLE NO. lines) of a phi-file (probably the same for other types, but for now only phi has been investigated) is as follows:
First the column header line that is formatted like this:

1. One space
2. All the column names, except the last one, left justified with a field size of 13 characters
3. The final column name

The column names are

1. SUBJECT_ID
2. ID
3. One column per ETA of the model named ETA(n) where n is the number of the ETA starting from 1.
4. A flattened triangular correlation matrix with columns ETC(1,1), ETC(2,1) up to ETC(n,n) with same n as above and all off diagonals given.
5. OBJ

The data is right justified with a field size of 13 characters (padded with spaces). Except for the final OBJ column that is also right justified, but with a field size of 22 characters.

The number format is integer for the first two columns, scientific with 5 decimals and 2 digits exponent for the ETA and ETC columns. The number in the OBJ column is written in regular decimal format and will always take up 19 characters with first character being space for positive numbers or "-" for negative numbers. For numbers having an integer part of zero will take up 20 characters.

$ETAS
~~~~~

The path for the FILE option in $ETAS can have a maximum of 54 characters (tested with NONMEM 7.4.3)


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
