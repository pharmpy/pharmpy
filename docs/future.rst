.. _future:

======
Future
======

General plans and thoughts about future development.

Wishlist
========

A wishlist of Pharmpy features that would be attractive to implement (difficult/impossible with PsN today).

NONMEM ``PRED`` nonzero exit
----------------------------

**Feature wish**: Diagnosing ``NONZERO VALUE OF ETA``-search errors (and using numbers) in ``PRDERR``.

**Problem**: You estimate. NONMEM tells you::

   1THERE ARE ERROR MESSAGES IN FILE PRDERR

So you look and there you find:

.. code:: Fortran

   ON WORKER: WORKER1,ON DIRECTORY: worker1/: Problem=1 Subproblem=0 Superproblem1=0 Iteration1=0 Superproblem2=0 Iteration2=0
   0PRED EXIT CODE = 1
   0INDIVIDUAL NO.      94   ID= 1.10200650000000E+07   (WITHIN-INDIVIDUAL) DATA REC NO.  20
    THETA=
     6.21E+01   5.35E-02   7.96E-02   9.69E-02   6.27E-01   7.75E-03  -2.81E-03   0.00E+00   2.63E-02   3.12E+00
     2.15E-02   1.20E+00   1.55E-01   2.56E+00   2.96E-03   0.00E+00  -1.68E+00   8.94E-01   2.13E+00   6.21E-03
    -3.73E-01  -1.60E-01   0.00E+00   7.05E+01  -9.59E-03   7.18E+01   6.31E+01   3.33E-01   6.37E-01
    ETA=
    -1.65E+01  -4.89E+00   4.23E+00  -1.79E+00   9.14E-01   3.31E+00   9.48E+00   1.07E+01   9.94E-01  -2.01E+00
     1.21E+00   6.45E-01
    OCCURS DURING SEARCH FOR ETA AT A NONZERO VALUE OF ETA
   ON WORKER: WORKER2,ON DIRECTORY: worker2/
   0PRED EXIT CODE = 1
   0INDIVIDUAL NO.     214   ID= 1.12000220000000E+07   (WITHIN-INDIVIDUAL) DATA REC NO.  11
    THETA=
     6.21E+01   5.35E-02   7.96E-02   9.69E-02   6.27E-01   7.75E-03  -2.81E-03   0.00E+00   2.63E-02   3.12E+00
     2.15E-02   1.20E+00   1.55E-01   2.56E+00   2.96E-03   0.00E+00  -1.68E+00   8.94E-01   2.13E+00   6.21E-03
    -3.73E-01  -1.60E-01   0.00E+00   7.05E+01  -9.59E-03   7.18E+01   6.31E+01   3.33E-01   6.37E-01
    ETA=
    -1.61E+01  -4.75E+00   4.64E+00  -3.71E-01  -2.02E-01   4.27E+00   2.39E+01   2.40E-01  -6.70E-02   2.04E-01
    -2.83E-01   3.03E-01
    OCCURS DURING SEARCH FOR ETA AT A NONZERO VALUE OF ETA
   ON WORKER: WORKER4,ON DIRECTORY: worker4/
   0PRED EXIT CODE = 1
   [..]

**Idea**: You of course want to debug it to *fix it*:

- What does these values mean? Why did it break on those?
- How does the vector compare to the *other* (individual) posteriors? The population (distribution)?
- The data for those individuals?
- The *specific record*? Shouldn't we be able to figure out why that breaks?
