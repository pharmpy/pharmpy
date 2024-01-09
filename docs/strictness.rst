.. _strictness:

==========
Strictness
==========

Strictness criteria for model selection can be specified in the AMD tools.
Models that do not fulfill the strictness criteria will be excluded from the model ranking and will therefore
not be able to be selected as best models.
The strictness argument in the AMD tools consists of a string of logically arranged criteria.
Implemented strictness criteria are:

+----------------------------------+-------------+--------------------------------------------------+
| Strictness criterion             | Type        | Description                                      |
+==================================+=============+==================================================+
| ``minimization_successful``      | Boolean     | True if minimization was successful              |
+----------------------------------+-------------+--------------------------------------------------+
| ``rounding_errors``              | Boolean     | True if minimization terminated due              | 
|                                  |             | rounding errors                                  |
+----------------------------------+-------------+--------------------------------------------------+
| ``sigdigs``                      | Numeric     | Number of significant digits                     |
+----------------------------------+-------------+--------------------------------------------------+
| ``maxevals_exceeded``            | Boolean     | True if minimization terminated due              |
|                                  |             | maximum evaluations exceeded.                    |
+----------------------------------+-------------+--------------------------------------------------+
| ``rse``                          | Numeric     | Relative standard errors of the                  |
|                                  |             | parameters.                                      |
+----------------------------------+-------------+--------------------------------------------------+
| ``rse_theta``                    | Numeric     | Relative standard errors of the                  |
|                                  |             | theta parameters.                                |
+----------------------------------+-------------+--------------------------------------------------+
| ``rse_omega``                    | Numeric     | Relative standard errors of the                  |
|                                  |             | omega parameters.                                |
+----------------------------------+-------------+--------------------------------------------------+
| ``rse_sigma``                    | Numeric     | Relative standard errors of the                  |
|                                  |             | sigma parameters.                                |
+----------------------------------+-------------+--------------------------------------------------+
| ``condition_number``             | Numeric     | Condition number of the covariance               |
|                                  |             | matrix                                           |
+----------------------------------+-------------+--------------------------------------------------+
| ``final_zero_gradient``          | Boolean     | True if at least one parameter has               |
|                                  |             | a final zero gradient or if final                |
|                                  |             | gradient is nan                                  |
+----------------------------------+-------------+--------------------------------------------------+
| ``final_zero_gradient_theta``    | Boolean     | True if at least one theta parameter has         |
|                                  |             | a final zero gradient or if final                |
|                                  |             | gradient is nan                                  |
+----------------------------------+-------------+--------------------------------------------------+
| ``final_zero_gradient_omega``    | Boolean     | True if at least one omega parameter has         |
|                                  |             | a final zero gradient or if final                |
|                                  |             | gradient is nan                                  |
+----------------------------------+-------------+--------------------------------------------------+
| ``final_zero_gradient_sigma``    | Boolean     | True if at least one sigma parameter has         |
|                                  |             | a final zero gradient or if gradient             |
|                                  |             | gradient is nan                                  |
+----------------------------------+-------------+--------------------------------------------------+
| ``estimate_near_boundary``       | Boolean     | True if at least one parameter                   |
|                                  |             | estimate is near its boundary                    |
|                                  |             | (maximum distance to 0 = 0.001, maximum distance |
|                                  |             | to non-zero bound = 2 significant digits         |
+----------------------------------+-------------+--------------------------------------------------+
| ``estimate_near_boundary_theta`` | Boolean     | True if at least one theta parameter             |
|                                  |             | estimate is near its boundary                    |
|                                  |             | (maximum distance to 0 = 0.001, maximum distance |
|                                  |             | to non-zero bound = 2 significant digits         |
+----------------------------------+-------------+--------------------------------------------------+
| ``estimate_near_boundary_omega`` | Boolean     | True if at least one omega parameter             |
|                                  |             | estimate is near its boundary                    |
|                                  |             | (maximum distance to 0 = 0.001, maximum distance |
|                                  |             | to non-zero bound = 2 significant digits         |
+----------------------------------+-------------+--------------------------------------------------+
| ``estimate_near_boundary_sigma`` | Boolean     | True if at least one sigma parameter             |
|                                  |             | estimate is near its boundary                    |
|                                  |             | (maximum distance to 0 = 0.001, maximum distance |
|                                  |             | to non-zero bound = 2 significant digits         |
+----------------------------------+-------------+--------------------------------------------------+

The strictness criteria can be arranged logically, e.g.:

.. code::
   
   "(A or B) and C < n"

where n is a number and A, B and C are strictness criteria.

Allowed logical operators are: ``and``, ``or``, ``not``, ``<``, ``<=``, ``==``, ``>``, ``>=``, ``!=``.

If the statement evaluates to ``False`` then the strictness criteria are not fulfilled and the model will be excluded
from the results.

Examples
========

.. code::

    strictness = "minimization_successful or (rounding_errors and sigdigs >= 0.1)"

In this example the strictness criteria states that either the minimization must be successful or else the
minimization was terminated due to rounding errors and the number of sigdigs is greater or equal to 0.1.
This example is the default strictness criterion for AMD and all subtools.

.. code::

    strictness = "minimization_successful and rse < 0.4"

This means that the minimization must be successful and that all parameters must have an RSE smaller than 0.4.
