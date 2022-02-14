.. _projects:

Project suggestions
===================

Here are some suggestions for projects suitable for Google Summer of Code, Master Thesis Work or other types of internships. If you are a student or otherwise interested feel free to drop any of the `maintainers <https://pharmpy.github.io/latest/contributors.html>`_ and email.


* | Step wise covariate search (scm)
  | Implement a known algorithm and workflow for covariate search in a model.
  | This will involve using the Pharmpy API and dask to create a workflow.

* | Estimation tool
  | Implement the FO non-linear mixed effects model parameter estimation method.
  | This will involve connecting sympy expressions with data in pandas and using
  | scipy for optimization.
	
* | Simulation tool
  | Design interface for a simulation tool and create two implementations. One using some
  | external software, e.g. NONMEM and one internal
	
* | Switch lark parser from Earley to LALR(1)
  | Grammars needs to be adapted to LALR(1) constraints
	
* | Switch to using symengine instead if sympy for statements
	
* | Allow using median() and mean() in symbolic expressions
  | A sympy meets pandas task

* | Model diff
  | Create a diff tool to see differences between two models on different levels
  | On model code level, on Pharmpy object level and potentially on a high human readable level

* | Plotting in altair, holoviews or bokeh
  | Implement visualizations to compare different libraries

* | Use pyarrow to read datasets or result tables
  | How fast can we read datasets? pyarrow meets pandas

* | Optimize dask local runs
  | What are the bottlenecks when running dask locally and are there optimizations that can be made?
