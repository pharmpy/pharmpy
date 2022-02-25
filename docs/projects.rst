.. _projects:

Project suggestions
===================

Here are some suggestions for projects suitable for Google Summer of Code, Master Thesis Work or other types of internships. If you are a student or otherwise interested feel free to drop any of the `maintainers <https://pharmpy.github.io/latest/contributors.html>`_ an email.
In our team we are using Python, tox and pytest for testing, github actions for continuous integration and sphinx for documentation.

* | **Step wise covariate search (scm)**
  | Implement a known step wise search algorithm and workflow for covariate search in a model.
  | Covariates are factors such as body weight and sex that help explain
  | variability between individuals in for drug trial data.
  | This will involve using the Pharmpy API to create a workflow.

    * Outcomes: A tool that could be directly used by researchers
    * Skills: Python, pandas
    * Size: 350h
    * Difficulty: Medium
    * Mentors: Rikard Nordgren, Stella Belin and/or Zhe Huang

* | **Estimation tool**
  | Implement the first order (FO) non-linear mixed effects model parameter estimation method.
  | This will involve connecting sympy expressions with data in pandas and using
  | scipy for optimization. The main focus of the project will be on the function optimization part.

    * Outcomes: A tool that will serve as a proof of concept and also to simplify internal testing
    * Skills: Python, pandas, sympy, scipy, linear algebra, optimization	
    * Size: 350h
    * Difficulty: Hard
    * Mentors: Rikard Nordgren, Stella Belin and/or Zhe Huang

* | **Simulation tool**
  | Design interface for a simulation tool and create one or possibly two implementations. One using some
  | external software, e.g. NONMEM and one implemented from scratch. Simulation in this context is to
  | generate observations (for example plasma drug concentrations) from a model.

    * Outcomes: A simulation tool that will simplify simulation for researchers
    * Skills: Python, pandas
    * Size: 350h
    * Difficulty: Medium
    * Mentors: Rikard Nordgren, Stella Belin and/or Zhe Huang

* | **Switch lark parser from Earley to LALR(1)**
  | Model files are currently parsed with an earley parser using the lark Python package. Lark
  | also supports an LALR(1) parser, which is much faster.
  | Grammars needs to be adapted to the stricter LALR(1) constraints.

    * Outcomes: Much faster parsing of model description files
    * Skills: Python, parsing
    * Size: 175h
    * Difficulty: Hard
    * Mentors: Rikard Nordgren, Stella Belin and/or Zhe Huang
	
* | **Switch to using symengine instead if sympy for statements**
  | Currently sympy is used for almost all handling of mathematical expressions.
  | Since symengine is mostly compatible with sympy and faster it would be
  | beneficial to try to use symengine where applicable.

    * Outcomes: Faster reading and handling of models
    * Skills: Python, sympy
    * Size: 175h
    * Difficulty: Easy
    * Mentors: Rikard Nordgren, Stella Belin and/or Zhe Huang
	
* | **Allow using median() and mean() in symbolic expressions**
  | A sympy meets pandas task. Allow custom functions in sympy
  | expressions. These functions should be possible to evaluate
  | numerically. For example "median(WGT) * theta" where "WGT"
  | is available in a DataFrame.

    * Outcomes: More powerful input expressions for researchers
    * Skills: Python, sympy, pandas
    * Size: 175h
    * Difficulty: Easy
    * Mentors: Rikard Nordgren, Stella Belin and/or Zhe Huang

* | **Model diff**
  | Create a diff tool to see differences between two models on different levels
  | On model code level (text), on Pharmpy object level and potentially on a high human readable level
  | separating different high level model concepts.

    * Outcomes: One or more model diff tools
    * Skills: Python
    * Size: 175h-350h
    * Difficulty: Easy-Hard
    * Mentors: Rikard Nordgren, Stella Belin and/or Zhe Huang

* | **Compare plotting in various plotting libraries**
  | Implement different standard plots to compare different libraries. Different areas of comparison
  | could be ease of use, serialization of plots, interactivity and rendering in Jupyter and Rstudio. 
  | Examples of plotting libraries to explore are altair, holoviews and bokeh.

    * Outcomes: Example plots using various tools and a report on their suitability
    * Skills: Python, pandas, plotting
    * Size: 175h
    * Difficulty: Easy
    * Mentors: Rikard Nordgren, Stella Belin and/or Zhe Huang

* | **Monitor ongoing workflows**
  | Some workflow take a very long time to run and researchers would benefit from being able to
  | monitor what is happening. For example convergence of a parameter estimation. We would
  | like to develop some sort of dashboard that could complement the dask dashboard with
  | tool specific information that gets updated in realtime.

    * Outcomes: A dashboard
    * Skills: Python, dask, plotting
    * Size: 350h
    * Difficulty: Medium
    * Mentors: Rikard Nordgren, Stella Belin and/or Zhe Huang

* | **Restartable workflows**
  | Long running workflows might for different reasons fail in the middle. Currently worflows will have
  | to be restarted after fixing the cause of failure. We would like to be able to restart workflows
  | and use partial results and continue.

    * Outcomes: A general idea for restartability and one or more workflows becoming restartable.
    * Skills: Python, dask
    * Size: 350h
    * Difficulty: Hard
    * Mentors: Rikard Nordgren, Stella Belin and/or Zhe Huang
