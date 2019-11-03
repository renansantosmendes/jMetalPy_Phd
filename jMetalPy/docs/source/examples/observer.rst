Observer entities
========================

It is possible to attach any number of observers to any algorithm to get information from each iteration.
For example, a basic algorithm observer will print the number of evaluations, the objectives from the best individual in the population and the computing time:

.. code-block:: python

   basic = BasicAlgorithmObserver(frequency=1.0)
   algorithm.observable.register(observer=basic)

A progress bar observer will print a `smart progress meter <https://github.com/tqdm/tqdm>`_ that increases, on each iteration, a fixed value (`step`) until the maximum is reached.

.. code-block:: python

   max_evaluations = 25000

   algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=BitFlipMutation(1.0 / problem.number_of_bits),
        crossover=SPXCrossover(0.9),
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(max=25000)
    )

   progress_bar = ProgressBarObserver(max=max_evaluations)
   algorithm.observable.register(progress_bar)

   algorithm.run()

.. code-block:: console

   $ Progress:  50%|#####     | 12500/25000 [13:59<14:12, 14.66it/s]

A full list of all available observers can be found at the :py:mod:`jmetal.util.observer` module.
