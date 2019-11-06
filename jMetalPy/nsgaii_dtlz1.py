from jmetalpy.algorithm.multiobjective.nsgaii import NSGAII
from jmetalpy.lab.visualization import Plot, InteractivePlot
from jmetalpy.operator import SBXCrossover, PolynomialMutation
from jmetalpy.problem import DTLZ2
from jmetalpy.util.observer import ProgressBarObserver, VisualizerObserver
from jmetalpy.util.solutions import print_function_values_to_file, print_variables_to_file
from jmetalpy.util.solutions.comparator import DominanceComparator
from jmetalpy.util.termination_criterion import StoppingByEvaluations
from jmetalpy.core.quality_indicator import *
from jmetalpy.lab.experiment import Experiment, Job, generate_summary_from_experiment

if __name__ == '__main__':
    problem = DTLZ2()
    #problem.reference = '../../resources/reference_front/DTLZ2.3D.pf'

    max_evaluations = 4000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations),
        dominance_comparator=DominanceComparator()
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))

    jobs = [Job(algorithm=algorithm,
        algorithm_tag='NSGAII',
        problem_tag=problem.get_name(),
        run=1)]

    output_directory = '/home/renansantos/Área de Trabalho/data'

    experiment = Experiment(output_dir=output_directory, jobs=jobs)
    experiment.run()

    # Generate summary file
    generate_summary_from_experiment(
        input_dir=output_directory,
        reference_fronts='/home/user/jMetalPy/resources/reference_front',
        quality_indicators=[HyperVolume([1.0, 1.0, 1.0])]
    )

    #algorithm.run()
    front = jobs[0].algorithm.get_result()

    # Plot front
    plot_front = Plot(plot_title='Pareto front approximation', reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # Plot interactive front
    plot_front = InteractivePlot(plot_title='Pareto front approximation', reference_front=problem.reference_front,
     axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.label)
    print_variables_to_file(front, 'VAR.'+ algorithm.label)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
    print('Size of the last front', len(front))
    
    # pareto_dominance = DominanceComparator()
    # non_dominated_solutions = []
    # for i in range(len(front)):
    #     for j in range(len(front)):
    #         if i != j:
    #             if pareto_dominance.compare(front[i],front[j]) == 0:
    #                 non_dominated_solutions.append(front[i])
    
    from jmetalpy.util.ranking import FastNonDominatedRanking                
    print('Non-dominated solutions size',
           len(FastNonDominatedRanking().compute_ranking(front,100)))