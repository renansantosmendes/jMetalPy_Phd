from jmetalpy.algorithm.multiobjective.nsgaii import NSGAII
from jmetalpy.lab.visualization import Plot, InteractivePlot
from jmetalpy.operator import SBXCrossover, PolynomialMutation, DifferentialEvolutionCrossover
from jmetalpy.problem import DTLZ2
from jmetalpy.util.observer import ProgressBarObserver, VisualizerObserver
from jmetalpy.util.solutions import print_function_values_to_file, print_variables_to_file
from jmetalpy.util.solutions.comparator import DominanceComparator
from jmetalpy.util.termination_criterion import StoppingByEvaluations
from jmetalpy.core.quality_indicator import *
from jmetalpy.lab.experiment import Experiment, Job, generate_summary_from_experiment
from jmetalpy.algorithm.multiobjective.moead import MOEAD
from jmetal.util.aggregative_function import Tschebycheff

if __name__ == '__main__':
    number_of_objectives = 4
    problem = DTLZ2(number_of_objectives=number_of_objectives)
    #problem.reference = '../../resources/reference_front/DTLZ2.3D.pf'

    max_evaluations = 10000
    algorithm = MOEAD(
        problem=problem,
        population_size=100,
        crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5, K=0.5),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        aggregative_function=Tschebycheff(dimension=problem.number_of_objectives),
        neighbor_size=20,
        neighbourhood_selection_probability=0.9,
        max_number_of_replaced_solutions=2,
        weight_files_path='/../../resources/MOEAD_weights',#
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )
    
    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    #algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))

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
        quality_indicators=[HyperVolume([1.0, 1.0, 1.0, 1.0])]
    )

    #algorithm.run()
    # front = jobs[0].algorithm.get_result()

    # # Plot front
    # plot_front = Plot(plot_title='Pareto front approximation', reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    # plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # # Plot interactive front
    # plot_front = InteractivePlot(plot_title='Pareto front approximation', reference_front=problem.reference_front,
    #  axis_labels=problem.obj_labels)
    # plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # # Save results to file
    # print_function_values_to_file(front, 'FUN.' + algorithm.label)
    # print_variables_to_file(front, 'VAR.'+ algorithm.label)

    # print('Algorithm (continuous problem): ' + algorithm.get_name())
    # print('Problem: ' + problem.get_name())
    # print('Computing time: ' + str(algorithm.total_computing_time))
    # print('Size of the last front', len(front))
    
    print(algorithm.solutions[0])