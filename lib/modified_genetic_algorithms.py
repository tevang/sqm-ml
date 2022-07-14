# ConsScorTK is released under the GNU General Public License v3 (see http://www.gnu.org/licenses/gpl.txt).
# If you have any questions, comments, or suggestions, please don't hesitate to contact me, Thomas Evangelidis,
# at tevang3@gmail.com . If you use ConsScorTK in your work, please cite [REFERENCE HERE].

#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

"""The :mod:`algorithms` module is intended to contain some specific algorithms
in order to execute very common evolutionary algorithms. The method used here
are more for convenience than reference as the implementation of every 
evolutionary algorithm may vary infinitely. Most of the algorithms in this 
module use operators registered in the toolbox. Generaly, the keyword used are
:meth:`mate` for crossover, :meth:`mutate` for mutation, :meth:`~deap.select`
for selection and :meth:`evaluate` for evaluation.

You are encouraged to write your own algorithms in order to make them do what
you really want them to do.
"""

import os
import pickle
import random
import sys

import lib.utils.print_functions

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from . import modified_tools as tools
from . import ConsScoreTK_Statistics

def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.
    
    :param population: A list of individuals to variate.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.
    
    The variator goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.
    A first loop over :math:`P_\mathrm{o}` is executed to mate consecutive
    individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting
    :math:`P_\mathrm{o}` is returned.
    
    This variation is named *And* beceause of its propention to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematicaly, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]
    
    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
            del offspring[i-1].fitness.values, offspring[i].fitness.values
    
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
    
    return offspring

def eaSimple_verbose(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, ScoringFunctionNames_list=None, logfile=None, logbook=None, start_gen=0):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.
    
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population.
    
    It uses :math:`\lambda = \kappa = \mu` and goes as follow.
    It first initializes the population (:math:`P(0)`) by evaluating
    every individual presenting an invalid fitness. Then, it enters the
    evolution loop that begins by the selection of the :math:`P(g+1)`
    population. Then the crossover operator is applied on a proportion of
    :math:`P(g+1)` according to the *cxpb* probability, the resulting and the
    untouched individuals are placed in :math:`P'(g+1)`. Thereafter, a
    proportion of :math:`P'(g+1)`, determined by *mutpb*, is 
    mutated and placed in :math:`P''(g+1)`, the untouched individuals are
    transferred :math:`P''(g+1)`. Finally, those new individuals are evaluated
    and the evolution loop continues until *ngen* generations are completed.
    Briefly, the operators are applied in the following order ::
    
        evaluate(population)
        for i in range(ngen):
            offspring = select(population)
            offspring = mate(offspring)
            offspring = mutate(offspring)
            evaluate(offspring)
            population = offspring
    
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.
    
    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    if start_gen == 0:  # if this is not a restart, randomize the seed according to current time
        random.seed()
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    if stats is not None:
        stats.update(population)
    if verbose:
        column_names = ["gen", "evals"]
        if stats is not None:
            column_names += list(stats.functions.keys())
        logger = tools.EvolutionLogger(column_names)
        logger.logHeader()
        logger.logGeneration(evals=len(population), gen=start_gen, stats=stats)

    # Begin the generational process
    for gen in range(1+start_gen, ngen+1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        
        # Variate the pool of individuals (mutation AND cross-over). The modified individuals have their fitness invalidated
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        
        # Evaluate the individuals with an invalid fitness (mutated or crossed-over)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)    # offspring is the population of all individuals after mutation and cross-over to some of them
            
        # Replace the current population by the offspring
        population[:] = offspring
            
        # Update the statistics with the new population
        if stats is not None:
            stats.update(population)
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if verbose:
            logger.logGeneration(evals=len(invalid_ind), gen=gen, stats=stats)
        
        # write checkpoint file
        cp = dict(population=population, generation=gen, halloffame=halloffame, logbook=logbook, rndstate=random.getstate()) # Fill the dictionary using the dict(key=value[, ...]) constructor
        pickle.dump(cp, open("scoringfunction_selection_checkpoint_gen"+str(gen)+".pkl", "wb"))
        
        with open(logfile, 'a') as f:
            f.write("Generation "+str(gen)+"\n")
            for cse_index in range(0, len(halloffame)): # HallOfFame stores the individuals based on their fitness value in ascending order
                best_scoring_functions_string=""
                for indx in set(halloffame[cse_index]):
                    if indx >= 0:
                        best_scoring_functions_string+="  "+ScoringFunctionNames_list[indx]
                f.write("Suggested Scoring Function Combination No "+str(cse_index+1)+":"+best_scoring_functions_string+"\n")
                print(lib.utils.print_functions.bcolors.BOLDGREEN + "Suggested Scoring Function Combination No " + str(cse_index + 1) + ":", best_scoring_functions_string + lib.utils.print_functions.bcolors.ENDBOLD) # return the scoring functions that maximize the objective function
        
    return population

def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.
    
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population.
    
    It uses :math:`\lambda = \kappa = \mu` and goes as follow.
    It first initializes the population (:math:`P(0)`) by evaluating
    every individual presenting an invalid fitness. Then, it enters the
    evolution loop that begins by the selection of the :math:`P(g+1)`
    population. Then the crossover operator is applied on a proportion of
    :math:`P(g+1)` according to the *cxpb* probability, the resulting and the
    untouched individuals are placed in :math:`P'(g+1)`. Thereafter, a
    proportion of :math:`P'(g+1)`, determined by *mutpb*, is 
    mutated and placed in :math:`P''(g+1)`, the untouched individuals are
    transferred :math:`P''(g+1)`. Finally, those new individuals are evaluated
    and the evolution loop continues until *ngen* generations are completed.
    Briefly, the operators are applied in the following order ::
    
        evaluate(population)
        for i in range(ngen):
            offspring = select(population)
            offspring = mate(offspring)
            offspring = mutate(offspring)
            evaluate(offspring)
            population = offspring
    
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.
    
    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    if stats is not None:
        stats.update(population)
    if verbose:
        column_names = ["gen", "evals"]
        if stats is not None:
            column_names += list(stats.functions.keys())
        logger = tools.EvolutionLogger(column_names)
        logger.logHeader()
        logger.logGeneration(evals=len(population), gen=0, stats=stats)

    # Begin the generational process
    for gen in range(1, ngen+1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        
        # Variate the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
            
        # Replace the current population by the offspring
        population[:] = offspring
        
        # Update the statistics with the new population
        if stats is not None:
            stats.update(population)

        if verbose:
            logger.logGeneration(evals=len(invalid_ind), gen=gen, stats=stats)

    return population

def varAndConvergence(population, toolbox, cxpb, mutpb, gen, eta_stride):
    """Modified: Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.
    
    :param population: A list of individuals to variate.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.
    
    The variator goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.
    A first loop over :math:`P_\mathrm{o}` is executed to mate consecutive
    individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting
    :math:`P_\mathrm{o}` is returned.
    
    This variation is named *And* beceause of its propention to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematicaly, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]
    if eta_stride >= 1:
        myeta = 2*(gen/eta_stride)  # increase myeta at each generation
    elif eta_stride <= 0:
        myeta = -1*eta_stride  # in this case keep myeta constant in every generation
        
    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
            del offspring[i-1].fitness.values, offspring[i].fitness.values
    
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i], eta=myeta)
            del offspring[i].fitness.values
    
    return offspring


def eaSimpleConvergence(population, toolbox, cxpb, mutpb, ngen, eta_stride, stats=None,
             halloffame=None, verbose=__debug__):
    """Modified: This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.
    
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param eta_stride:  The number of generation needed to pass in order eta to be increase by 4
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population.
    
    It uses :math:`\lambda = \kappa = \mu` and goes as follow.
    It first initializes the population (:math:`P(0)`) by evaluating
    every individual presenting an invalid fitness. Then, it enters the
    evolution loop that begins by the selection of the :math:`P(g+1)`
    population. Then the crossover operator is applied on a proportion of
    :math:`P(g+1)` according to the *cxpb* probability, the resulting and the
    untouched individuals are placed in :math:`P'(g+1)`. Thereafter, a
    proportion of :math:`P'(g+1)`, determined by *mutpb*, is 
    mutated and placed in :math:`P''(g+1)`, the untouched individuals are
    transferred :math:`P''(g+1)`. Finally, those new individuals are evaluated
    and the evolution loop continues until *ngen* generations are completed.
    Briefly, the operators are applied in the following order ::
    
        evaluate(population)
        for i in range(ngen):
            offspring = select(population)
            offspring = mate(offspring)
            offspring = mutate(offspring)
            evaluate(offspring)
            population = offspring
    
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.
    
    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    if stats is not None:
        stats.update(population)
    if verbose:
        column_names = ["gen", "evals"]
        if stats is not None:
            column_names += list(stats.functions.keys())
        logger = tools.EvolutionLogger(column_names)
        logger.logHeader()
        logger.logGeneration(evals=len(population), gen=0, stats=stats)

    # Begin the generational process
    for gen in range(1, ngen+1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        
        # Variate the pool of individuals
        offspring = varAndConvergence(offspring, toolbox, cxpb, mutpb, gen, eta_stride)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
            
        # Replace the current population by the offspring
        population[:] = offspring
        
        # Update the statistics with the new population
        if stats is not None:
            stats.update(population)

        if verbose:
            logger.logGeneration(evals=len(invalid_ind), gen=gen, stats=stats)

    return population


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.
    
    :param population: A list of individuals to variate.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.
    
    The variator goes as follow. On each of the *lambda_* iteration, it
    selects one of the three operations; crossover, mutation or reproduction.
    In the case of a crossover, two individuals are selected at random from
    the parental population :math:`P_\mathrm{p}`, those individuals are cloned
    using the :meth:`toolbox.clone` method and then mated using the
    :meth:`toolbox.mate` method. Only the first child is appended to the
    offspring population :math:`P_\mathrm{o}`, the second child is discarded.
    In the case of a mutation, one individual is selected at random from
    :math:`P_\mathrm{p}`, it is cloned and then mutated using using the
    :meth:`toolbox.mutate` method. The resulting mutant is appended to
    :math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
    selected at random from :math:`P_\mathrm{p}`, cloned and appended to
    :math:`P_\mathrm{o}`.
    
    This variation is named *Or* beceause an offspring will never result from
    both operations crossover and mutation. The sum of both probabilities
    shall be in :math:`[0, 1]`, the reproduction probability is
    1 - *cxpb* - *mutpb*.
    """
    assert (cxpb + mutpb) <= 1.0, ("The sum of the crossover and mutation "
        "probabilities must be smaller or equal to 1.0.")
    
    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))
    
    return offspring


def eaMuPlusLambda_verbose(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, start_gen=0,
                   stats=None, halloffame=None, verbose=__debug__, ScoringFunctionNames_list=None, WEIGHTS=None, SOLUTIONS=None, logfile=None, logbook=None):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population.
    
    First, the individuals having an invalid fitness are evaluated. Then, the
    evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by a crossover, a mutation or a
    reproduction proportionally to the probabilities *cxpb*, *mutpb* and 1 -
    (cxpb + mutpb). The offspring are then evaluated and the next generation
    population is selected from both the offspring **and** the population.
    Briefly, the operators are applied as following ::
    
        evaluate(population)
        for i in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)
    
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    if start_gen == 0:  # if this is not a restart, randomize the seed according to current time
        random.seed()
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    if stats is not None:
        stats.update(population)
    if verbose:
        column_names = ["gen", "evals"]
        if stats is not None:
            column_names += list(stats.functions.keys())
        logger = tools.EvolutionLogger(column_names)
        logger.logHeader()
        logger.logGeneration(evals=len(population), gen=start_gen, stats=stats)

    # Begin the generational process
    for gen in range(1+start_gen, ngen+1):
        # Variate the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
        
        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)
        
        # Update the statistics with the new population
        if stats is not None:
            stats.update(population)
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if verbose:
            logger.logGeneration(evals=len(invalid_ind), gen=gen, stats=stats)
        
        # write checkpoint file
        cp = dict(population=population, generation=gen, halloffame=halloffame, logbook=logbook, rndstate=random.getstate()) # Fill the dictionary using the dict(key=value[, ...]) constructor
        pickle.dump(cp, open("scoringfunction_selection_checkpoint_gen"+str(gen)+".pkl", "wb"))
        
        #for sfset in halloffame.__iter__():
        #    print(sfset)
        #print("DEBUG: halloffame=",halloffame
        R_list = []
        for ind in halloffame: # foreach individual in the Pareto front
            R = 0
            for fitness,weight in zip(ind.fitness.values, WEIGHTS):
                R += weight*(fitness)**2    # assemble the equation of the hyper-ellipsoid that gives the weighted score of each individual
            #print("DEBUG: reading individual with fitness=",ind.fitness.values," and R=",R
            R_list.append(R)
        best_individuals = [] # make a list of the best SOLUTIONS individuals of the Pareto front
        for high_R in sorted(R_list, reverse=True)[0:(SOLUTIONS-1)]:
            best_individuals.append(halloffame[R_list.index(high_R)])
        
        with open(logfile, 'a') as f:
            f.write("Generation "+str(gen)+"\n")
            for cse_index in range(0, len(best_individuals)):
                ind = best_individuals[cse_index] # get the next individual from the top SOLUTIONS list
                best_scoring_functions_string=""
                for index in set(ind):   # use set to keep the unique scoring function names in the individual
                    if index >= 0:
                        best_scoring_functions_string+="  "+ScoringFunctionNames_list[index]
                f.write("Suggested Scoring Function Combination No "+str(cse_index+1)+":"+best_scoring_functions_string + "\n")
                print(lib.utils.print_functions.bcolors.BOLDGREEN + "Suggested Scoring Function Combination No " + str(cse_index + 1) + ":", best_scoring_functions_string + lib.utils.print_functions.bcolors.ENDBOLD) # return the scoring functions that maximize the objective function
        
    return population


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population.
    
    First, the individuals having an invalid fitness are evaluated. Then, the
    evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by a crossover, a mutation or a
    reproduction proportionally to the probabilities *cxpb*, *mutpb* and 1 -
    (cxpb + mutpb). The offspring are then evaluated and the next generation
    population is selected from both the offspring **and** the population.
    Briefly, the operators are applied as following ::
    
        evaluate(population)
        for i in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)
    
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    if stats is not None:
        stats.update(population)
    if verbose:
        column_names = ["gen", "evals"]
        if stats is not None:
            column_names += list(stats.functions.keys())
        logger = tools.EvolutionLogger(column_names)
        logger.logHeader()
        logger.logGeneration(evals=len(population), gen=0, stats=stats)

    # Begin the generational process
    for gen in range(1, ngen+1):
        # Variate the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        if stats is not None:
            stats.update(population)
        if verbose:
            logger.logGeneration(evals=len(invalid_ind), gen=gen, stats=stats)

    return population


def varOrConvergence(population, toolbox, lambda_, cxpb, mutpb, gen, eta_stride):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.
    
    :param population: A list of individuals to variate.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.
    
    The variator goes as follow. On each of the *lambda_* iteration, it
    selects one of the three operations; crossover, mutation or reproduction.
    In the case of a crossover, two individuals are selected at random from
    the parental population :math:`P_\mathrm{p}`, those individuals are cloned
    using the :meth:`toolbox.clone` method and then mated using the
    :meth:`toolbox.mate` method. Only the first child is appended to the
    offspring population :math:`P_\mathrm{o}`, the second child is discarded.
    In the case of a mutation, one individual is selected at random from
    :math:`P_\mathrm{p}`, it is cloned and then mutated using using the
    :meth:`toolbox.mutate` method. The resulting mutant is appended to
    :math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
    selected at random from :math:`P_\mathrm{p}`, cloned and appended to
    :math:`P_\mathrm{o}`.
    
    This variation is named *Or* beceause an offspring will never result from
    both operations crossover and mutation. The sum of both probabilities
    shall be in :math:`[0, 1]`, the reproduction probability is
    1 - *cxpb* - *mutpb*.
    """
    assert (cxpb + mutpb) <= 1.0, ("The sum of the crossover and mutation "
        "probabilities must be smaller or equal to 1.0.")
    
    if eta_stride >= 1:
        myeta= 2*(gen/eta_stride)  # increase myeta at each generation
    elif eta_stride <= 0:
        myeta = -1*eta_stride    # in this case keep myeta constant in every generation. Recall that we changed the sign of ETA during input parsing to discriminate
                                 # between constant ETA (negative values) and increasing ETA (positive values)
    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind, eta=myeta)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))
    
    return offspring

def eaMuPlusLambdaConvergence(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, eta_stride,
                   stats=None, halloffame=None, verbose=__debug__):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population.
    
    First, the individuals having an invalid fitness are evaluated. Then, the
    evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by a crossover, a mutation or a
    reproduction proportionally to the probabilities *cxpb*, *mutpb* and 1 -
    (cxpb + mutpb). The offspring are then evaluated and the next generation
    population is selected from both the offspring **and** the population.
    Briefly, the operators are applied as following ::
    
        evaluate(population)
        for i in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)
    
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    if stats is not None:
        stats.update(population)
    if verbose:
        column_names = ["gen", "evals"]
        if stats is not None:
            column_names += list(stats.functions.keys())
        logger = tools.EvolutionLogger(column_names)
        logger.logHeader()
        logger.logGeneration(evals=len(population), gen=0, stats=stats)

    # Begin the generational process
    for gen in range(1, ngen+1):
        # Variate the population
        offspring = varOrConvergence(population, toolbox, lambda_, cxpb, mutpb, gen, eta_stride)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        if stats is not None:
            stats.update(population)
        if verbose:
            logger.logGeneration(evals=len(invalid_ind), gen=gen, stats=stats)

    return population

def eaMuPlusLambdaCrowding(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, eta,
                   stats=None, halloffame=None, verbose=__debug__):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population.
    
    First, the individuals having an invalid fitness are evaluated. Then, the
    evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by a crossover, a mutation or a
    reproduction proportionally to the probabilities *cxpb*, *mutpb* and 1 -
    (cxpb + mutpb). The offspring are then evaluated and the next generation
    population is selected from both the offspring **and** the population.
    Briefly, the operators are applied as following ::
    
        evaluate(population)
        for i in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)
    
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    if stats is not None:
        stats.update(population)
    if verbose:
        column_names = ["gen", "evals"]
        if stats is not None:
            column_names += list(stats.functions.keys())
        logger = tools.EvolutionLogger(column_names)
        logger.logHeader()
        logger.logGeneration(evals=len(population), gen=0, stats=stats)

    # Begin the generational process
    for gen in range(1, ngen+1):
        # Variate the population
        offspring = varOrConvergence(population, toolbox, lambda_, cxpb, mutpb, gen, eta)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        if stats is not None:
            stats.update(population)
        if verbose:
            logger.logGeneration(evals=len(invalid_ind), gen=gen, stats=stats)

    return population

    
def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                    stats=None, halloffame=None, verbose=__debug__):
    """This is the :math:`(\mu~,~\lambda)` evolutionary algorithm.
    
    :param population: A list of individuals.    
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population.
    
    First, the individuals having an invalid fitness are evaluated. Then, the
    evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by a crossover, a mutation or a
    reproduction proportionally to the probabilities *cxpb*, *mutpb* and 1 -
    (cxpb + mutpb). The offspring are then evaluated and the next generation
    population is selected **only** from the offspring. Briefly, the operators
    are applied as following ::
    
        evaluate(population)
        for i in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(offspring, mu)
    
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    if stats is not None:
        stats.update(population)
    if verbose:
        column_names = ["gen", "evals"]
        if stats is not None:
            column_names += list(stats.functions.keys())
        logger = tools.EvolutionLogger(column_names)
        logger.logHeader()
        logger.logGeneration(evals=len(population), gen=0, stats=stats)

    # Begin the generational process
    for gen in range(1, ngen+1):
        # Variate the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(offspring, mu)

        # Update the statistics with the new population
        if stats is not None:
            stats.update(population)
        if verbose:
            logger.logGeneration(evals=len(invalid_ind), gen=gen, stats=stats)

    return population

def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None, 
                     verbose=__debug__):
    """This is algorithm implements the ask-tell model proposed in 
    [Colette2010]_, where ask is called `generate` and tell is called `update`.
    
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.

    :returns: The final population.
    
    The toolbox should contain a reference to the generate and the update method 
    of the chosen strategy.

    .. [Colette2010] Collette, Y., N. Hansen, G. Pujol, D. Salazar Aponte and
       R. Le Riche (2010). On Object-Oriented Programming of Optimizers -
       Examples in Scilab. In P. Breitkopf and R. F. Coelho, eds.:
       Multidisciplinary Design Optimization in Computational Mechanics,
       Wiley, pp. 527-565;

    """
    if verbose:
        column_names = ["gen", "evals"]
        if stats is not None:
            column_names += list(stats.functions.keys())
        logger = tools.EvolutionLogger(column_names)
        logger.logHeader()
    
    for gen in range(ngen):
        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        if halloffame is not None:
            halloffame.update(population)
        
        # Update the strategy with the evaluated individuals
        toolbox.update(population)
        
        if stats is not None:
            stats.update(population)
        
        if verbose:
            logger.logGeneration(evals=len(population), gen=gen, stats=stats)

    return population

