# ____________________________________________________________________________________
# Multiple strategy enhanced hybrid algorithm BAGWO combining beetle antennae search and grey wolf optimizer for global optimization
# (BAGWO)source codes version 1.2
#
# Update Date: 2025/05/07
#
# Developed in MATLAB R2022b
#
# Author and programmer: Fan Zhang
# Main Contributors: Fan Zhang, Chuankai Liu, Peng Liu, Shuiting Ding
#
# E-Mail: fan.zhang@buaa.edu.cn, liuchuankai@buaa.edu.cn, auroraus2020@outlook.com
# Project homepage: https://github.com/auroraua/BAGWO
# Paper: https://doi.org/10.1038/s41598-025-98816-0
# ____________________________________________________________________________________

import numpy as np
from scipy.stats import qmc

# BAGWO
"""
Input Parameter
  N: Number of search agents in the swarm
  Max_iteration: Maximum number of iteration times
  lb: Lower limit vector of decision variable
  ub: Upper limit vector of decision variable
  dim: Number of decision variables
  fobj: The objective function corresponding to the optimization problem
"""
def BAGWO(N, Max_iteration, lb, ub, fobj, dim=0):
    """串行版本算法，适合简单的优化问题"""
    # Decision variable vector upper and lower limit processing
    if isinstance(ub, (int, float)):
        ub = np.full(dim, ub)
        lb = np.full(dim, lb)
    else:
        dim = len(lb)
        ub = np.array(ub)
        lb = np.array(lb)

    convergenceCurve = []
    xbd = ub - lb  # The difference vector between upper limit and lower limit of decision variables

    # BAGWO Initial parameter setting
    c = 1.0  # Initial antennae length of beetles
    localStep = 10  # Number of local exploitation steps
    finalCharisma = 0.99  # The final charisma

    charisma = 0  # Initial charisma

    # Latin hypercube sampling of initial decision variables of population
    sampler = qmc.LatinHypercube(d=dim)
    X = sampler.random(n=N) * xbd + lb

    tempX = np.ones((N, dim))  # Used to update and record the search agent's individual best case in local exploitation
    Fitness = np.full(N, 1e50)  # Initialize Population fitness
    bestX = X[0, :]  # The decision variables vector corresponding to the historical best search agent
    bestFitness = fobj(bestX)  # The fitness corresponding to the historical best search agent

    a = 1 / Max_iteration ** 1  # anterior antennae length coefficient, Equation(13)
    Ns = int(np.ceil(Max_iteration * ((0.5) ** (
                0.6342 * Max_iteration ** 0.1775))))  # The number of iterations corresponding to the change in the decay rate of the beetles antennae length, Equation(16)
    decayRate = (a / c) ** (1 / Ns)  # Decay rate of beetles length of antennae, Equation(11)

    # Iterative solution of optimization problem
    for iterNum in range(Max_iteration):
        for k in range(N):
            # Local exploitation process for each search agent
            for m in range(int(np.ceil(localStep * np.cos(np.pi / 2 * iterNum / Max_iteration)))):  # Equation(16)
                dir = np.random.uniform(-1, 1, dim)  # The orientation of the beetle's head is random
                dir = dir / np.linalg.norm(dir)  # Normalized, Equation(17)
                arm = dir * c * xbd  # Antenna detection distance
                xr = X[k, :] + arm  # Update the position of the right antennae, Equation(18)
                xr = np.maximum(xr, lb)
                xr = np.minimum(xr,
                                ub)  # Ensure that the position of the right antennae corresponding to the right antennae are within the range of the upper limit and lower limit
                xl = X[k, :] - arm
                xl = np.maximum(xl, lb)
                xl = np.minimum(xl, ub)
                fr = fobj(xr)  # Get the corresponding fitness at the right antennae
                fl = fobj(xl)
                minv = min(fr, fl)

                # Update search agent individual best record
                if minv < Fitness[k]:
                    Fitness[k] = minv
                    # Update and record the current search agent's best search in local exploitation
                    if fr < fl:
                        tempX[k, :] = xr
                    else:
                        tempX[k, :] = xl
                    X[k, :] = X[k, :] - 2.0 * arm * np.sign(
                        fr - fl)  # Update the current search agent's location, Equation(20)
                    X[k, :] = np.maximum(X[k, :], lb)
                    X[k, :] = np.minimum(X[k, :], ub)
                else:
                    X[k, :] = X[k, :] - 0.5 * arm * np.sign(
                        fr - fl)  # Update the current search agent's location, Equation(21)
                    X[k, :] = np.maximum(X[k, :], lb)
                    X[k, :] = np.minimum(X[k, :], ub)

        # Update historical best search agent, Equation(6) and (7)
        indexOfSort = np.argsort(Fitness)  # Record the index of search agents in the swarm, sorted by fitness
        if iterNum == 0:
            bestFitness = Fitness[indexOfSort[0]]
            bestX = tempX[indexOfSort[0], :]
        else:
            if Fitness[indexOfSort[0]] < bestFitness:
                bestFitness = Fitness[indexOfSort[0]]
                bestX = tempX[indexOfSort[0], :]

        convergenceCurve.append(bestFitness)

        # Summon search agents in the population
        for pop in range(N):
            X[pop, :] = X[pop, :] + charisma * (bestX - X[pop, :])  # Equation(22)

        #
        if iterNum == Ns:
            b = 10 ** (-1 * (0.7928 * Max_iteration ** 0.5031))  # Calculate hind antennae length coefficient b by Equation (14)
            decayRate = (b / c) ** (1 / (Max_iteration - Ns))  # Update antennae length decay rate, contained within Equation (11).

        charisma = 1 / (1 + 100 * ((1 - finalCharisma) / 100) ** (
                    iterNum / Max_iteration))  # Update the charisma, Equation(8)
        c = c * decayRate  # Update antennae length by Equation (11)

    return bestFitness, bestX, np.array(convergenceCurve)
