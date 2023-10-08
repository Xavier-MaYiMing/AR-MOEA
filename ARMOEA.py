#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/22 10:18
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : ARMOEA.py
# @Statement : An indicator-based multiobjective evolutionary algorithm with reference point adaptation
# @Reference : Tian Y, Cheng R, Zhang X, et al. An indicator-based multiobjective evolutionary algorithm with reference point adaptation for better versatility[J]. IEEE Transactions on Evolutionary Computation, 2017, 22(4): 609-622.
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import cdist


def cal_obj(pop, nobj=3):
    # DTLZ5
    g = np.sum((pop[:, nobj - 1:] - 0.5) ** 2, axis=1)
    temp = np.tile(g.reshape((g.shape[0], 1)), (1, nobj - 2))
    pop[:, 1: nobj - 1] = (1 + 2 * temp * pop[:, 1: nobj - 1]) / (2 + 2 * temp)
    temp1 = np.concatenate((np.ones((g.shape[0], 1)), np.cos(pop[:, : nobj - 1] * np.pi / 2)), axis=1)
    temp2 = np.concatenate((np.ones((g.shape[0], 1)), np.sin(pop[:, np.arange(nobj - 2, -1, -1)] * np.pi / 2)), axis=1)
    return np.tile((1 + g).reshape(g.shape[0], 1), (1, nobj)) * np.fliplr(np.cumprod(temp1, axis=1)) * temp2


def factorial(n):
    # calculate n!
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def combination(n, m):
    # choose m elements from an n-length set
    if m == 0 or m == n:
        return 1
    elif m > n:
        return 0
    else:
        return factorial(n) // (factorial(m) * factorial(n - m))


def reference_points(npop, nvar):
    # calculate approximately npop uniformly distributed reference points on nvar dimensions
    h1 = 0
    while combination(h1 + nvar, nvar - 1) <= npop:
        h1 += 1
    points = np.array(list(combinations(np.arange(1, h1 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
    points = (np.concatenate((points, np.zeros((points.shape[0], 1)) + h1), axis=1) - np.concatenate((np.zeros((points.shape[0], 1)), points), axis=1)) / h1
    if h1 < nvar:
        h2 = 0
        while combination(h1 + nvar - 1, nvar - 1) + combination(h2 + nvar, nvar - 1) <= npop:
            h2 += 1
        if h2 > 0:
            temp_points = np.array(list(combinations(np.arange(1, h2 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
            temp_points = (np.concatenate((temp_points, np.zeros((temp_points.shape[0], 1)) + h2), axis=1) - np.concatenate((np.zeros((temp_points.shape[0], 1)), temp_points), axis=1)) / h2
            temp_points = temp_points / 2 + 1 / (2 * nvar)
            points = np.concatenate((points, temp_points), axis=0)
    return points


def cal_dis(objs, refs):
    # calculate the distance between each solution to each adjusted reference point
    npop = objs.shape[0]
    (nref, nobj) = refs.shape
    objs = np.where(objs > 1e-6, objs, 1e-6)
    refs = np.where(refs > 1e-6, refs, 1e-6)

    # adjust the location of each reference point
    cosine = 1 - cdist(objs, refs, 'cosine')
    normP = np.sqrt(np.sum(objs ** 2, axis=1))
    normR = np.sqrt(np.sum(refs ** 2, axis=1))
    d1 = np.tile(normP.reshape((npop, 1)), (1, nref)) * cosine
    d2 = np.tile(normP.reshape((npop, 1)), (1, nref)) * np.sqrt(1 - cosine ** 2)
    nearest = np.argmin(d2, axis=0)
    refs = refs * np.tile((d1[nearest, np.arange(nref)] / normR).reshape((nref, 1)), (1, nobj))
    return cdist(objs, refs)


def nd_sort(objs):
    # fast non-domination sort
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    rank = np.zeros(npop, dtype=int)
    ind = 1
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    return rank


def update_ref(arch, W, Range):
    # update reference points
    # Step 1. Delete duplicated and dominated solutions
    (nref, nobj) = W.shape
    ind = nd_sort(arch) == 1
    arch = arch[ind]
    arch = np.unique(arch, axis=0)

    # Step 2. Update the ideal point
    if np.any(Range):
        Range[0] = np.min((Range[0], np.min(arch, axis=0)), axis=0)
    elif np.any(arch):
        Range = np.zeros((2, nobj))
        Range[0] = np.min(arch, axis=0)
        Range[1] = np.max(arch, axis=0)

    # Step 3. Update archive and reference points
    if arch.shape[0] <= 1:
        refs = W
    else:
        # Step 3.1. Find contributing solutions and valid weight vectors
        tarch = arch - Range[0]
        W *= (Range[1] - Range[0])
        dis = cal_dis(tarch, W)
        nearest1 = np.argmin(dis, axis=0)
        con_sols = np.unique(nearest1)  # contributing solutions
        nearest2 = np.argmin(dis, axis=1)
        valid_W = np.unique(nearest2[con_sols])  # valid reference points

        # Step 3.2. Update archive
        choose = np.full(tarch.shape[0], False)
        choose[con_sols] = True
        cosine = 1 - cdist(tarch, tarch, 'cosine')
        np.fill_diagonal(cosine, 0)
        while np.sum(choose) < min(3 * nref, tarch.shape[0]):
            unselected = np.where(~choose)[0]
            best = np.argmin(np.max(cosine[~choose][:, choose], axis=1))
            choose[unselected[best]] = True
        arch = arch[choose]
        tarch = tarch[choose]

        # Step 3.3. Update reference points
        refs = np.concatenate((W[valid_W], tarch), axis=0)
        choose = np.concatenate((np.full(valid_W.shape[0], True), np.full(tarch.shape[0], False)))
        cosine = 1 - cdist(refs, refs, 'cosine')
        np.fill_diagonal(cosine, 0)
        while np.sum(choose) < min(nref, refs.shape[0]):
            selected = np.where(~choose)[0]
            best = np.argmin(np.max(cosine[~choose][:, choose], axis=1))
            choose[selected[best]] = True
        refs = refs[choose]
    return arch, refs, Range


def mating_selection(pop, objs, refs, Range):
    # mating selection
    (npop, nvar) = pop.shape
    dis = cal_dis(objs - Range[0], refs)
    convergence = np.min(dis, axis=1)
    rank = np.argsort(dis, axis=0)
    dis = np.sort(dis, axis=0)

    # Step 1. Calculate the fitness of noncontributing solutions
    noncontributing = np.full(npop, True)
    noncontributing[rank[0]] = False
    metric = np.sum(dis[0]) + np.sum(convergence[noncontributing])
    fitness = np.full(npop, np.inf)
    fitness[noncontributing] = metric - convergence[noncontributing]

    # Step 2. Calculate the fitness of contributing solutions
    for p in np.where(~noncontributing)[0]:
        temp = rank[0] == p
        temp_noncontributing = np.full(npop, False)
        temp_noncontributing[rank[1, temp]] = True
        temp_noncontributing = np.logical_and(noncontributing, temp_noncontributing)
        fitness[p] = metric - np.sum(dis[0, temp]) + np.sum(dis[1, temp]) - np.sum(convergence[temp_noncontributing])

    # Step 3. Binary tournament selection
    nm = npop if npop % 2 == 0 else npop + 1  # mating pool size
    mating_pool = np.zeros((nm, nvar))
    for i in range(nm):
        [ind1, ind2] = np.random.choice(npop, 2, replace=False)
        if fitness[ind1] > fitness[ind2]:
            mating_pool[i] = pop[ind1]
        else:
            mating_pool[i] = pop[ind2]
    return mating_pool


def last_selection(objs, refs, Range, K):
    # select K solutions from the last front
    npop = objs.shape[0]
    nref = refs.shape[0]
    Distance = cal_dis(objs - Range[0], refs)
    Convergence = np.min(Distance, axis=1)
    dis = np.sort(Distance, axis=0)
    rank = np.argsort(Distance, axis=0)
    Remain = np.full(npop, True)
    while np.sum(Remain) > K:
        Noncontributing = Remain.copy()
        Noncontributing[rank[0]] = False
        METRIC = np.sum(dis[0]) + np.sum(Convergence[Noncontributing])
        Metric = np.full(npop, np.inf)
        Metric[Noncontributing] = METRIC - Convergence[Noncontributing]
        for p in np.where(np.logical_and(Remain, ~Noncontributing))[0]:
            temp = rank[0] == p
            noncontributing = np.full(npop, False)
            noncontributing[rank[1, temp]] = True
            noncontributing = np.logical_and(noncontributing, Noncontributing)
            Metric[p] = METRIC - np.sum(dis[0, temp]) + np.sum(dis[1, temp]) - np.sum(Convergence[noncontributing])
        Del = np.argmin(Metric)
        temp = rank != Del
        Remain[Del] = False
        dis = dis.T[temp.T].reshape((np.sum(Remain), nref), order='F')
        rank = rank.T[temp.T].reshape((np.sum(Remain), nref), order='F')
    return Remain


def environmental_selection(pop, objs, refs, Range, num):
    # environmental selection
    rank = nd_sort(objs)
    selected = np.full(pop.shape[0], False)
    ind = 1
    while np.sum(selected) + np.sum(rank == ind) <= num:
        selected[rank == ind] = True
        ind += 1
    if num != np.sum(selected):
        last = np.where(rank == ind)[0]
        remain = last_selection(objs[last], refs, Range, num - np.sum(selected))
        selected[last[remain]] = True
    Range[1] = np.max(objs[selected], axis=0)
    Range[1, Range[1] - Range[0] < 1e-6] = 1
    return pop[selected], objs[selected], Range


def crossover(mating_pool, lb, ub, eta_c):
    # simulated binary crossover (SBX)
    (noff, nvar) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, nvar))
    mu = np.random.random((nm, nvar))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, (nm, nvar))
    beta[np.random.random((nm, nvar)) < 0.5] = 1
    beta[np.tile(np.random.random((nm, 1)) > 1, (1, nvar))] = 1
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, eta_m):
    # polynomial mutation
    (npop, nvar) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, nvar)) < 1 / nvar
    mu = np.random.random((npop, nvar))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def main(npop, iter, lb, ub, nobj=3, eta_c=20, eta_m=20):
    """
    The main loop
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nobj: the dimension of the objective space (default = 3)
    :param eta_c: spread factor distribution index (default = 20)
    :param eta_m: perturbance factor distribution index (default = 20)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = cal_obj(pop, nobj)  # objectives
    W = reference_points(npop, nobj)  # original reference points
    arch, refs, Range = update_ref(objs, W, [])  # archive, reference points, ideal and nadir points

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 100 == 0:
            print('Iteration: ' + str(t + 1) + ' completed.')

        # Step 2.1. Mating selection + crossover + mutation
        mating_pool = mating_selection(pop, objs, refs, Range)
        off = crossover(mating_pool, lb, ub, eta_c)
        off = mutation(off, lb, ub, eta_m)
        off_objs = cal_obj(off, nobj)

        # Step 2.2. Update reference points
        arch, refs, Range = update_ref(np.concatenate((arch, off_objs), axis=0), W, Range)

        # Step 2.3. Environmental selection
        pop, objs, Range = environmental_selection(np.concatenate((pop, off), axis=0), np.concatenate((objs, off_objs), axis=0), refs, Range, npop)

    # Step 3. Sort the results
    pf = objs
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.view_init(45, 45)
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    z = [o[2] for o in pf]
    ax.scatter(x, y, z, color='red')
    ax.set_xlabel('objective 1')
    ax.set_ylabel('objective 2')
    ax.set_zlabel('objective 3')
    plt.title('The Pareto front of DTLZ5')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(100, 300, np.array([0] * 12), np.array([1] * 12))
