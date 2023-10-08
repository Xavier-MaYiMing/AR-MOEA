### AR-MOEA: An indicator-based multiobjective evolutionary algorithm with reference point adaptation

##### Reference: Tian Y, Cheng R, Zhang X, et al. An indicator-based multiobjective evolutionary algorithm with reference point adaptation for better versatility[J]. IEEE Transactions on Evolutionary Computation, 2017, 22(4): 609-622.

| Variables | Meaning                                              |
| --------- | ---------------------------------------------------- |
| npop      | Population size                                      |
| iter      | Iteration number                                     |
| lb        | Lower bound                                          |
| ub        | Upper bound                                          |
| nobj      | The dimension of objective space (default = 3)       |
| eta_c     | Spread factor distribution index (default = 30)      |
| eta_m     | Perturbance factor distribution index (default = 20) |
| nvar      | The dimension of decision space                      |
| pop       | Population                                           |
| objs      | Objectives                                           |
| W         | Oringinal reference vectors                          |
| arch      | Archive                                              |
| refs      | Reference points                                     |
| Range     | Ideal and nadir points                               |
| pf        | Pareto front                                         |

#### Test problem: DTLZ5

$$
\begin{aligned}
	& \theta_i = \frac{\pi}{4(1 + g(x_M))}(1 + 2g(x_M)x_i), \quad i = 1, \cdots, n \\
	& g(x_M) = \sum_{x_i \in x_M} (x_i - 0.5) ^ 2 \\
	& \min \\
	& f_1(x) = (1 + g(x_M)) \cos(\theta_i \pi / 2) \cdots \cos(\theta_{M-2} \pi /2) \cos(\theta_{M - 1} \pi /2) \\
	& f_2(x) = (1 + g(x_M)) \cos(\theta_i \pi / 2) \cdots \cos(\theta_{M-2} \pi /2) \sin(\theta_{M - 1} \pi /2) \\
	& f_3(x) = (1 + g(x_M)) \cos(\theta_i \pi / 2) \cdots \sin(\theta_{M-2} \pi /2) \\
	& \vdots \\
	& f_M(x) = (1 + g(x_M)) \sin(\theta_1 \pi /2) \\
	& \text{subject to} \\
	& x_i \in [0, 1], \quad \forall i = 1, \cdots, n
\end{aligned}
$$



#### Example

```python
if __name__ == '__main__':
    main(100, 300, np.array([0] * 12), np.array([1] * 12))
```

##### Output:

![](https://github.com/Xavier-MaYiMing/AR-MOEA/blob/main/Pareto%20front.png)

