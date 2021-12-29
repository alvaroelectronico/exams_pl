from simplex import Simplex
from fractions import Fraction
import numpy as np


def get_all_info(c, b, A, B):
    B = B.astype(float)
    b = b.astype(float)
    c = c.astype(float)
    A = A.astype(float)

    pos_var_basicas = [[j for j in range(A.shape[1]) if not False in (A[:, j] == B[:, i])][0] for i in
                       range(B.shape[1])]
    cB = c[pos_var_basicas]
    invB = np.linalg.inv(B)
    piB = np.dot(cB.T, invB)
    pB = np.dot(pi, A)
    uB = np.dot(invB, b)
    xB = [0 for i in range(A.shape[1])]
    for i, val in enumerate(uB):
        xB[pos_var_basicas[i]] = val[0]

    return cB, uB, pB, piB, xB, invB



objective = ('maximize', '2x_1 + 1x_2')
constraints = ['3x_1 + 1x_2  <= 30',
               '4x_1 + 3x_2 <= 16',
               '1x_1 + 2x_2 >= 4',
               '1x_1 + 3x_2 >= 5']
model = Simplex(num_vars=2, constraints=constraints, objective_function=objective)

cB, uB, pB, piB, xB, invB = get_all_info(model.c, model.b, model.A, model.all_bases[0])