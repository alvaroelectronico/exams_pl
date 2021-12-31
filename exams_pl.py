from simplex import Simplex
from fractions import Fraction
from collections import namedtuple
import numpy as np

FORM_VB = "V^B = c-c^BB^{-1}A"
FORM_piB = "\pi^B = c^BB^{-1}"
FORM_pB = "p^B = B^{-1}A"

BasicSolution = namedtuple(
    "basic_solution", ["B", "cB", "uB", "pB", "piB", "xB", "invB", "VB", "basic_vars_id"]
)


def gen_all_matrices(c, b, A, B):
    B = B.astype(float)
    b = b.astype(float)
    c = c.astype(float)
    A = A.astype(float)

    basic_vars_id = [
        [j for j in range(A.shape[1]) if not False in (A[:, j] == B[:, i])][0]
        for i in range(B.shape[1])
    ]
    cB = c[basic_vars_id]
    invB = np.linalg.inv(B)
    piB = np.dot(cB.T, invB)
    pB = np.dot(invB, A)
    uB = np.dot(invB, b)
    xB = [0 for i in range(A.shape[1])]
    for i, val in enumerate(uB):
        xB[basic_vars_id[i]] = val[0]
    VB = c.T - np.dot(cB.T, pB)
    cB, uB, pB, piB, xB, invB, VB = map(
        array_to_fraction, [cB, uB, pB, piB, xB, invB, VB]
    )

    return B, cB, uB, pB, piB, xB, invB, VB, basic_vars_id


def array_to_fraction(arr):
    to_fraction = lambda t: Fraction(t).limit_denominator()
    vfunc = np.vectorize(to_fraction)
    return vfunc(arr)


def fraction_to_tex(frac):
    if frac.denominator == 1 or frac.denominator == 0:
        str = "{}".format(frac.numerator)
    else:
        str = "\\frac{{{}}}{{{}}}".format(frac.numerator, frac.denominator)
    return str


def matrix_to_tex(m, brackets="round"):
    if brackets == "round":
        str = "\\begin{pmatrix}\n"
    for r in range(m.shape[0]):
        for c in range(m.shape[1] - 1):
            str += "{} & ".format(fraction_to_tex(m[r, c]))
            # print(m[r, c])
        str += format(fraction_to_tex(m[r, m.shape[1] - 1]))
        str += "\\\\\n"
    if brackets == "round":
        str += "\\end{pmatrix}\n"
    return str


def operatons_to_tex(matrices):
    str = ""
    for i, m in enumerate(matrices):
        str += matrix_to_tex(m[0])
        if len(m) > 1:
            str += m[1]
    return str


objective = ("max", "80x_1 + 45x_2 + 95x_3")
constraints = [
    "2x_1 + 1x_2 + 2.5x_3 <= 100",
    "1x_1 + 0.5x_2 + 1x_3 >= 40",
    "1x_1 + 0x_2 + 1x_3 >= 10",
]
model = Simplex(num_vars=3, constraints=constraints, objective_function=objective)

all_basic_solutions = list()
for B in model.all_bases:
    B, cB, uB, pB, piB, xB, invB, VB, basic_vars_id = gen_all_matrices(model.c, model.b, model.A, B)
    all_basic_solutions.append(BasicSolution(B, cB, uB, pB, piB, xB, invB, VB, basic_vars_id))


# print(matrix_to_tex(invB))
print(model.model_tex)
print("")
print(model.compose_tableaux(float("inf")))
print(FORM_piB)

B, cB, uB, pB, piB, xB, invB, VB, basic_vars_id = all_basic_solutions[-1]
c, A, b = model.c, model.A, model.b

VB_tex = FORM_VB + " = "
VB_tex += operatons_to_tex([[c.T, "-"], [cB.T], [invB], [A, " = "], [VB]])
print(VB_tex)

piB_tex = FORM_piB + " = "
piB_tex += operatons_to_tex([[cB.T], [invB, " = "], [piB]])
print("")
print(piB_tex)

pB_tex = FORM_pB + " = "
pB_tex += operatons_to_tex([[invB],[A, " = "],[pB]])
print("")
print(pB_tex)