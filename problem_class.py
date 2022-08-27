from simplex import Simplex
from fractions import Fraction
from collections import namedtuple
import numpy as np
import six

FORM_VB = "V^B = c-c^BB^{-1}A"
FORM_piB = "\pi^B = c^BB^{-1}"
FORM_pB = "p^B = B^{-1}A"

BasicSolution = namedtuple(
    "basic_solution", ["B", "cB", "uB", "pB", "piB", "xB", "z", "invB", "VB", "cB_f1", "p1B_f1", "vB_f1", "z_f1"]
)


class LP_Problem():

    def __init__(self, num_vars, constraints, objective_function):

        self.model = Simplex(num_vars, constraints, objective_function)
        self.c1 = self.model.c1.astype('float')
        self.c2 = self.model.c2.astype('float')
        self.b = self.model.b.astype('float')
        self.A = self.model.A.astype('float')
        self.num_vars = self.model.num_vars
        self.num_s_vars = self.model.num_s_vars
        self.num_r_vars = self.model.num_r_vars
        self.bases_info = dict()  # a dictionary with a BasicSolution named tuple for every element in bases
        self.model.solve_model()
        self.bases = self.model.all_bases
        self.gen_all_bases_info()

    def gen_base_info(self, basic_vars_id):
        base = self.A[:, basic_vars_id]
        c2_b = self.c2[basic_vars_id]
        base_inv = np.linalg.inv(base)
        pi2_b = np.dot(c2_b.T, base_inv)
        p_b = np.dot(base_inv, self.A)
        u_b = np.dot(base_inv, self.b)
        x_b = [0 for i in range(self.A.shape[1])]
        for i, val in enumerate(u_b):
            x_b[basic_vars_id[i]] = val[0]
        v2_b = self.c2.T - np.dot(c2_b.T, p_b)

        c1_b = self.c1[basic_vars_id]
        pi1_b = np.dot(c1_b.T, base_inv)
        v1_b = self.c1.T - np.dot(c1_b.T, p_b)

        z1 = np.dot(self.c1.T, x_b)
        z2 = np.dot(self.c2.T, x_b)

        self.bases_info[tuple(basic_vars_id)] = BasicSolution(base, c2_b, u_b, p_b, pi2_b, x_b, z2, base_inv, \
                                                              v2_b, c1_b, pi1_b, v1_b, z1)

        self.bases_info[tuple(basic_vars_id)] = basic_solution_to_fraction(self.bases_info[tuple(basic_vars_id)])

    def gen_all_bases_info(self):
        for basic_vars_id in self.bases:
            self.gen_base_info(basic_vars_id)

    def tableau_tex(self, basic_vars_id):
        base_info = self.bases_info[tuple(basic_vars_id)]
        str = ""
        # Line 0. Phase 1
        str += "Phase 1 & {}".format(-base_info.z_f1)
        return str

        # for i, row in enumerate(self.coeff_matrix):
        #     if i == 1:
        #         str += "\hline\n"
        #     if i != 0:
        #         str += "${}$".format(self.var_names[self.basic_vars[i]])
        #     if i == 0:
        #         str += " & {}".format(-row[len(row) - 1])
        #     else:
        #         str += " & {}".format(row[len(row) - 1])
        #     for j in range(0, len(row) - 1):
        #         if i == 0:
        #             str += " & {}".format(-row[j])
        #         else:
        #             str += " & {}".format(row[j])
        #     str += "\\\\ \n"
        # str += "\hline\n"
        # return str.3.


def array_to_fraction(arr):
    to_fraction = lambda t: Fraction(t).limit_denominator()
    vfunc = np.vectorize(to_fraction)
    return vfunc(arr)

def arrays_to_fraction(list_arr):
    list_arr_frac = list()
    for arr in list_arr:
        list_arr_frac.append(array_to_fraction(arr))
    return list_arr_frac

def basic_solution_to_fraction(basic_sol):
    basic_sol_frac = BasicSolution(arrays_to_fraction(basic_sol)[0],
                                   arrays_to_fraction(basic_sol)[1],
                                   arrays_to_fraction(basic_sol)[2],
                                   arrays_to_fraction(basic_sol)[3],
                                   arrays_to_fraction(basic_sol)[4],
                                   arrays_to_fraction(basic_sol)[5],
                                   arrays_to_fraction(basic_sol)[6],
                                   arrays_to_fraction(basic_sol)[7],
                                   arrays_to_fraction(basic_sol)[8],
                                   arrays_to_fraction(basic_sol)[9],
                                   arrays_to_fraction(basic_sol)[10],
                                   arrays_to_fraction(basic_sol)[11],
                                   arrays_to_fraction(basic_sol)[12])
    return basic_sol_frac


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
        str += format(fraction_to_tex(m[r, m.shape[1] - 1]))
        str += "\\\\\n"
    if brackets == "round":
        str += "\\end{pmatrix}\n"
    return str


def operations_to_tex(matrices):
    str = ""
    for i, m in enumerate(matrices):
        if isinstance(m[0], six.string_types):
            str += m[0]
        else:
            str += matrix_to_tex(m[0])
        if len(m) > 1:
            str += m[1]
    return str


def create_equation(content_str):
    str = "\\begin{equation}\n\\begin{split}\n"
    str += content_str
    str += "\n\\end{split}\n\\end{equation}"
    return str


# objective = ("max", "80x_1 + 45x_2 + 95x_3")
# constraints = [
#     "2x_1 + 1x_2 + 2.5x_3 <= 100",
#     "1x_1 + 0.5x_2 + 1x_3 = 30",
#     "1x_1 + 0x_2 + 1x_3 >= 8"
# ]
# modelA = Simplex(num_vars=3, constraints=constraints, objective_function=objective)
# problemA = LP_Problem(modelA.c1, modelA.c2, modelA.b, modelA.A, modelA.num_vars, modelA.num_s_vars, modelA.num_r_vars)
# problemA.bases = modelA.all_bases
# problemA.gen_all_bases_info()
# print(modelA.compose_tableaux())
# solutionA = problemA.bases_info[tuple(problemA.bases[len(problemA.bases) - 1])]
#
# objectiveB = ("max", "70x_1 + 80x_2 + 40x_3")
# constraintsB = [
#     "4x_1 + 8x_2 + 6x_3 <= 400",
#     "1x_1 + 1x_2 + 0.5x_3 = 90",
#     "1x_1 + 1x_2 + 1x_3 >= 20",
# ]
# modelB = Simplex(num_vars=3, constraints=constraintsB, objective_function=objectiveB)
# problemB = LP_Problem(modelB.c1, modelB.c2, modelB.b, modelB.A, modelB.num_vars,
#                       modelB.num_s_vars, modelB.num_r_vars)
# problemB.bases = modelB.all_bases
# problemB.gen_all_bases_info()
# print(modelB.compose_tableaux())
# solutionB = problemB.bases_info[tuple(problemB.bases[len(problemB.bases) - 1])]
#
# problems = [problemA, problemB]
# solutions = [solutionA, solutionB]
# models = [modelA, modelB]
#
# no_model = 0
#
# solution_txt = solutions[no_model]
# problem_txt = problems[no_model]
# model_txt = models[no_model]
#
#
# # Base of optima solution
# base = solution_txt.B
# base_inv = solution_txt.invB
#
# # Tex for B and B-1
# text = "B = "
# text += matrix_to_tex(array_to_fraction(base))
# text += "\,\, B^{-1} = "
# text += matrix_to_tex(array_to_fraction(base_inv))
# print(text)
#
# # Shadow prices
# piB_tex = FORM_piB + " = "
# piB_tex += operations_to_tex([[array_to_fraction(solution_txt.cB.T)], [array_to_fraction(base_inv), " = "],
#                               [array_to_fraction(solution_txt.piB)]])
# print("")
# print(piB_tex)
#
# # V^B_3
#
#
# # New variable
#
# c_4 = [np.array(85).reshape(-1, 1),
#        np.array(80).reshape(-1, 1)]
# a_4 = [np.array([1, 1, 0]).reshape(-1, 1),
#         np.array([2, 1, 0]).reshape(-1, 1)]
#
# no_model = 1
# solution_txt = solutions[no_model]
# problem_txt = problems[no_model]
# model_txt = models[no_model]
# c_4_txt = c_4[no_model]
# a_4_txt = a_4[no_model]
#
# v_b_4_txt = c_4_txt - np.dot(np.dot(solution_txt.cB.T, solution_txt.invB), a_4_txt)
# c_4f = array_to_fraction(c_4_txt)
# a_4f = array_to_fraction(a_4_txt)
# v_b_4f = array_to_fraction(v_b_4_txt)
# c_bf = array_to_fraction(solution_txt.cB)
# base_invf = array_to_fraction(solution_txt.invB)
#
# # A
# text = "A = "
# text += matrix_to_tex(a_4f)
# print(text)
#
# # VB
# text = "V^B_4 = c_4-c^BB^{-1}A_4 = "
# text += operations_to_tex([[c_4f, " - "], [c_bf.T], [base_invf], [a_4f]])
# text += " = "
# text += fraction_to_tex(v_b_4f[0, 0])
# print(text)
#
#
#
#
#
#
#
#
#
#
#
#
#
# """
# Code to be part of a class
# """
#
# base, c2_b, u_b, p_b, pi2_b, x_b, base_inv, v2_b, c1_b, pi1_b, v1_b = all_basic_solutions[-1]
# c2, c1, A, b = modelA.c2, modelA.c1, modelA.A, modelA.b
#
# VB_tex = FORM_VB + " = "
# VB_tex += operations_to_tex([[c2.T, "-"], [c2_b.T], [base_inv], [A, " = "], [v2_b]])
# print(VB_tex)
#
# piB_tex = FORM_piB + " = "
# piB_tex += operations_to_tex([[problemA.c2.T], [base_inv, " = "], [pi2_b]])
# print("")
# print(piB_tex)
#
# pB_tex = FORM_pB + " = "
# pB_tex += operations_to_tex([[base_inv], [A, " = "], [p_b]])
# print("")
# print(pB_tex)
#
# """
# Specific code (sandbox)
# """
# b2 = np.copy(b)
# b2[2][0] = Fraction(11).limit_denominator()
# b
# b2
#
# np.dot(base_inv, b2)
#
# ub2 = array_to_fraction(np.dot(base_inv, b2))
#
# str = operations_to_tex([["u^B = "], [base_inv], [b2, "="], [ub2]])
# str = create_equation(str)
# print(str)
#
# pB3 = p_b[:, 2].reshape(-1, 1)
# str = operations_to_tex([["p^B_3 = "], [pB3]])
# str = create_equation(str)
# print(str)
#
# objective3 = ("max", "80x_1 + 45x_2 + 95x_3")
# constraints3 = [
#     "2x_1 + 1x_2 + 2.5x_3 <= 100",
#     "1x_1 + 0.5x_2 + 1x_3 >= 40",
#     "1x_1 + 0x_2 + 1x_3 >= 10",
#     "1x_1 + 1x_2 + 0x_3 <= 80"
# ]
# model3 = Simplex(num_vars=3, constraints=constraints3, objective_function=objective3)
#
# model3.compose_tableaux()
#
# model3.tableaux_list
#
# print(model3.tableaux_list[4])
