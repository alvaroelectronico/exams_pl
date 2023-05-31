from simplex import Simplex
from fractions import Fraction
from collections import namedtuple
import numpy as np
import six
from pyperclip import *


FORM_VB = "V^B = c-c^BB^{-1}A"
FORM_piB = "\pi^B = c^BB^{-1}"
FORM_pB = "p^B = B^{-1}A"

BasicSolution = namedtuple(
    "basic_solution", ["B", "cB", "uB", "pB", "piB", "xB", "z", "invB", "vB", "cB_f1", "p1B_f1", "vB_f1", "z_f1"]
)


class LP_Problem():

    def __init__(self, num_vars, constraints, objective_function):

        self.model = Simplex(num_vars, constraints, objective_function)
        self.c1 = self.model.c1.astype('float')
        self.c2 = self.model.c2.astype('float')
        self.b = self.model.b.astype('float')
        self.A = self.model.A.astype('float')
        self.var_names = self.model.var_names
        self.num_vars = self.model.num_vars
        self.num_s_vars = self.model.num_s_vars
        self.num_r_vars = self.model.num_r_vars
        self.num_total_vars = self.num_vars + self.num_s_vars + self.num_r_vars
        self.bases_info = dict()  # a dictionary with a BasicSolution named tuple for every element in bases
        self.model.solve_model()
        self.simplex_bases = self.model.all_bases
        self.simplex_first_feasible_base = self.model.first_feasible_base
        self.gen_all_bases_info()
        self.sense = self.model.objective
        self.objective_function = self.model.objective_function
        self.constraints = self.model.constraints

    def gen_base_info(self, basic_vars_id):
        B = self.A[:, basic_vars_id]
        cB = self.c2[basic_vars_id]
        invB = np.linalg.inv(B)
        piB = np.dot(cB.T, invB)
        pB = np.dot(invB, self.A)
        uB = np.dot(invB, self.b)
        xB = [0 for i in range(self.A.shape[1])]
        for i, val in enumerate(uB):
            xB[basic_vars_id[i]] = val[0]
        vB = self.c2.T - np.dot(cB.T, pB)

        cB_f1 = self.c1[basic_vars_id]
        piB_f1 = np.dot(cB_f1.T, invB)
        vB_f1 = self.c1.T - np.dot(cB_f1.T, pB)

        z_f1 = np.dot(self.c1.T, xB)
        z = np.dot(self.c2.T, xB)

        # "basic_solution", ["B", "cB", "uB", "pB", "piB", "xB", "z", "invB", "vB", "cB_f1", "p1B_f1", "vB_f1", "z_f1"]

        base_info = BasicSolution(B, cB, uB, pB, piB, xB, z, invB, vB, cB_f1, piB_f1, vB_f1, z_f1)

        base_info = basic_solution_to_fraction(base_info)
        return base_info
        # self.bases_info[tuple(basic_vars_id)] = BasicSolution(base, c2_b, u_b, p_b, pi2_b, x_b, z2, base_inv, \
        #                                                       v2_b, c1_b, pi1_b, v1_b, z1)
        #
        # self.bases_info[tuple(basic_vars_id)] = basic_solution_to_fraction(self.bases_info[tuple(basic_vars_id)])

    def gen_all_bases_info(self):
        for basic_vars_id in self.simplex_bases:
            self.bases_info[tuple(basic_vars_id)] = self.gen_base_info(basic_vars_id)

    def tableau_basic_sol_tex(self, basic_vars_id):
        base_info = self.gen_base_info(basic_vars_id)
        str = ""

        # OF and reduced costs for phase 1 if they apply (there are artifitial variables or it is the first
        # feasible base for the second phase).
        artificial_variables_exist = len([i for i in basic_vars_id if i >= self.num_vars + self.num_s_vars]) > 0
        is_fist_feasible_sol_phase2 = set(basic_vars_id) == set(self.simplex_first_feasible_base)
        if artificial_variables_exist or is_fist_feasible_sol_phase2:
            str += "Phase 1 & {} ".format(fraction_to_tex(-base_info.z_f1[0]))
            for i in base_info.vB_f1[0]:
                str += "& {} ".format(fraction_to_tex(i))
            str += "\\\\ \n"

        # OF and reduced costs for phase 2
        str += "Phase 2 & {} ".format(fraction_to_tex(-base_info.z[0]))
        for i in base_info.vB[0]:
            str += "& {} ".format(i)
        str += "\\\\ \n"

        # Horizontal line to seprate reduced cost from the rest of the table
        str += "\hline\n"

        # Rows for subtitution rates
        for i in range(0, self.A.shape[0]):
            str += "${}$".format(self.var_names[basic_vars_id[i]])
            str += " & {} ".format(base_info.uB[i][0])
            for j in range(0, self.num_total_vars):
                str += " & {}".format(base_info.pB[i][j])
            str += "\\\\ \n"
        str += "\hline\n"
        return str

    def tableau_tex_header(self):
        str = "\\begin{center}\n\\begin{tabular}{c|c|"
        str += "c" * self.num_total_vars
        str += "|}\n"
        str += " & $z$"
        for i in range(self.num_total_vars):
            str += " & ${}$".format(self.var_names[i])
        str += "\\\\ \n"
        return str

    def tableau_tex_wrap(self):
        return "\end{tabular}\n\end{center}\n"

    def compose_tableau(self, basic_solutions):
        str = self.tableau_tex_header()
        for base in basic_solutions:
            str += self.tableau_basic_sol_tex(base)
        str += self.tableau_tex_wrap()
        return str

    def formulation_tex(self):
        tex_str = "\\begin{equation}\n\\begin{split}\n"
        if 'min' in self.sense.lower():
            tex_str += "\mbox{min. } z = " + self.objective_function + "\\\\\n"
        else:
            tex_str += "\mbox{max. } z = " + self.objective_function + "\\\\\n"
        tex_str += "s.a.:\\\\\n"
        char_to_replace = {'<=': '\\leq',
                           '>=': '\\geq'}
        for expression in self.constraints:
            # Iterate over all key-value pairs in dictionary
            for key, value in char_to_replace.items():
                # Replace key character with value character in string
                expression = expression.replace(key, value)
            tex_str += expression + "\\\\\n"
        for i in range(1, self.num_vars + 1):
            tex_str += "x_{}".format(i)
            if i != self.num_vars:
                tex_str += ",\\,\\,"
        tex_str += "\geq 0\\\\\n"
        tex_str += "\\end{split}\n\\end{equation}"
        return tex_str

    def formulation_phase1_tex(self):
        tex_str = "\\begin{equation}\n\\begin{split}\n"

        tex_str += "\mbox{max. } z' = "
        for i, j in enumerate(self.c1):
            if j[0] < 0:
                tex_str += fraction_to_tex(array_to_fraction(j)[0]) + self.var_names[i]
            elif j[0] > 0 and i > 0:
                tex_str += " + " + fraction_to_tex(array_to_fraction(j)[0]) + self.var_names[i]
        tex_str += "\\\\\n"
        tex_str += "s.a.:\\\\\n"
        non_neg_tr = ""
        A, b = arrays_to_fraction([self.A, self.b])
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                if A[i][j] > 0 and j > 0:
                    tex_str += "+"
                if A[i][j] != 0:
                    tex_str += fraction_to_tex(A[i][j]) + self.var_names[j]
            tex_str += " = " + fraction_to_tex(b[i][0])
            tex_str += "\\\\\n"
        tex_str += ",\,".join(self.var_names) + "\\geq 0"

        tex_str += "\\end{split}\n\\end{equation}"

        return tex_str

    # def compose_tableaux(self, first_tableau=0, last_tableau=float('inf')):
    #     last_tableau = min(len(self.tableaux_list), last_tableau)
    #     first_tableau = max(0, min(first_tableau, last_tableau - 1))
    #     str = self.tableau_tex_header()
    #     for i in range(first_tableau, last_tableau):
    #         str += self.tableaux_list[i]
    #     str += self.tableau_tex_wrap()
    #     return str


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

def fraction_to_tex(frac, use_dolar=False):
    if frac.denominator == 1 or frac.denominator == 0:
        str = "{:.0f}".format(frac.numerator)
    else:
        if use_dolar:
            str = "$\\frac{{{}}}{{{}}}$".format(frac.numerator, frac.denominator)
        else:
            str = "\\frac{{{}}}{{{}}}".format(frac.numerator, frac.denominator)
    return str


def matrix_to_tex(m, brackets="round"):
    if brackets == "round":
        str = "\\begin{pmatrix}\n"
    for r in range(m.shape[0]):
        for c in range(m.shape[1] - 1):
            element = m[r, c]
            if isinstance(element, Fraction):
                str += "{} & ".format(fraction_to_tex(element, use_dolar=False))
            elif int(element) == element:
                str += "{:.0f} & ".format(element)
            else:
                str += "{} & ".format(element)
        element = m[r, m.shape[1] - 1]
        if isinstance(element, Fraction):
            str += "{}".format(fraction_to_tex(m[r, m.shape[1] - 1], use_dolar=False))
        elif int(element) == element:
            str += "{:.0f}".format(m[r, m.shape[1] - 1])
        else:
            str += "{}".format(element)
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


if __name__ == "__main__":
    objective = ("max", "3x_1 + 2x_2 + 1x_3 + 2x_4")
    constraints = [
        "1x_1 + 3x_2 + 0x_3 = 60",
    "2x_1 + 1x_2 + 3x_3 + 1x_4 <= 10",
    "2x_1 + 1x_2 + 1x_3 -5x4 >= 50"

]
    problem = LP_Problem(num_vars=4, objective_function=objective, constraints=constraints)

    copy(problem.formulation_tex())
    copy(problem.formulation_phase1_tex())

    tableau = problem.compose_tableau(problem.simplex_bases)
    copy(tableau)

    print("finished")