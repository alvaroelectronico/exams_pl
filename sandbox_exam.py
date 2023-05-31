from exams_pl.problem_class import *
# from problem_class import *

objective = ('maximize', '2x_1 + 1x_2')
constraints = ['3x_1 + 1x_2  <= 30',
               '-4x_1 +3x_2 <= 16',
               '1x_1 -2x_2 >= 4',
               '1x_1 + 3x_2 >= 5']

problem = LP_Problem(num_vars=2, objective_function=objective, constraints=constraints)
print(problem.formulation_phase1_tex())

print(problem.compose_tableau(problem.simplex_bases))
print(problem.tableau_basic_sol_tex(problem.simplex_bases[0]))

problem.simplex_bases
problem.gen_all_bases_info()
problem.bases_info.keys()
problem.bases_info[2, 3, 6, 7].pB

array_to_fraction(problem.)

print(matrix_to_tex(array_to_fraction(problem.c2)))





basic_solution_to_fraction(problem.bases_info[2, 3, 6, 7])
