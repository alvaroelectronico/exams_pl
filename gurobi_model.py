import gurobipy as gp
from gurobipy import GRB
import numpy as np
from fractions import Fraction

class GurobiModel:
    def __init__(self, num_vars, constraints, objective_function, sense):
        self.num_vars = num_vars
        self.constraints = constraints
        self.objective_function = objective_function
        self.sense = sense
        
        self.model = None
        self.vars = None
        self.constrs = None
        
    def solve(self, print_details=True, print_model=True, as_fractions=True):
        """
        Crea y resuelve el modelo usando Gurobi directamente
        
        Args:
            print_details (bool): Si es True, imprime los detalles de la solución
            print_model (bool): Si es True, imprime el modelo antes de resolverlo
            as_fractions (bool): Si True, muestra valores como fracciones
        """
        try:
            # Crear modelo
            self.model = gp.Model("LP_Problem")
            
            # Definir variables primero
            self.vars = self.model.addVars(range(1, self.num_vars + 1), 
                                         lb=0, 
                                         name="x")
            
            # Imprimir el modelo si se solicita
            if print_model:
                self.print_model()
            
            # Definir función objetivo después de crear las variables
            obj_expr = self._parse_objective()
            if 'min' in self.sense.lower():
                self.model.setObjective(obj_expr, GRB.MINIMIZE)
            else:
                self.model.setObjective(obj_expr, GRB.MAXIMIZE)
            
            # Definir restricciones
            self.constrs = []
            for constraint in self.constraints:
                lhs_expr, sense, rhs_val = self._parse_constraint(constraint)
                if sense == '<=':
                    constr = self.model.addConstr(lhs_expr <= rhs_val)
                elif sense == '>=':
                    constr = self.model.addConstr(lhs_expr >= rhs_val)
                else:  # sense == '='
                    constr = self.model.addConstr(lhs_expr == rhs_val)
                self.constrs.append(constr)
            
            # Actualizar el modelo para que refleje todos los cambios
            self.model.update()
            
            # Resolver
            self.model.optimize()
            
            # Verificar solución
            if self.model.status == GRB.OPTIMAL:
                if print_details:
                    self.print_solution_details(as_fractions=as_fractions)
                return True
            return False
            
        except gp.GurobiError as e:
            print(f"Error de Gurobi: {str(e)}")
            return False
    
    def _parse_objective(self):
        """Parsea la función objetivo"""
        obj_expr = 0
        
        # Asegurarnos de que tenemos la expresión correcta
        if isinstance(self.objective_function, tuple):
            obj_str = self.objective_function[1]  # Tomar la expresión de la tupla
        else:
            obj_str = self.objective_function  # Si ya es una cadena
            
        print(f"\nParseando función objetivo: {obj_str}")  # Debug
        
        # Dividir por '+' y procesar cada término
        terms = [term.strip() for term in obj_str.replace('-', '+-').split('+')]
        terms = [term for term in terms if term]  # Eliminar términos vacíos
        
        print(f"Términos encontrados: {terms}")  # Debug
        
        for term in terms:
            term = term.strip()
            if 'x_' in term:
                # Separar coeficiente y variable
                if term.startswith('x_'):
                    coeff = 1
                    var_idx = int(term.split('x_')[1])
                elif term.startswith('-x_'):
                    coeff = -1
                    var_idx = int(term.split('x_')[1])
                else:
                    parts = term.split('x_')
                    coeff_str = parts[0].strip()
                    var_idx = int(parts[1])
                    
                    try:
                        coeff = float(coeff_str)
                    except ValueError as e:
                        print(f"Error al convertir coeficiente: {coeff_str}")
                        continue
                
                print(f"Término: {term}, Coeficiente: {coeff}, Variable: x_{var_idx}")  # Debug
                obj_expr += coeff * self.vars[var_idx]
        
        return obj_expr
    
    def _parse_constraint(self, constraint):
        """Parsea una restricción"""
        if '<=' in constraint:
            lhs, rhs = constraint.split('<=')
            sense = '<='
        elif '>=' in constraint:
            lhs, rhs = constraint.split('>=')
            sense = '>='
        else:
            lhs, rhs = constraint.split('=')
            sense = '='
        
        lhs_expr = 0
        terms = lhs.split('+')
        for term in terms:
            term = term.strip()
            if 'x_' in term:
                if term.startswith('x_'):
                    coeff = 1
                    var = int(term.split('x_')[1])
                else:
                    coeff = float(term.split('x_')[0])
                    var = int(term.split('x_')[1])
                lhs_expr += coeff * self.vars[var]
        
        rhs_val = float(rhs.strip())
        return lhs_expr, sense, rhs_val
    
    def get_solution(self):
        """Obtiene los valores de las variables"""
        return {i: self.vars[i].x for i in range(1, self.num_vars + 1)}
    
    def get_objective_value(self):
        """Obtiene el valor de la función objetivo"""
        return self.model.objVal
    
    def get_reduced_costs(self):
        """Obtiene los costes reducidos"""
        return {i: self.vars[i].RC for i in range(1, self.num_vars + 1)}
    
    def get_dual_values(self):
        """Obtiene los valores duales (precios sombra)"""
        return [constr.pi for constr in self.constrs]
    
    def get_obj_coefficients_sensitivity(self):
        """Obtiene el análisis de sensibilidad de los coeficientes de la función objetivo"""
        sensitivity = {}
        for i in range(1, self.num_vars + 1):
            var = self.vars[i]
            sensitivity[f'x_{i}'] = {
                'current_value': var.obj,
                'lower_bound': var.SAObjLow,
                'upper_bound': var.SAObjUp
            }
        return sensitivity
    
    def get_rhs_sensitivity(self):
        """Obtiene el análisis de sensibilidad de los términos independientes"""
        sensitivity = {}
        for i, constr in enumerate(self.constrs):
            sensitivity[f'restriccion_{i+1}'] = {
                'current_value': constr.RHS,
                'lower_bound': constr.SARHSLow,
                'upper_bound': constr.SARHSUp,
                'dual_value': constr.pi,
                'slack': constr.slack
            }
        return sensitivity
    
    def _to_fraction_str(self, value, tolerance=1e-10):
        """
        Convierte un float a string de fracción, manejando casos especiales
        """
        try:
            if abs(value) < tolerance:
                return "0"
            if value >= 1e30:  # Consideramos esto como infinito
                return "∞"
            if value <= -1e30:  # Consideramos esto como menos infinito
                return "-∞"
            if abs(value) > 1e6:  # Para números muy grandes, usar notación decimal
                return f"{value:.2f}"
            # Para números normales, usar fracciones
            return str(Fraction(value).limit_denominator())
        except (OverflowError, ValueError):
            # Si hay algún error en la conversión, devolver el valor como decimal
            return f"{value:.4f}"

    def print_solution_details(self, as_fractions=True):
        """
        Imprime todos los detalles de la solución
        
        Args:
            as_fractions (bool): Si True, muestra valores como fracciones
        """
        print(f"\nSolución óptima encontrada:")
        if as_fractions:
            print(f"Valor objetivo: {self._to_fraction_str(self.get_objective_value())}")
        else:
            print(f"Valor objetivo: {self.get_objective_value():.4f}")
        
        print("\nValores de las variables:")
        for i, val in self.get_solution().items():
            if as_fractions:
                print(f"x_{i} = {self._to_fraction_str(val)}")
            else:
                print(f"x_{i} = {val:.4f}")
        
        print("\nPrecios sombra (valores duales):")
        for i, dual in enumerate(self.get_dual_values(), 1):
            if as_fractions:
                print(f"Restricción {i}: {self._to_fraction_str(dual)}")
            else:
                print(f"Restricción {i}: {dual:.4f}")
        
        print("\nCostes reducidos:")
        for var, rc in self.get_reduced_costs().items():
            if as_fractions:
                print(f"x_{var}: {self._to_fraction_str(rc)}")
            else:
                print(f"x_{var}: {rc:.4f}")
        
        print("\nAnálisis de sensibilidad de coeficientes de la función objetivo:")
        for var, data in self.get_obj_coefficients_sensitivity().items():
            print(f"\n{var}:")
            if as_fractions:
                print(f"  Coeficiente actual: {self._to_fraction_str(data['current_value'])}")
                print(f"  Rango: [{self._to_fraction_str(data['lower_bound'])}, "
                      f"{self._to_fraction_str(data['upper_bound'])}]")
            else:
                print(f"  Coeficiente actual: {data['current_value']:.4f}")
                print(f"  Rango: [{data['lower_bound']:.4f}, {data['upper_bound']:.4f}]")
        
        print("\nAnálisis de sensibilidad de términos independientes (RHS):")
        for rest, data in self.get_rhs_sensitivity().items():
            print(f"\n{rest}:")
            if as_fractions:
                print(f"  Valor actual RHS: {self._to_fraction_str(data['current_value'])}")
                print(f"  Precio sombra: {self._to_fraction_str(data['dual_value'])}")
                print(f"  Slack: {self._to_fraction_str(data['slack'])}")
                print(f"  Rango RHS: [{self._to_fraction_str(data['lower_bound'])}, "
                      f"{self._to_fraction_str(data['upper_bound'])}]")
            else:
                print(f"  Valor actual RHS: {data['current_value']:.4f}")
                print(f"  Precio sombra: {data['dual_value']:.4f}")
                print(f"  Slack: {data['slack']:.4f}")
                print(f"  Rango RHS: [{data['lower_bound']:.4f}, {data['upper_bound']:.4f}]")
    
    def print_model(self):
        """
        Imprime el modelo en formato legible
        """
        print("\nModelo a construir:")
        print("\nFunción objetivo:")
        if 'min' in self.sense.lower():
            print("Minimizar:")
        else:
            print("Maximizar:")
        # Mostrar la expresión completa de la función objetivo
        print(f"  {self.objective_function[1]}")  # Mostrar solo la expresión, no la tupla completa
        
        print("\nRestricciones:")
        for i, constraint in enumerate(self.constraints, 1):
            print(f"  {i}. {constraint}")
            
        print("\nVariables:")
        print(f"  x_1 hasta x_{self.num_vars} ≥ 0")
        
        if self.model is not None and self.vars is not None:
            print("\nModelo construido en Gurobi:")
            print("\nVariables creadas:")
            try:
                # Iterar sobre el diccionario de variables de manera segura
                for i in range(1, self.num_vars + 1):
                    var = self.vars.get(i)
                    if var is not None:
                        name = f"x_{i}"  # Usar nuestro propio formato de nombre
                        print(f"  {name}")
            except Exception as e:
                print(f"Error al mostrar variables: {str(e)}")
    