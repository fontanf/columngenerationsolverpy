import math
import time


class Column:
    """Structure that represents a column."""

    def __init__(self):
        self.row_indices = []
        self.row_coefficients = []
        self.objective_coefficient = 0

        self.branching_priority = 0
        self.extra = None

    def __str__(self):
        s = f"objective coefficient: {self.objective_coefficient}\n"
        s += "row indices: " + " ".join(self.row_indices) + "\n"
        s += "row coefficients: " + " ".join(self.row_coefficients) + "\n"
        return s


class Parameters:
    """Input of the column generation algorithm.

    Contains the structures that describe the master problem and the oracle for
    the pricing problem.
    """

    def __init__(self, number_of_rows):
        self.row_lower_bounds = [None] * number_of_rows
        self.row_upper_bounds = [None] * number_of_rows
        self.row_coefficient_lower_bounds = [None] * number_of_rows
        self.row_coefficient_upper_bounds = [None] * number_of_rows
        self.objective_sense = "min"
        self.column_lower_bound = None
        self.column_upper_bound = None
        self.dummy_column_objective_coefficient = None
        self.pricing_solver = None
        self.columns = None


def display_initialize(parameters, verbose):
    if verbose:
        print("-"*79)
        print('{:10}'.format("Time"), end='')
        if parameters.objective_sense == "min":
            print('{:14}'.format("UB"), end='')
            print('{:14}'.format("LB"), end='')
        else:
            print('{:14}'.format("LB"), end='')
            print('{:14}'.format("UB"), end='')
        print('{:14}'.format("Gap"), end='')
        print('{:14}'.format("Gap (%)"), end='')
        print('{:32}'.format("Comment"), end='')
        print()
        print("-"*79)


def display(primal, dual, message, start, verbose):
    if verbose:
        absolute_gap = abs(primal - dual)
        relative_gap = 100.0 * absolute_gap / max(abs(primal), abs(dual))
        print(
                '{:<10.3f}'.format(time.time() - start)
                + '{:14}'.format(primal)
                + '{:14}'.format(dual)
                + '{:14}'.format(absolute_gap)
                + '{:14}'.format(relative_gap)
                + '{:32}'.format(message))


def display_end(output, start, verbose):
    if verbose:
        total_time = time.time() - start
        primal = output["solution_value"]
        dual = output["bound"]
        absolute_gap = abs(primal - dual)
        relative_gap = 100.0 * absolute_gap / max(abs(primal), abs(dual))
        total_number_of_columns = output["total_number_of_columns"]
        number_of_columns_added = output["number_of_columns_added"]
        print("---")
        print(f"Solution value: {primal}")
        print(f"Bound: {dual}")
        print(f"Absolute gap: {absolute_gap}")
        print(f"Relative gap (%): {relative_gap}")
        print(f"Total number of columns: {total_number_of_columns}")
        print(f"Number of columns added: {number_of_columns_added}")
        print(f"Total time: {total_time}")


def is_feasible(parameters, solution):
    """Return True iff 'solution' is feaisble for the problem described by
    'parameters'.

    Parameters
    ----------
    parameters : Parameters
        'Parameter' structure of the problem.
    solution : list of (int, float)
        Solution given as a list of pairs of column indices and values.

    """

    m = len(parameters.row_lower_bounds)
    row_values = [0.0] * m
    for column_id, value in solution:
        column = parameters.columns[column_id]
        for index, coef in zip(column.row_indices, column.row_coefficients):
            row_values[index] += value * coef

    for val, lb, ub in zip(row_values,
                           parameters.row_lower_bounds,
                           parameters.row_upper_bounds):
        if val < lb:
            return False
        if val > ub:
            return False
    return True


def compute_value(parameters, solution):
    """Compute and return the value of 'solution' for the problem described by
    'parameters'.

    Parameters
    ----------
    parameters : Parameters
        'Parameter' structure of the problem.
    solution : list of (int, float)
        Solution given as a list of pairs of column indices and values.

    """

    return sum(parameters.columns[column_id].objective_coefficient * value
               for column_id, value in solution)


def to_solution(parameters, columns):
    """Convert a solution given as a list pairs of column indices and values to
    a solution given as a list of pairs of columns and values."""

    return [(parameters.columns[column_id], value)
            for column_id, value in columns]


def compute_reduced_cost(column, duals):
    """Compute and return the reduced cost of 'column'."""

    return (column.objective_coefficient -
            sum(duals[index] * coef
                for index, coef in zip(column.row_indices,
                                       column.row_coefficients)))


def norm(new_rows, vector_1, vector_2):
    return math.sqrt(sum(vector_1[i] * vector_2[i] for i in new_rows))
