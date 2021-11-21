import math

TOL = 1e-4


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
        self.columns = []


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

    # print("is_feasible")
    # print(solution)
    # for val, lb, ub in zip(row_values,
    #                        parameters.row_lower_bounds,
    #                        parameters.row_upper_bounds):
    #     print(lb, val, ub)

    for val, lb, ub in zip(row_values,
                           parameters.row_lower_bounds,
                           parameters.row_upper_bounds):
        if val < lb - TOL:
            return False
        if val > ub + TOL:
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
