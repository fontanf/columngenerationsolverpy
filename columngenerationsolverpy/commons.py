import math

TOL = 1e-4


class Column:
    """Structure that represents a column.

    The values are stored sparsely, i.e. only non-zeros are stored.

    Attributes
    ----------

    objective_coefficient : float
        Coefficient of the column in the objective function.
    row_indices : list of int
        Indices of the rows with a non-zero coefficient.
    row_coefficients : list of float
        Values of the non-zeros of the column.
    branching_priority : int
        Branching priority of the column. A column with a smaller priority will
        always be branched first.
    extra : any
        Extra attribute which can be used to store additional information about
        the column which are not contained in the row values. For example, if
        the column represents a route, the order in which the locations are
        visited can be stored in this attribute.

    """

    def __init__(self):
        self.row_indices = []
        self.row_coefficients = []
        self.objective_coefficient = 0

        self.branching_priority = 0
        self.extra = None

    def __str__(self):
        s = f"objective coefficient: {self.objective_coefficient}\n"
        s += "row indices: " + str(self.row_indices) + "\n"
        s += "row coefficients: " + str(self.row_coefficients)
        return s


class Parameters:
    """Input of the column generation algorithm.

    Contains the structures that describe the master problem and the oracle for
    the pricing problem.

    Attributes
    ----------

    objective_sense : "min" or "max"
        Sense of the objective function.
    column_lower_bound : float
        Lower bound of the columns.
    column_upper_bound : float
        Upper bound of the columns.
    dummy_column_objective_coefficient : float
        Coefficient of the dummy columns in the objective function. It should
        be as small as possible to avoid numerical issues, but still ensure
        that no dummy column can be taken in an optimal solution.
    row_lower_bounds : list of float
        Lower bounds of the rows.
    row_upper_bounds : list of float
        Upper bounds of the rows.
    row_coefficient_lower_bounds : list of float
        Lower bounds of the coefficients of the variables for each row.
    row_coefficient_upper_bounds : list of float
        Upper bounds of the coefficients of the variables for each row.
    pricing_solver : PricingSolver
        Object solving the pricing problem.
    columns : list of Column
        Structure storing the columns. It can be used to provide initial
        columns. The newly generated columns will be added to this structure.

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
