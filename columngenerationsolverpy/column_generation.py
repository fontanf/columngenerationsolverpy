import time
import pulp

from .commons import compute_reduced_cost, TOL


def column_generation(parameters, **optional_parameters):
    start = time.time()
    # Read parameters.
    linear_programming_solver = optional_parameters.get(
            "linear_programming_solver", "CLP")
    maximum_number_of_iterations = optional_parameters.get(
            "maximum_number_of_iterations", float('inf'))
    time_limit = optional_parameters.get(
            "time_limit", float('inf'))
    verbose = optional_parameters.get(
            "verbose", True)
    fixed_columns = optional_parameters.get(
            "fixed_columns", [])
    debug = optional_parameters.get(
            "debug", False)

    if verbose:
        print("======================================")
        print("       Column Generation Solver       ")
        print("======================================")
        print()
        print("Algorithm")
        print("---------")
        print("Column Generation")
        print()
        print("Parameters")
        print("----------")
        print(f"Linear programming solver:     {linear_programming_solver}")
        print(f"Maximum number of iterations:  {maximum_number_of_iterations}")
        print(f"Time limit:                    {time_limit}")

    # Initialize output structure.
    output = {
            "solution": None,
            "solution_value": None,
            "number_of_iterations": 0,
            "number_of_columns_added": 0,
            "time_lp_solve": 0.0,
            "time_pricing": 0.0}

    # Initial print.
    if verbose:
        print()
        print(
                '{:>10}'.format("Time")
                + '{:>8}'.format("It")
                + '{:>14}'.format("Obj")
                + '{:>8}'.format("Col"))
        print(
                '{:>10}'.format("----")
                + '{:>8}'.format("--")
                + '{:>14}'.format("---")
                + '{:>8}'.format("---"))

    m = len(parameters.row_lower_bounds)

    # Compute row values.
    row_values = [0.0] * m
    c0 = 0.0
    for column_id, value in fixed_columns:
        column = parameters.columns[column_id]
        for index, coef in zip(column.row_indices, column.row_coefficients):
            row_values[index] += value * coef
        c0 += column.objective_coefficient

    # Compute fixed rows.
    new_row_indices = [-2] * m
    new_rows = []
    row_pos = 0
    for row in range(m):
        if (
                parameters.column_lower_bound >= 0
                and parameters.row_coefficient_lower_bounds[row] >= 0
                and row_values[row] > parameters.row_upper_bounds[row]):
            # Infeasible.
            return output
        if (
                parameters.column_lower_bound >= 0
                and parameters.row_coefficient_lower_bounds[row] >= 0
                and row_values[row] == parameters.row_upper_bounds[row]):
            continue
        new_row_indices[row] = row_pos
        new_rows.append(row)
        row_pos += 1
    new_number_of_rows = row_pos
    if new_number_of_rows == 0:
        return output

    # Compute new row bounds.
    new_row_lower_bounds = [None] * new_number_of_rows
    new_row_upper_bounds = [None] * new_number_of_rows
    for row in range(new_number_of_rows):
        new_row_lower_bounds[row] = (
                parameters.row_lower_bounds[new_rows[row]]
                - row_values[new_rows[row]])
        new_row_upper_bounds[row] = (
                parameters.row_upper_bounds[new_rows[row]]
                - row_values[new_rows[row]])

    # Initialize solver
    if parameters.objective_sense == "min":
        master_problem = pulp.LpProblem("Master", pulp.LpMinimize)
    else:
        master_problem = pulp.LpProblem("Master", pulp.LpMaximize)
    objective = pulp.LpConstraintVar("Objective")
    master_problem.setObjective(objective)
    constraints = []
    for row in range(new_number_of_rows):
        var_lb = pulp.LpConstraintVar(
                "c" + str(row) + "_lb",
                pulp.LpConstraintGE,
                new_row_lower_bounds[row])
        constraints.append(var_lb)
        master_problem += var_lb
        var_ub = pulp.LpConstraintVar(
                "c" + str(row) + "_ub",
                pulp.LpConstraintLE,
                new_row_upper_bounds[row])
        constraints.append(var_ub)
        master_problem += var_ub

    solver_column_indices = []

    # Add dummy columns.
    number_of_dummy_columns = 0
    solver_column_indices.append(-1)
    number_of_dummy_columns += 1
    pulp.LpVariable(
            "vd-1",
            parameters.column_lower_bound,
            parameters.column_upper_bound,
            pulp.LpContinuous,
            parameters.dummy_column_objective_coefficient * objective)
    for row in range(new_number_of_rows):
        if new_row_lower_bounds[row] > 0:
            solver_column_indices.append(-1)
            number_of_dummy_columns += 1
            pulp.LpVariable(
                    "vd" + str(row),
                    parameters.column_lower_bound,
                    parameters.column_upper_bound,
                    pulp.LpContinuous,
                    (parameters.dummy_column_objective_coefficient * objective
                     + new_row_lower_bounds[row] * constraints[2 * row]))
        if new_row_upper_bounds[row] < 0:
            solver_column_indices.append(-1)
            number_of_dummy_columns += 1
            pulp.LpVariable(
                    "vd" + str(row),
                    parameters.column_lower_bound,
                    parameters.column_upper_bound,
                    pulp.LpContinuous,
                    (parameters.dummy_column_objective_coefficient * objective
                     - new_row_lower_bounds[row] * constraints[2 * row + 1]))

    # Initialize pricing solver.
    infeasible_columns = parameters.pricing_solver.initialize_pricing(
            parameters.columns, fixed_columns)
    feasible = [True] * len(parameters.columns)
    if infeasible_columns is not None:
        for column_id in infeasible_columns:
            feasible[column_id] = False

    # Add initial columns.
    for column_id, column in enumerate(parameters.columns):
        if not feasible[column_id]:
            continue
        ri = []
        rc = []
        ok = True
        for i, c in zip(column.row_indices, column.row_coefficients):
            # The column might not be feasible.
            # For example, it corresponds to the same bin / machine that a
            # currently fixed column or it contains an item / job also
            # included in a currently fixed column.
            if (
                    parameters.column_lower_bound >= 0
                    and c >= 0
                    and row_values[i] + c > parameters.row_upper_bounds[i]):
                ok = False
                break
            if new_row_indices[i] < 0:
                continue
            ri.append(new_row_indices[i])
            rc.append(c)
        if not ok:
            continue
        solver_column_indices.append(column_id)
        pulp.LpVariable(
                "v" + str(column_id),
                parameters.column_lower_bound,
                parameters.column_upper_bound,
                pulp.LpContinuous,
                (column.objective_coefficient * objective
                 + pulp.lpSum(
                     c * constraints[2 * i]
                     for i, c in zip(ri, rc))
                 + pulp.lpSum(
                     c * constraints[2 * i + 1]
                     for i, c in zip(ri, rc))))

    duals = [0.0] * m
    while True:
        # Solve LP
        start_lpsolve = time.time()
        if debug:
            print("Master problem:")
            print(master_problem)
            print()
            print("Solve master problem:")
            master_problem.solve()
            print()
        else:
            master_problem.solve(pulp.PULP_CBC_CMD(msg=0))
        end_lpsolve = time.time()
        output["time_lp_solve"] += end_lpsolve - start_lpsolve
        obj = c0
        if pulp.value(master_problem.objective):
            obj += pulp.value(master_problem.objective)
        if debug:
            print("Objective:", obj)
            print()

        # Display.
        if verbose:
            print('{:>10.3f}'.format(time.time() - start), end='')
            print('{:>8}'.format(output["number_of_iterations"]), end='')
            print('{:>14f}'.format(obj), end='')
            print('{:>8}'.format(output["number_of_columns_added"]), end='')
            print()

        # Check time.
        if time.time() - start > time_limit:
            break
        # Check maximum number of iterations.
        if output["number_of_iterations"] > maximum_number_of_iterations:
            break

        # Search for new columns.
        # Get duals from linear programming solver.
        for row_pos in range(new_number_of_rows):
            d1 = master_problem.constraints["c" + str(row_pos) + "_ub"].pi
            d2 = master_problem.constraints["c" + str(row_pos) + "_lb"].pi
            duals[new_rows[row_pos]] = d1 + d2
        # Call pricing solver on the computed separation point.
        start_pricing = time.time()
        all_columns = parameters.pricing_solver.solve_pricing(duals)
        end_pricing = time.time()
        output["time_pricing"] += end_pricing - start_pricing
        # Look for negative reduced cost columns.
        new_columns = []
        for column in all_columns:
            rc = compute_reduced_cost(column, duals)
            if debug:
                print("Column:")
                print(column)
                print("Reduced cost:", rc)
                print()
            if parameters.objective_sense == "min" and rc <= 0 - TOL:
                new_columns.append(column)
            if parameters.objective_sense == "max" and rc >= 0 + TOL:
                new_columns.append(column)

        # Stop the column generation procedure if no negative reduced cost
        # column has been found.
        if len(new_columns) == 0:
            break

        for column in new_columns:
            # Add new column to the global column pool.
            parameters.columns.append(column)
            output["number_of_columns_added"] += 1
            # Add new column to the local LP solver.
            ri = []
            rc = []
            for i, c in zip(column.row_indices, column.row_coefficients):
                if new_row_indices[i] < 0:
                    continue
                ri.append(new_row_indices[i])
                rc.append(c)
            column_id = len(parameters.columns) - 1
            solver_column_indices.append(column_id)
            pulp.LpVariable(
                    "v" + str(column_id),
                    parameters.column_lower_bound,
                    parameters.column_upper_bound,
                    pulp.LpContinuous,
                    (column.objective_coefficient * objective
                     + pulp.lpSum(
                         c * constraints[2 * i]
                         for i, c in zip(ri, rc))
                     + pulp.lpSum(
                         c * constraints[2 * i + 1]
                         for i, c in zip(ri, rc))))

        output["number_of_iterations"] += 1

    # Compute solution value.
    output["solution_value"] = obj

    # Compute solution.
    output["solution"] = []
    for var in master_problem.variables():
        if var.name[1] == 'd' or var.name == "__dummy":
            continue
        column_id = int(var.name[1:])
        column_value = var.varValue
        if abs(column_value) > TOL:
            output["solution"].append((column_id, column_value))

    if verbose:
        total_time = time.time() - start
        primal = output["solution_value"]
        total_number_of_columns = len(parameters.columns)
        number_of_columns_added = output["number_of_columns_added"]
        number_of_iterations = output["number_of_iterations"]
        time_lp_solve = output["time_lp_solve"]
        time_pricing = output["time_pricing"]
        print()
        print("Final statistics")
        print("----------------")
        print("Time:" + " " * 24 + '{:<11.3f}'.format(total_time))
        print(f"Solution value:              {primal}")
        print(f"Total number of columns:     {total_number_of_columns}")
        print(f"Number of columns added:     {number_of_columns_added}")
        print(f"Number of iterations:        {number_of_iterations}")
        print("Time LP solve:" + " " * 15 + '{:<11.3f}'.format(time_lp_solve))
        print("Time pricing:" + " " * 16 + '{:<11.3f}'.format(time_pricing))

    return output
