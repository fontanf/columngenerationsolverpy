from .commons import is_feasible, compute_value, to_solution, TOL
from .column_generation import column_generation

import treesearchsolverpy

import time
import math
from functools import total_ordering


class BranchingScheme:

    @total_ordering
    class Node:

        def __init__(self):
            self.id = None
            self.father = None
            self.column_id = None
            self.column_value = None
            self.value_sum = None
            self.depth = None
            self.discrepancy = None

            self.is_feasible = None
            self.solution_value = None

            self.column_value_best = None
            self.column_id_best = None
            self.next_child_pos = -1
            self.next_child_value = None

        def __lt__(self, other):
            if self.discrepancy != other.discrepancy:
                return self.discrepancy < other.discrepancy
            if self.value_sum != other.value_sum:
                return self.value_sum > other.value_sum
            return self.id < other.id

    def __init__(self, parameters, **kwargs):
        self.parameters = parameters
        self.start = time.time()

        # Read parameters.
        self.verbose = kwargs.get(
                "verbose", True)
        self.maximum_discrepancy = kwargs.get(
                "maximum_discrepancy", float('inf'))

        self.output = {
                "time_lp_solve": 0.0,
                "time_pricing": 0.0,
                "number_of_columns_added": 0,
                "solution": None,
                }

        if self.parameters.objective_sense == "min":
            self.output["solution_value"] = float('+inf')
            self.output["bound"] = float('-inf')
        else:
            self.output["solution_value"] = float('-inf')
            self.output["bound"] = float('+inf')
        self.id = 0

    def root(self):
        node = self.Node()
        node.father = None
        node.column_id = None
        node.column_value = 0
        node.value_sum = 1
        node.discrepancy = 0
        node.depth = 0
        node.guide = 0
        node.id = self.id
        self.id += 1
        return node

    def next_child(self, father):
        # First call to next_child for this node.
        if father.next_child_pos == -1:
            # print("next_child", father, father.discrepancy, father.value_sum)
            father.next_child_pos = 0

            # Compute fixed_columns and tabu.
            fixed_columns = []
            tabu = [False] * len(self.parameters.columns)
            node_tmp = father
            while node_tmp.father is not None:
                if node_tmp.column_value != 0:
                    fixed_columns.append(
                            (node_tmp.column_id, node_tmp.column_value))
                tabu[node_tmp.column_id] = True
                node_tmp = node_tmp.father
            is_feasible(self.parameters, fixed_columns)

            # Run column generation procedure.
            maximum_number_of_iterations = float('inf')
            output_cg = column_generation(
                    self.parameters,
                    fixed_columns=fixed_columns,
                    maximum_number_of_iterations=maximum_number_of_iterations,
                    verbose=False)
            # print(output_cg)
            self.output["time_lp_solve"] += output_cg["time_lp_solve"]
            self.output["time_pricing"] += output_cg["time_pricing"]
            self.output["number_of_columns_added"] \
                += output_cg["number_of_columns_added"]
            if father.depth == 0:
                if (
                        output_cg["number_of_iterations"]
                        < maximum_number_of_iterations):
                    self.output["bound"] = output_cg["solution_value"]
            # If infeasible, prune.
            if output_cg["solution"] is None:
                father.next_child_pos = -2
                return None
            is_feasible(self.parameters, output_cg["solution"])
            # for column_id, column in enumerate(self.parameters.columns):
            #     print(column_id, column.objective_coefficient)
            #     print(column.row_indices)
            #     print(column.row_coefficients)

            # Check bound
            if self.parameters.objective_sense == "min":
                if self.output["solution_value"] \
                        <= output_cg["solution_value"] + TOL:
                    father.next_child_pos = -2
                    return None
            else:
                if self.output["solution_value"] \
                        >= output_cg["solution_value"] - TOL:
                    father.next_child_pos = -2
                    return None

            # Compute next column to branch on.
            column_id_best = None
            column_value_best = None
            diff_best = None
            branching_priority_best = None
            for column_id, column_value in output_cg["solution"]:
                if column_id < len(tabu) and tabu[column_id]:
                    continue
                bp = self.parameters.columns[column_id].branching_priority
                c = math.ceil(column_value)
                if abs(c) > TOL:
                    if (column_id_best is None
                            or branching_priority_best < bp
                            or (branching_priority_best == bp
                                and diff_best > c - column_value)):
                        column_id_best = column_id
                        column_value_best = c
                        diff_best = c - column_value
                        branching_priority_best = bp
                f = math.floor(column_value)
                if abs(f) > TOL:
                    if (
                            column_id_best is None
                            or branching_priority_best < bp
                            or (branching_priority_best == bp
                                and diff_best > column_value - f)):
                        column_id_best = column_id
                        column_value_best = f
                        diff_best = column_value - f
                        branching_priority_best = bp
                # print(column_id, f, column_value, c)
            if column_id_best is None:
                father.next_child_pos = -2
                return None

            # Update father information.
            father.column_id_best = column_id_best
            father.column_value_best = column_value_best
            father.next_child_value = self.parameters.column_lower_bound
            # print(column_id_best, column_value_best, diff_best)
            # print(self.parameters.columns[column_id].row_indices)
            # print(self.parameters.columns[column_id].row_coefficients)

        # Build child node.
        child = self.Node()
        child.father = father
        child.column_id = father.column_id_best
        child.column_value = father.next_child_value
        child.value_sum = father.value_sum + child.column_value
        child.discrepancy = (
                father.discrepancy
                + abs(father.column_value_best - child.column_value))
        child.depth = father.depth + 1
        child.id = self.id
        self.id += 1

        # Update father information.
        father.next_child_value += 1
        if father.next_child_value > self.parameters.column_upper_bound:
            father.next_child_pos = -2

        # Compute child.is_feasible and child.solution_value.
        fixed_columns = []
        node_tmp = child
        while node_tmp.father is not None:
            if node_tmp.column_value != 0:
                fixed_columns.append(
                        (node_tmp.column_id, node_tmp.column_value))
            node_tmp = node_tmp.father
        child.is_feasible = is_feasible(self.parameters, fixed_columns)
        # print(child.is_feasible)
        if child.is_feasible:
            child.solution_value = compute_value(
                    self.parameters, fixed_columns)

        if child.discrepancy > self.maximum_discrepancy:
            return None

        return child

    def infertile(self, node):
        return node.next_child_pos == -2

    def leaf(self, node):
        return False

    def bound(self, node_1, node_2):
        return False

    # Solution pool.

    def better(self, node_1, node_2):
        if not node_1.is_feasible:
            return False
        if not node_2.is_feasible:
            return True
        if self.parameters.objective_sense == "min":
            return node_2.solution_value - TOL > node_1.solution_value
        else:
            return node_2.solution_value + TOL < node_1.solution_value

    def equals(self, node_1, node_2):
        return False

    # Dominances.

    def comparable(self, node):
        return False

    # Outputs.

    def display(self, node):
        return ""


def greedy(parameters, **kwargs):
    # Read parameters.
    verbose = kwargs.get(
            "verbose", True)
    linear_programming_solver = kwargs.get(
            "linear_programming_solver", "CLP")
    time_limit = kwargs.get(
            "time_limit", float('inf'))

    # Initial display.
    if verbose:
        print("======================================")
        print("       Column Generation Solver       ")
        print("======================================")
        print()
        print("Algorithm")
        print("---------")
        print("Greedy")
        print()
        print("Parameters")
        print("----------")
        print(f"Linear programming solver:     "
              f"{linear_programming_solver}")
        print(f"Time limit:                    {time_limit}")

    branching_scheme = BranchingScheme(parameters, **kwargs)
    output_ts = treesearchsolverpy.greedy(
            branching_scheme,
            verbose=False)

    elapsed_time = time.time() - branching_scheme.start
    branching_scheme.output["elapsed_time"] = elapsed_time

    # Compute fixed_columns.
    fixed_columns = []
    node = output_ts["solution_pool"].best
    node_tmp = node
    while node_tmp.father is not None:
        if node_tmp.column_value != 0:
            fixed_columns.append(
                    (node_tmp.column_id, node_tmp.column_value))
        node_tmp = node_tmp.father
    # print(fixed_columns)
    branching_scheme.output["solution"] = to_solution(
            parameters, fixed_columns)
    branching_scheme.output["solution_value"] = node.solution_value

    # Final display.
    if verbose:
        primal = branching_scheme.output["solution_value"]
        dual = branching_scheme.output["bound"]
        if parameters.objective_sense == "min":
            absolute_gap = primal - dual
        else:
            absolute_gap = dual - primal
        denom = max(abs(primal), abs(dual))
        if absolute_gap == 0:
            relative_gap = 0
        elif denom != 0:
            relative_gap = 100.0 * absolute_gap / denom
        else:
            relative_gap = float('inf')
        total_number_of_columns = len(parameters.columns)
        o = branching_scheme.output
        print()
        print("Final statistics")
        print("----------------")
        print(f"Solution value:              {primal}")
        print(f"Bound:                       {dual}")
        print(f"Absolute gap:                {absolute_gap}")
        print(f"Relative gap:                {round(relative_gap, 2)}")
        print(f"Total number of columns:     {total_number_of_columns}")
        print("Time:" + " " * 24 + '{:<11.3f}'.format(elapsed_time))
        print("Time LP solve:" + " " * 15
              + '{:<11.3f}'.format(o['time_lp_solve']))
        print("Time pricing:" + " " * 16
              + '{:<11.3f}'.format(o['time_pricing']))

    return branching_scheme.output


def limited_discrepancy_search(parameters, **kwargs):
    # Read parameters.
    verbose = kwargs.get(
            "verbose", True)
    linear_programming_solver = kwargs.get(
            "linear_programming_solver", "CLP")
    maximum_discrepancy = kwargs.get(
            "maximum_discrepancy", float('inf'))
    time_limit = kwargs.get(
            "time_limit", float('inf'))

    # Initial display.
    if verbose:
        print("======================================")
        print("       Column Generation Solver       ")
        print("======================================")
        print()
        print("Algorithm")
        print("---------")
        print("Limited Discrepancy Search")
        print()
        print("Parameters")
        print("----------")
        print(f"Linear programming solver:     "
              f"{linear_programming_solver}")
        print(f"Maximum discrepancy:           {maximum_discrepancy}")
        print(f"Time limit:                    {time_limit}")

        print()
        print(
                '{:>10}'.format("Time")
                + '{:>14}'.format("Primal")
                + '{:>14}'.format("Dual")
                + '{:>14}'.format("Gap")
                + '{:>14}'.format("Gap (%)")
                + '{:>32}'.format("Comment"))
        print(
                '{:>10}'.format("----")
                + '{:>14}'.format("------")
                + '{:>14}'.format("----")
                + '{:>14}'.format("---")
                + '{:>14}'.format("-------")
                + '{:>32}'.format("-------"))

    branching_scheme = BranchingScheme(parameters, **kwargs)

    def new_solution_callback(output):
        # Compute fixed_columns.
        fixed_columns = []
        node = output["solution_pool"].best
        node_tmp = node
        while node_tmp.father is not None:
            if node_tmp.column_value != 0:
                fixed_columns.append(
                        (node_tmp.column_id, node_tmp.column_value))
            node_tmp = node_tmp.father
        branching_scheme.output["solution"] = to_solution(
                parameters, fixed_columns)
        branching_scheme.output["solution_value"] = node.solution_value
        if verbose:
            primal = branching_scheme.output["solution_value"]
            dual = branching_scheme.output["bound"]
            if parameters.objective_sense == "min":
                absolute_gap = primal - dual
            else:
                absolute_gap = dual - primal
            denom = max(abs(primal), abs(dual))
            if absolute_gap == 0:
                relative_gap = 0
            elif denom != 0:
                relative_gap = 100.0 * absolute_gap / denom
            else:
                relative_gap = float('inf')
            message = (
                    f"node {output['number_of_nodes']}"
                    + f" discrepancy {node.discrepancy}")
            print(
                    '{:>10.3f}'.format(time.time() - branching_scheme.start)
                    + '{:>14f}'.format(primal)
                    + '{:>14f}'.format(dual)
                    + '{:>14f}'.format(absolute_gap)
                    + '{:>14.2f}'.format(relative_gap)
                    + '{:>32}'.format(message))

    treesearchsolverpy.best_first_search(
            branching_scheme,
            new_solution_callback=new_solution_callback,
            time_limit=time_limit,
            verbose=False)

    elapsed_time = time.time() - branching_scheme.start
    branching_scheme.output["elapsed_time"] = elapsed_time

    # Final display.
    if verbose:
        primal = branching_scheme.output["solution_value"]
        dual = branching_scheme.output["bound"]
        if parameters.objective_sense == "min":
            absolute_gap = primal - dual
        else:
            absolute_gap = dual - primal
        denom = max(abs(primal), abs(dual))
        if absolute_gap == 0:
            relative_gap = 0
        elif denom != 0:
            relative_gap = 100.0 * absolute_gap / denom
        else:
            relative_gap = float('inf')
        total_number_of_columns = len(parameters.columns)
        o = branching_scheme.output
        print()
        print("Final statistics")
        print("----------------")
        print(f"Solution value:              {primal}")
        print(f"Bound:                       {dual}")
        print(f"Absolute gap:                {absolute_gap}")
        print(f"Relative gap:                {round(relative_gap, 2)}")
        print(f"Number of columns:           {total_number_of_columns}")
        print("Time:" + " " * 24 + '{:<11.3f}'.format(elapsed_time))
        print("Time LP solve:" + " " * 15
              + '{:<11.3f}'.format(o['time_lp_solve']))
        print("Time pricing:" + " " * 16
              + '{:<11.3f}'.format(o['time_pricing']))

    return branching_scheme.output
