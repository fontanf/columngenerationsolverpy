"""Cutting Stock Problem.

Input:
- a capacity c
- n items; for each item j = 1..n, a weight wⱼ and a demand qⱼ
Problem:
- pack all items such that the total weight of the items in a bin does not
  exceed the capacity.
Objective:
- minimize the number of bin used.

The linear programming formulation of the problem based on DantzigWolfe
decomposition is written as follows:

Variables:
- yᵏ ∈ {0, qmax} representing a set of item_types fitting into a bin.
  yᵏ = q iff the corresponding set of item_types is selected q times.
  xⱼᵏ = q iff yᵏ contains q copies of item type j, otherwise 0.

Program:

min ∑ₖ yᵏ

qⱼ <= ∑ₖ xⱼᵏ yᵏ <= qⱼ     for all item_types j
                                     (each item selected exactly qⱼ times)
                                                        Dual variables: vⱼ

The pricing problem consists in finding a variable of negative reduced cost.
The reduced cost of a variable yᵏ is given by:
rc(yᵏ) = 1 - ∑ⱼ xⱼᵏ vⱼ
       = - ∑ⱼ vⱼ xⱼᵏ + 1

Therefore, finding a variable of minimum reduced cost reduces to solving
a Bounded Knapsack Problem with item_types with profit vⱼ.
"""

import columngenerationsolverpy

import json


class ItemType:
    id = -1
    weight = 0
    demand = 0


class Instance:

    def __init__(self, filepath=None):
        self.item_types = []
        if filepath is not None:
            with open(filepath) as json_file:
                data = json.load(json_file)
                self.capacity = data["capacity"]
                item_types = zip(
                        data["item_type_weights"],
                        data["item_type_demands"])
                for (weight, demand) in item_types:
                    self.add_item_type(weight, demand)

    def add_item_type(self, weight, demand):
        item_type = ItemType()
        item_type.id = len(self.item_types)
        item_type.weight = weight
        item_type.demand = demand
        self.item_types.append(item_type)

    def write(self, filepath):
        data = {"capacity": self.capacity,
                "item_type_weights": [item.weight for item in self.item_types],
                "item_type_demands": [item.demand for item in self.item_types]}
        with open(filepath, 'w') as json_file:
            json.dump(data, json_file)

        # with open(filepath + "_", 'w') as f:
        #     f.write(str(len(self.item_types))
        #             + " " + str(self.capacity) + "\n")
        #     for item_type in self.item_types:
        #         f.write(str(item_type.weight)
        #                 + " " + str(item_type.demand) + "\n")

    def check(self, filepath):
        print("Checker")
        print("-------")
        with open(filepath) as json_file:
            data = json.load(json_file)
            # Compute number_of_bins.
            number_of_bins = sum(copies for copies, items in data["items"])
            # Compute number_of_overweighted_bins.
            number_of_overweighted_bins = 0
            for copies, items in data["items"]:
                weight = sum(self.item_types[item_type_id].weight
                             for item_type_id in items)
                if weight > self.capacity:
                    number_of_overweighted_bins += copies
            # Compute number_of_unsatisfied_demands.
            demands = {item_type_id: 0
                       for item_type_id in range(len(self.item_types))}
            for copies, items in data["items"]:
                for item_type_id in items:
                    demands[item_type_id] += copies
            number_of_unsatisfied_demands = 0
            for item_type_id, item_type in enumerate(self.item_types):
                if demands[item_type_id] != item_type.demand:
                    number_of_unsatisfied_demands += 1

            is_feasible = (
                    (number_of_unsatisfied_demands == 0)
                    and (number_of_overweighted_bins == 0))
            objective_value = number_of_bins
            print(f"Number of overweighted bins: "
                  f"{number_of_overweighted_bins}")
            print(f"Number of unsatisfied demands: "
                  f"{number_of_unsatisfied_demands}")
            print(f"Feasible: {is_feasible}")
            print(f"Objective value: {objective_value}")
            return (is_feasible, objective_value)


class PricingSolver:

    def __init__(self, instance):
        self.instance = instance
        self.filled_demands = None

    def initialize_pricing(self, columns, fixed_columns):
        self.filled_demands = [0] * len(instance.item_types)
        for column_id, column_value in fixed_columns:
            column = columns[column_id]
            for row_index, row_coefficient in zip(column.row_indices,
                                                  column.row_coefficients):
                self.filled_demands[row_index] += (
                        column_value * row_coefficient)

    def solve_pricing(self, duals):
        # Build subproblem instance.
        capacity = self.instance.capacity
        weights = []
        profits = []
        kp2csp = []
        for item_type_id, item_type in enumerate(self.instance.item_types):
            profit = duals[item_type_id]
            if profit <= 0:
                continue
            for _ in range(self.filled_demands[item_type_id],
                           self.instance.item_types[item_type_id].demand):
                profits.append(profit)
                weights.append(self.instance.item_types[item_type_id].weight)
                kp2csp.append(item_type_id)

        # Solve subproblem instance.
        n = len(weights)
        t = [[0 for x in range(capacity + 1)] for y in range(n + 1)]
        for j in range(1, n + 1):
            for x in range(0, weights[j - 1]):
                t[j][x] = t[j - 1][x]
            for x in range(weights[j - 1], capacity + 1):
                t[j][x] = max(t[j - 1][x],
                              t[j - 1][x - weights[j - 1]] + profits[j - 1])
        solution_kp = []
        c = capacity
        for item_id in range(n, 0, -1):
            if t[item_id][c] != t[item_id - 1][c]:
                solution_kp.append(item_id - 1)
                c -= weights[item_id - 1]

        # Retrieve column.
        column = columngenerationsolverpy.Column()
        column.objective_coefficient = 1
        demands = [0] * len(self.instance.item_types)
        for j in solution_kp:
            demands[kp2csp[j]] += 1
        for item_type in self.instance.item_types:
            if demands[item_type.id] > 0:
                column.row_indices.append(item_type.id)
                column.row_coefficients.append(demands[item_type.id])
        return [column]


def get_parameters(instance):
    # Create object to return. Parameter: number of constraints in the
    # exponential formulation.
    p = columngenerationsolverpy.Parameters(len(instance.item_types))
    # Compute the maximum demands. It is used as column upper bound and to
    # define the cost of the dummy columns.
    maximum_demand = max(item_type.demand
                         for item_type in instance.item_types)
    # Objective sense.
    p.objective_sense = "min"
    # Column bounds.
    p.column_lower_bound = 0
    p.column_upper_bound = maximum_demand
    # Row bounds.
    for item_type in instance.item_types:
        p.row_lower_bounds[item_type.id] = item_type.demand
        p.row_upper_bounds[item_type.id] = item_type.demand
        p.row_coefficient_lower_bounds[item_type.id] = 0
        p.row_coefficient_upper_bounds[item_type.id] = item_type.demand
    # Dummy column objective coefficient.
    p.dummy_column_objective_coefficient = 2 * maximum_demand
    # Pricing solver.
    p.pricing_solver = PricingSolver(instance)
    return p


def to_solution(columns, fixed_columns):
    solution = []
    for column, value in fixed_columns:
        s = []
        for index, coef in zip(column.row_indices, column.row_coefficients):
            s += [index] * coef
        solution.append((value, s))
    return solution


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
            "-a", "--algorithm",
            type=str,
            default="column_generation",
            help='')
    parser.add_argument(
            "-i", "--instance",
            type=str,
            help='')
    parser.add_argument(
            "-c", "--certificate",
            type=str,
            default=None,
            help='')

    args = parser.parse_args()

    if args.algorithm == "checker":
        instance = Instance(args.instance)
        instance.check(args.certificate)

    elif args.algorithm == "generator":
        import random
        random.seed(0)
        for number_of_item_types in range(1, 101):
            instance = Instance()
            instance.capacity = 100
            for item_id in range(number_of_item_types):
                weight = random.randint(10, 50)
                demand = random.randint(1, 1000)
                instance.add_item_type(weight, demand)
            instance.write(
                    args.instance + "_" + str(number_of_item_types) + ".json")

    elif args.algorithm == "column_generation":
        instance = Instance(args.instance)
        output = columngenerationsolverpy.column_generation(
                get_parameters(instance))

    else:
        instance = Instance(args.instance)
        parameters = get_parameters(instance)
        if args.algorithm == "greedy":
            output = columngenerationsolverpy.greedy(
                    parameters)
        elif args.algorithm == "limited_discrepancy_search":
            output = columngenerationsolverpy.limited_discrepancy_search(
                    parameters,
                    maximum_discrepancy=1)
        solution = to_solution(parameters.columns, output["solution"])
        if args.certificate is not None:
            data = {"items": solution}
            with open(args.certificate, 'w') as json_file:
                json.dump(data, json_file)
            print()
            instance.check(args.certificate)
