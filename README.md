# Column Generation Solver (Python)

A solver based on column generation.

This is the Python3 version of the C++ package [fontanf/columngenerationsolver](https://github.com/fontanf/columngenerationsolver).

![columngeneration](img/columngeneration.jpg?raw=true "columngeneration")

[image source](https://commons.wikimedia.org/wiki/File:ColonnesPavillonTrajan.jpg)

## Description

The goal of this repository is to provide a simple framework to quickly implement heuristic algorithms based on column generation.

Algorithms:
* Column Generation `column_generation`

## Examples

[Cutting Stock Problem](examples/cuttingstock.py)

## Usage, running examples from command line

Install
```shell
pip3 install columngenerationsolverpy
```

Running an example:
```shell
mkdir -p data/cuttingstock/instance
python3 -m examples.cuttingstock -a generator -i data/cuttingstock/instance
python3 -m examples.cuttingstock -a column_generation -i data/cuttingstock/instance_50.json
```

Update:
```shell
pip3 install --upgrade columngenerationsolverpy
```

## Usage, Python library

See examples.

