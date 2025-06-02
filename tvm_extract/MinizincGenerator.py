import os
import json

from math import ceil
from typing import List
from AllocationFinder import AllocationFinder
from MemBlock import MemBlock
from collections.abc import Iterable
import numpy as np


def write_dzn(dico):
    total_string = ""
    for key, elt in dico.items():
        if isinstance(elt, list):
            if isinstance(elt[0], Iterable):  # Nested list : 2D array ?
                arr = "\n    |".join(str(y) for y in elt)
            else:  # 1D Array
                arr = ",\n    ".join(str(y) for y in elt)  # Join and indent
            total_string += f"{key} = [\n    {arr}\n];\n"
        else:
            total_string += f"{key} = {elt}\n"
    return total_string


def generate_minizinc_model(alloc_finder: AllocationFinder, export_name="current", export_path="../minizinc/", transfer_fn=lambda x: x) -> str:
    """
    Generate a MiniZinc program from AllocationFinder output, modeling 2D strip packing
    with data dependencies and dynamic lifetimes.
    """
    # Define constraints
    memblocks: List[MemBlock] = []
    for blocks in alloc_finder.memblocks.values():
        memblocks.extend(blocks)

    # Data export
    dzn = {}
    # Memory Sizes
    sizeY = [max(1, int(transfer_fn(np.prod(mb.shape)))) for mb in memblocks]
    sizeY = [ceil(x / min(sizeY)) for x in sizeY]
    sum_sizey = sum(sizeY)
    dzn["sizeY"] = sizeY

    # Operation Costs
    operation_costs = [1 for _ in memblocks]
    sum_opcost = sum(operation_costs)
    dzn["op_cost"] = operation_costs

    # Assign indices to memblocks
    mb_index = {mb._id: i + 1 for i, mb in enumerate(memblocks)}
    n = len(memblocks)

    # Memory footprint: fixed memory size (in 32-byte units)
    # sizeY = [max(1, int(np.log2(np.prod(mb.shape)))) for mb in memblocks]

    index_dep_info: List[str] = []
    # Lifetime constraints
    lifetime_constraints = []
    for mb in memblocks:
        i = mb_index[mb._id]

        for dep in mb.depends_on:
            j = mb_index[dep._id]
            # Each jth dependency must remain alive during computation of ith mb
            lifetime_constraints.append(f"constraint posX[{j}] + sizeX[{j}] >= posX[{i}] + op_cost[{i}];")
            # the jth dependency must be existing before i is computed
            lifetime_constraints.append(f"constraint posX[{j}] <= posX[{i}];")
            index_dep_info.append(f"{j},{i}")  # Dep j -> i for plotting

    dependency_block = "\n".join(lifetime_constraints)
    index_dep_info[0] = "|" + index_dep_info[0]
    index_dep_info[-1] += "|"
    dzn["index_dep_info"] = index_dep_info

    # No too early alloc constraints for nodes with no dependencies (input, params...)
    # This is only useful to reduce the search space
    leaf_constraints = []
    for mb in memblocks:
        i = mb_index[mb._id]
        if not mb.depends_on:  # If it depends on nothing (leaf)
            if len(mb.links_to) > 1:
                raise NotImplementedError("If links_to several nodes, we need to take the least restrictive constraint")
            else:
                j = mb_index[mb.links_to[0]._id]
                leaf_constraints.append(f"constraint posX[{i}] >= posX[{j}] - 1;")
    leaf_block = "\n".join(leaf_constraints)

    minizinc_code = f"""
% Automatically generated MiniZinc model from TVM AllocationFinder
include "diffn_k.mzn";
include "alldifferent.mzn";

int: n = {n};
int: n_dep = {len(index_dep_info)};
int: memsize = {int(sum_sizey / 2)};
int: max_time = {3 * n + 3 * sum_opcost};

% Sizes
array[1..n] of var 1..max_time: sizeX;  % Time dimension (lifetime)
array[1..n] of int: sizeY;
array[1..n, 1..2] of var int: sizes = array2d(1..n, 1..2,
    [ if j=1 then sizeX[i] else sizeY[i] endif | i in 1..n, j in 1..2 ]
);

% Costs
array[1..n] of int: op_cost;

% Positions
array[1..n] of var 0..max_time: posX;
array[1..n] of var 0..(memsize-1): posY;
array[1..n, 1..2] of var int: positions = array2d(1..n, 1..2,
    [ if j=1 then posX[i] else posY[i] endif | i in 1..n, j in 1..2 ]
);

% constraint alldifferent([posX[i] | i in 1..n]); % No two allocations / computations at once

constraint forall(i in 1..n) (
    posY[i] + sizeY[i] <= memsize
);

% Lifetime constraints respecting dependencies
{dependency_block}

% Restrictive dependencies to avoid early allocation of leaf nodes. A leaf node must not be allocated earlier than time - 1 from the tensor it links to
% Only useful to prune the search space.
{leaf_block}

% Non-overlapping constraints
% constraint forall(i, j in 1..n where i < j) (
%     (posX[i] + sizeX[i] <= posX[j]) \/
%     (posX[j] + sizeX[j] <= posX[i]) \/
%     (posY[i] + sizeY[i] <= posY[j]) \/
%     (posY[j] + sizeY[j] <= posY[i])
% );
constraint diffn_k(positions, sizes);

% Total time is the max horizontal usage
var 1..max_time: total_time;  % Decision variable to be minimized
constraint total_time = max([posX[i] + sizeX[i] | i in 1..n]);

% Redundant constraint : The total area of all rectangles must fit within the total area of the strip
% This can provide a lower bound on total_time or constrain the domains of sizeX
constraint redundant_constraint(total_time * memsize >= sum(i in 1..n) (sizeX[i] * sizeY[i]));


% Add Search strategy
solve :: int_search(
    [total_time] ++ [posX[i] | i in 1..n] ++ [posY[i] | i in 1..n] ++ [sizeX[i] | i in 1..n],
    first_fail, % Heuristic: Choose variable with the smallest domain (likely to fail fastest)
    indomain_min, % Value selection: Try the smallest value in the domain first
    % complete % Ensure the search is complete (finds optimal if possible)
) minimize total_time;
% solve minimize total_time;

array[1..n_dep][1..2] of int: index_dep_info;

output [
  "total_time = ", show(total_time), "\\n",
  "positions = [", 
  concat(["[" ++ show(posX[i]) ++ "," ++ show(posY[i]) ++ "]" ++ if i != n then ", " else "" endif | i in 1..n]), 
  "]\\n",
  "sizes = [",
  concat(["[" ++ show(sizeX[i]) ++ "," ++ show(sizeY[i]) ++ "]" ++ if i != n then ", " else "" endif | i in 1..n ]),
  "]\\n",
  "dep_info = [",
  concat(["[" ++ show(index_dep_info[i]) ++ "]" | i in 1..n ]),
  "]\\n",
];
"""
    
    with open(os.path.join(export_path, f"{export_name}_model.mzn"), "w") as f:
        f.write(minizinc_code)

    writable_str = write_dzn(dzn)
    with open(os.path.join(export_path, f"{export_name}_data.dzn"), "w") as f:
        f.write(writable_str)

    return minizinc_code
