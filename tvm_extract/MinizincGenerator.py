import os
from typing import List
from AllocationFinder import AllocationFinder
from MemBlock import MemBlock
import numpy as np


def generate_minizinc_model(alloc_finder: AllocationFinder) -> str:
    """
    Generate a MiniZinc program from AllocationFinder output, modeling 2D strip packing
    with data dependencies and dynamic lifetimes.
    """
    memblocks: List[MemBlock] = []
    for blocks in alloc_finder.memblocks.values():
        memblocks.extend(blocks)

    # Assign indices to memblocks
    mb_index = {mb._id: i + 1 for i, mb in enumerate(memblocks)}
    n = len(memblocks)

    # Memory footprint: fixed memory size (in 32-byte units)
    sizeY = [max(1, int(np.log2(np.prod(mb.shape)))) for mb in memblocks]
    sum_sizey = sum(sizeY)

    operation_costs = [1 for _ in memblocks]
    sum_opcost = sum(operation_costs)

    # Lifetime constraints
    lifetime_constraints = []
    for mb in memblocks:
        i = mb_index[mb._id]

        # Each jth dependency must remain alive during computation of ith mb
        for dep in mb.depends_on:
            j = mb_index[dep._id]
            lifetime_constraints.append(f"constraint posX[{j}] + sizeX[{j}] >= posX[{i}] + op_cost[{i}];")
            lifetime_constraints.append(f"constraint posX[{j}] <= posX[{i}];")

    # if links_to and depends_on constraints are equal and 1-to-1 filled in, this is useless.
    # for mb in memblocks:
    #     i = mb_index[mb._id]
    
    #     for dep in mb.links_to:
    #         j = mb_index[dep._id]
    #         lifetime_constraints.append(f"constraint posX[{i}] + sizeX[{i}] >= posX[{j}] + op_cost[{j}];")

    dependency_block = "\n".join(lifetime_constraints)

    # MiniZinc array string for sizeY
    sizeY_arr = ",\n    ".join(str(y) for y in sizeY)
    op_cost_arr = ",\n    ".join(str(c) for c in operation_costs)

    minizinc_code = f"""
% Automatically generated MiniZinc model from TVM AllocationFinder
include "alldifferent.mzn";

int: n = {n};
int: memsize = {int(sum_sizey / 2)};

array[1..n] of var 1..{5 * n + 5 * sum_opcost}: sizeX;  % Time dimension (lifetime)
array[1..n] of int: sizeY = [
    {sizeY_arr}
];
array[1..n] of int: op_cost = [
    {op_cost_arr}
];

array[1..n] of var 0..{5 * n + 5 * sum_opcost}: posX;  % Start time
array[1..n] of var 0..(memsize-1): posY;  % Memory offset

constraint alldifferent([posX[i] | i in 1..n]); % No two allocations / computations at once

% Total time is the max horizontal usage
var int: total_time;

constraint forall(i in 1..n) (
    posY[i] + sizeY[i] <= memsize
);

% Lifetime constraints respecting dependencies
{dependency_block}

% Non-overlapping constraints
constraint forall(i, j in 1..n where i < j) (
    (posX[i] + sizeX[i] <= posX[j]) \/
    (posX[j] + sizeX[j] <= posX[i]) \/
    (posY[i] + sizeY[i] <= posY[j]) \/
    (posY[j] + sizeY[j] <= posY[i])
);

constraint total_time = max([posX[i] + sizeX[i] | i in 1..n]);

solve minimize total_time;

output [
  "total_time = ", show(total_time), "\\n",
  "positions = [", 
  concat(["[" ++ show(posX[i]) ++ "," ++ show(posY[i]) ++ "]" ++ if i != n then ", " else "" endif | i in 1..n]), 
  "]\\n",
  "sizes = [",
  concat(["[" ++ show(sizeX[i]) ++ "," ++ show(sizeY[i]) ++ "]" ++ if i != n then ", " else "" endif | i in 1..n ]),
  "]\\n",
];
"""
    return minizinc_code

def save_minizinc_model(code: str, path: str = "model.mzn"):
    with open(path, "w") as f:
        f.write(code)
    print(f"[+] MiniZinc model saved to {path}")
