
% Automatically generated MiniZinc model from TVM AllocationFinder
include "diffn_k.mzn";
include "alldifferent.mzn";

int: n = 22;
int: n_dep = 22;
int: memsize = 1073;
int: max_time = 132;

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
constraint posX[20] + sizeX[20] >= posX[8] + op_cost[8];
constraint posX[20] <= posX[8];
constraint posX[2] + sizeX[2] >= posX[8] + op_cost[8];
constraint posX[2] <= posX[8];
constraint posX[3] + sizeX[3] >= posX[9] + op_cost[9];
constraint posX[3] <= posX[9];
constraint posX[8] + sizeX[8] >= posX[10] + op_cost[10];
constraint posX[8] <= posX[10];
constraint posX[9] + sizeX[9] >= posX[10] + op_cost[10];
constraint posX[9] <= posX[10];
constraint posX[10] + sizeX[10] >= posX[11] + op_cost[11];
constraint posX[10] <= posX[11];
constraint posX[21] + sizeX[21] >= posX[12] + op_cost[12];
constraint posX[21] <= posX[12];
constraint posX[4] + sizeX[4] >= posX[12] + op_cost[12];
constraint posX[4] <= posX[12];
constraint posX[5] + sizeX[5] >= posX[13] + op_cost[13];
constraint posX[5] <= posX[13];
constraint posX[12] + sizeX[12] >= posX[14] + op_cost[14];
constraint posX[12] <= posX[14];
constraint posX[13] + sizeX[13] >= posX[14] + op_cost[14];
constraint posX[13] <= posX[14];
constraint posX[22] + sizeX[22] >= posX[15] + op_cost[15];
constraint posX[22] <= posX[15];
constraint posX[6] + sizeX[6] >= posX[15] + op_cost[15];
constraint posX[6] <= posX[15];
constraint posX[7] + sizeX[7] >= posX[16] + op_cost[16];
constraint posX[7] <= posX[16];
constraint posX[15] + sizeX[15] >= posX[17] + op_cost[17];
constraint posX[15] <= posX[17];
constraint posX[16] + sizeX[16] >= posX[17] + op_cost[17];
constraint posX[16] <= posX[17];
constraint posX[14] + sizeX[14] >= posX[18] + op_cost[18];
constraint posX[14] <= posX[18];
constraint posX[17] + sizeX[17] >= posX[18] + op_cost[18];
constraint posX[17] <= posX[18];
constraint posX[18] + sizeX[18] >= posX[19] + op_cost[19];
constraint posX[18] <= posX[19];
constraint posX[1] + sizeX[1] >= posX[20] + op_cost[20];
constraint posX[1] <= posX[20];
constraint posX[11] + sizeX[11] >= posX[21] + op_cost[21];
constraint posX[11] <= posX[21];
constraint posX[11] + sizeX[11] >= posX[22] + op_cost[22];
constraint posX[11] <= posX[22];

% Restrictive dependencies to avoid early allocation of leaf nodes. A leaf node must not be allocated earlier than time - 1 from the tensor it links to
% Only useful to prune the search space.
constraint posX[1] >= posX[20] - 1;
constraint posX[2] >= posX[8] - 1;
constraint posX[3] >= posX[9] - 1;
constraint posX[4] >= posX[12] - 1;
constraint posX[5] >= posX[13] - 1;
constraint posX[6] >= posX[15] - 1;
constraint posX[7] >= posX[16] - 1;

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
  "total_time = ", show(total_time), "\n",
  "positions = [", 
  concat(["[" ++ show(posX[i]) ++ "," ++ show(posY[i]) ++ "]" ++ if i != n then ", " else "" endif | i in 1..n]), 
  "]\n",
  "sizes = [",
  concat(["[" ++ show(sizeX[i]) ++ "," ++ show(sizeY[i]) ++ "]" ++ if i != n then ", " else "" endif | i in 1..n ]),
  "]\n",
  "dep_info = [",
  concat(["[" ++ show(index_dep_info[i]) ++ "]" | i in 1..n ]),
  "]\n",
];
