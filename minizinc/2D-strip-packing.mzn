% 2D Strip Packing for Memory Allocation

int: memsize = 10;
int: n = 10;


% Memory objects (allocated tensors)
array[1..n, 1..2] of var int: positions;
array[1..n] of var int: posX = col(positions, 1);
array[1..n] of var int: posY = col(positions, 2);

% Sizes of tensors
array[1..n, 1..2] of int: sizes = array2d(
  1..n, 1..2, 
  [
     1,2,
     3,4,
     5,6,
     7,8,
     3,3,
     6,3,
     8,2,
     1,4,
     7,6,
     3,4,
  ]
);
array[1..n] of int: sizeX = col(sizes, 1);
array[1..n] of int: sizeY = col(sizes, 2);

% Objective: total time (max X extent)
int: h_min = max(sizeX);
int: h_max = sum(sizeX);

var h_min..h_max: total_time;

% Constraints
constraint forall(i in 1..n) (
    posX[i] >= 0 /\
    posY[i] >= 0 /\
    posY[i] + sizeY[i] <= memsize
);
% constraint forall(i in 1..n) (posY[i] + sizeY[i] <= memsize);  % Ne dépasse pas la taille mémoire
% constraint total_time = max(i in 1..n)(posX[i] + sizeX[i]);
constraint total_time = max([posX[i] + sizeX[i] | i in 1..n]);

% Non-overlapping constraint
constraint forall(i, j in 1..n where i < j) (
  (posX[i] + sizeX[i] <= posX[j]) \/
  (posX[j] + sizeX[j] <= posX[i]) \/
  (posY[i] + sizeY[i] <= posY[j]) \/
  (posY[j] + sizeY[j] <= posY[i])
);


solve minimize total_time;

output [
  "total_time = ", show(total_time), "\n",
  "positions = [", 
  concat([ "[" ++ show(posX[i]) ++ "," ++ show(posY[i]) ++ "]" ++ if i != n then ", " else "" endif | i in 1..n ]), 
  "]\n",
  "sizes = [",
  concat([ "[" ++ show(sizeX[i]) ++ "," ++ show(sizeY[i]) ++ "]" ++ if i != n then ", " else "" endif | i in 1..n ]),
  "]\n",
];