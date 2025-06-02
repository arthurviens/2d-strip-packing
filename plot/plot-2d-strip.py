import matplotlib.pyplot as plt
import numpy as np
import re

# Paste your MiniZinc output here
minizinc_output = """
total_time = 44
positions = [[43,0], [39,0], [28,0], [15,0], [36,0], [22,0], [0,0], [42,0], [8,0], [33,0]]
sizes = [[1,2], [3,4], [5,6], [7,8], [3,3], [6,3], [8,2], [1,4], [7,6], [3,4]]
"""

# minizinc_output = """
# total_time = 28
# positions = [[10,0], [7,0], [11,0], [16,2], [6,6], [0,6], [16,0], [27,0], [0,0], [24,0]]
# sizes = [[1,2], [3,4], [5,6], [7,8], [3,3], [6,3], [8,2], [1,4], [7,6], [3,4]]
# """

minizinc_output = """
total_time = 21
positions = [[5,4], [3,0], [0,4], [14,2], [0,0], [5,6], [13,0], [13,2], [6,0], [11,6]]
sizes = [[1,2], [3,4], [5,6], [7,8], [3,3], [6,3], [8,2], [1,4], [7,6], [3,4]]
"""

minizinc_output = """
total_time = 15
positions = [[0,0], [2,0], [4,0], [9,0], [11,0], [5,5], [7,0], [12,6], [14,0], [8,35], [10,15], [13,35], [1,30], [3,11], [6,30]]
sizes = [[2,15], [2,11], [2,5], [2,15], [2,6], [2,5], [2,19], [2,6], [1,20], [3,19], [4,20], [2,20], [3,15], [4,19], [2,19]]
"""

minizinc_output = """
total_time = 4
positions = [[0,341], [0,579], [0,148], [0,533], [0,146], [0,145], [0,150], [0,531], [3,205], [1,0], [1,295], [2,0], [0,295], [0,386], [0,0]]
sizes = [[1,45], [1,10], [1,1], [2,46], [1,2], [1,1], [2,145], [3,2], [1,205], [1,150], [2,205], [2,205], [1,46], [1,145], [1,145]]
"""

minizinc_output = """
total_time = 22
positions = [[0,0], [2,191], [4,145], [9,150], [11,0], [14,147], [16,0], [5,290], [7,410], [12,615], [17,617], [18,0], [21,0], [8,0], [10,205], [19,410], [20,205], [13,0], [15,410], [1,145], [3,0], [6,145]]
sizes = [[2,45], [2,10], [2,1], [2,46], [2,2], [2,27], [2,2], [2,1], [7,145], [8,2], [2,2], [3,205], [1,205], [3,150], [10,205], [2,205], [2,205], [3,147], [4,205], [3,46], [4,145], [2,145]]
dep_info = [[3,8], [22,9], [5,10], [7,11], [19,12], [11,12], [17,13], [9,14], [14,15], [4,15], [15,16], [10,16], [16,17], [12,17], [9,18], [18,19], [6,19], [1,20], [20,21], [2,21], [21,22], [8,22]]
"""

# Parse positions
splitted = minizinc_output.split("\n")
positions = re.findall(r'\[(\d+),(\d+)\]', [x for x in splitted if "positions" in x][0])
sizes = re.findall(r'\[(\d+),(\d+)\]', [x for x in splitted if "sizes" in x][0])
dep_info = re.findall(r'\[(\d+),(\d+)\]', [x for x in splitted if "dep_info" in x][0]) if "dep_info" in minizinc_output else None
positions = np.array([(int(x), int(y)) for x, y in positions])
sizes = np.array([(int(x), int(y)) for x, y in sizes])
dep_info = np.array([(int(x), int(y)) for x, y in dep_info]) if dep_info is not None else None

# Plot
fig, ax = plt.subplots()
# ax.set_yscale("log")
for (x, y), (w, h) in zip(positions, sizes):
    rect = plt.Rectangle((x, y), w, h, edgecolor='black', facecolor='skyblue', lw=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, f"{int(w)}x{int(h)}", ha='center', va='center', fontsize=8)

# if dep_info is not None:
#     for idx, (from_i, to_j) in enumerate(dep_info):
#         x = positions[from_i - 1][0] #+ sizes[from_i - 1][0] / 2
#         y = positions[from_i - 1][1] + sizes[from_i - 1][1] / 2
#         new_x = positions[to_j - 1][0] #+ sizes[to_j - 1][0] / 2
#         new_y = positions[to_j - 1][1] + sizes[to_j - 1][1] / 2
#         dx = new_x - x
#         dy = new_y - y
#         ax.arrow(x=x, y=y, dx=dx, dy=dy, linestyle="dotted")

# Display strip bounds
total_time = int(re.search(r'total_time\s*=\s*(\d+)', minizinc_output).group(1))
# memsize = 10
# print(f"Max sizes {(positions+sizes)[:, 0]} {max((positions+sizes)[:, 0])}")
ax.set_xlim(0, total_time)
ax.set_ylim(0, max((positions+sizes)[:, 1]))
ax.set_xlabel("Time")
ax.set_ylabel("Memory Space")
plt.title("2D Strip Packing Memory Layout")
plt.grid(True)
plt.tight_layout()
plt.show()