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
positions = [[3,0], [2,15], [0,0], [10,0], [9,15], [1,26], [7,31], [12,21], [14,50], [8,70], [11,50], [13,70], [4,31], [5,46], [6,65]]
sizes = [[5,15], [6,11], [2,5], [5,15], [4,6], [7,5], [8,19], [3,6], [1,20], [4,19], [3,20], [2,20], [2,15], [2,19], [2,19]]
"""

# Parse positions
splitted = minizinc_output.split("\n")
positions = re.findall(r'\[(\d+),(\d+)\]', [x for x in splitted if "positions" in x][0])
sizes = re.findall(r'\[(\d+),(\d+)\]', [x for x in splitted if "sizes" in x][0])
positions = np.array([(int(x), int(y)) for x, y in positions])
sizes = np.array([(int(x), int(y)) for x, y in sizes])

# Plot
fig, ax = plt.subplots()
for (x, y), (w, h) in zip(positions, sizes):
    rect = plt.Rectangle((x, y), w, h, edgecolor='black', facecolor='skyblue', lw=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, f"{w}x{h}", ha='center', va='center', fontsize=8)

# Display strip bounds
total_time = int(re.search(r'total_time\s*=\s*(\d+)', minizinc_output).group(1))
# memsize = 10
# print(f"Max sizes {(positions+sizes)[:, 0]} {max((positions+sizes)[:, 0])}")
ax.set_xlim(0, total_time)
ax.set_ylim(0, max((positions+sizes)[:, 1]))
ax.set_aspect('equal')
ax.set_xlabel("Time")
ax.set_ylabel("Memory Space")
plt.title("2D Strip Packing Memory Layout")
plt.grid(True)
plt.tight_layout()
plt.show()