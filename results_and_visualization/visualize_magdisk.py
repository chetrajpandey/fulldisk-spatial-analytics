import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import colorConverter

# Load the input image
# img = plt.imread('HMI.m2011.12.28_12.00.00.png')
img = plt.imread('mag1.png')

"""Define the radius of the circle to be drawn (relative to the radius of the input circle)
The original img size is 4096 by 4096. From the center, the image has 2048 pixels on each side. Now the 400 pixels on each side are used as padding
which means the radius of the full-disk is 1648 which corresponds to the circle with radius ranging from -90 to 90.
for a circle with radius ranging from -70 to 70. (1648/9)*7  = 1282
r = 1282
However for a snapshot taken from helioviewer, following radius (435px) were determined for the outer circle including -90 to 90
Based on that the the central locations were in the radius of 338 pixels.
"""
r1=435 #full-disk radius
r = 338 #central radius

# Create a new figure and plot the input image
fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')

# Remove ticks from the x and y axis
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)

# Define the coordinates for the center of the circle
x, y = img.shape[1] // 2, img.shape[0] // 2
fc = colorConverter.to_rgba('mediumblue', alpha=0.03)

# Draw a inner circle with radius r (central radius)centered at (x, y) on top of the input image
circle = plt.Circle((x, y), r, ec='mediumblue', fc=fc, fill=True, linewidth=1.2,  linestyle='--')
ax.add_artist(circle)

# Draw a outer circle to verify: circle with radius r1 centered at (x, y) on top of the input image
# circle1 = plt.Circle((x, y), r1, color='blue', fill=False, linestyle='--', linewidth=0.5)
# ax.add_artist(circle1)

# Draw a diameter line and add arrows pointing left and right
diameter = 2 * r
x_vals = [x - diameter/2, x + diameter/2]
y_vals = [y, y]
# ax.plot(x_vals, y_vals, 'r--', linewidth=2)
ax.arrow(x, y, (-diameter/2)+15, 0, head_width=10, head_length=15, fc='mediumblue', ec='mediumblue', lw=0.5, linestyle='dashed')
ax.arrow(x, y, (diameter/2)-15, 0, head_width=10, head_length=15, fc='mediumblue', ec='mediumblue', lw=0.5, linestyle='dashed')
ax.scatter(x, y, c='mediumblue', s=10, marker='|')

# Add text "+70°" and "-70°" on top of the arrows
ax.text((x-diameter/2)+60, y, '-70\u00B0', fontsize=8.4, color='white', va='bottom', ha='right')
ax.text((x+diameter/2)-65, y, '+70\u00B0', fontsize=8.4, color='white', va='bottom', ha='left')
ax.text(x, y-5, '0\u00B0', fontsize=8.4, color='white', va='bottom', ha='left')

# Add text "+70°" and "-70°" on top of the arrows
ax.text((x-diameter/2)-12, y+10, '$<$-70\u00B0', fontsize=8.4, color='white', va='bottom', ha='right')
ax.text((x+diameter/2)+7, y+10, '$>$+70\u00B0', fontsize=8.4, color='white', va='bottom', ha='left')

ax.text(x-445, y+10, 'E', fontsize=8.4, color='white', va='bottom', ha='right')
ax.text(x+445, y+10, 'W', fontsize=8.4, color='white', va='bottom', ha='left')

plt.tight_layout(pad=0)

# Save the resulting image
plt.savefig('annotated magneto.pdf', bbox_inches='tight', dpi=500)

