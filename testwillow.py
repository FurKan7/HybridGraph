import pickle
num = '0'
SL = pickle.load(open("/home/furkan/developments/image_matching/deep-graph-matching-consensus/SL" + num + ".p", 'rb'))
y = pickle.load(open("/home/furkan/developments/image_matching/deep-graph-matching-consensus/y" + num + ".p", 'rb'))
names = pickle.load(open("/home/furkan/developments/image_matching/deep-graph-matching-consensus/names" + num + ".p", 'rb'))
pos = pickle.load(open("/home/furkan/developments/image_matching/deep-graph-matching-consensus/pos" + num + ".p", 'rb'))

pred = SL[y[0]].argmax(dim=-1)

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sb
import matplotlib as mpl
from matplotlib.lines import Line2D
print(mpl.style.available)
mpl.style.use('seaborn-v0_8')
sb.set_style("white")

# colors = sb.color_palette("husl", 10, as_cmap = True)
colors = ['red', 'orange', 'yellow', 'lime', 'cyan', 'dodgerblue', 'blueviolet','hotpink', 'white', 'black']

rc = {"axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,
      "xtick.bottom" : False,
      "xtick.labelbottom" : False,
      "ytick.labelleft" : False,
      "ytick.left" : False}
plt.rcParams.update(rc)

f, ax = plt.subplots(1, 2)  # Adjust the figure size if needed

# Load and process the first image
num = 2
img_path1 = "/home/furkan/developments/image_matching" + names[0][num][2:] + ".png"
img1 = Image.open(img_path1).convert('RGB')
img1 = img1.resize((256, 256))
points1 = pos[0][num*10:num*10+10].cpu().numpy()

# Load and process the second image
img_path2 = "/home/furkan/developments/image_matching" + names[1][num][2:] + ".png"
img2 = Image.open(img_path2).convert('RGB')
img2 = img2.resize((256, 256))
points2 = pos[1][num*10:num*10+10].cpu().numpy()

# Draw the first image and its points
ax[0].imshow(img1, alpha=0.7)
sb.scatterplot(x=points1[:,0], y=points1[:,1], c=colors, s=75, linewidth=1, edgecolor='black', ax=ax[0])

# Draw the second image and its points
colors2 = [colors[i] for i in pred[num*10:num*10+10].cpu()]
ax[1].imshow(img2, alpha=0.7)
sb.scatterplot(x=points2[:,0], y=points2[:,1], c=colors2, s = 75, cmap = colors, linewidth=1, edgecolor='black', ax=ax[1])

# Add lines connecting the points
for i, color in zip(range(len(points1)), colors2):
    # Extract the starting point (x1, y1) and the ending point (x2, y2)
    x1, y1 = points1[i]
    x2, y2 = points2[i]
    
    # Calculate the end point's x coordinate relative to the second subplot
    transFigure = f.transFigure.inverted()
    coord1 = transFigure.transform(ax[0].transData.transform([x1, y1]))
    coord2 = transFigure.transform(ax[1].transData.transform([x2, y2]))
    
    line = mpl.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]), 
                            transform=f.transFigure, color=color, linestyle='-')
    f.lines.append(line)

# Set figure options and save
plt.tight_layout()
plt.savefig("/home/furkan/developments/image_matching/my_plot_with_lines_across.png", dpi=300)
plt.show()



# Normalize the points by the size of the image, if the image is 256x256


# Save the figure with the lines
# plt.savefig("/home/furkan/developments/image_matching/my_corrected_lines_plot.png", dpi=300)
# plt.show()
# print(labels)
print(pred[num*10:num*10+10])