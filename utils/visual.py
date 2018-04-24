import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
def show_frame_boxes(img, bboxes, fig_n=1):
    img = img.resize((img.size[0],img.size[1]))
    fig = plt.figure(fig_n)
    ax = fig.add_subplot(111)
    ax.imshow(np.uint8(img))
    for bbox in bboxes: 
        x, y, target_width, target_height = bbox.x, bbox.y, bbox.width, bbox.height
        r = matplotlib.patches.Rectangle((x-0.5*target_width,y-0.5*target_height), target_width, target_height, linewidth=2, edgecolor='r', fill=False)
        ax.add_patch(r)
    plt.ion()
    plt.show()