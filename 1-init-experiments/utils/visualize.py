from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import numpy as np
import torch

def create_scatter(coords: torch.Tensor):
    x, y, z = np.vsplit(coords.cpu().numpy(), 3)
    fig = plt.figure() 
    axs = Axes3D(fig) 
    plot = axs.scatter(x, y, z, color='green') 
    axs.set_xlabel('x-axis') 
    axs.set_ylabel('y-axis') 
    axs.set_zlabel('z-axis') 
    return plot