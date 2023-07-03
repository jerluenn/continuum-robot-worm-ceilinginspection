import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import numpy as np 
import pandas as pd 
from matplotlib.transforms import Affine2D
import mpl_toolkits.mplot3d.art3d as art3d


def main(csv_name): 

    df = pd.read_csv(csv_name)
    data_plot = df.to_numpy()
    data_plot = data_plot[:, 1:]

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(data_plot[0, :], data_plot[1, :], data_plot[2, :])

    # Plot out discs



    # Plot out cables 
    # Might need to put this in the other file and export it from there.

    # ax.legend()
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(0.4, 0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    p = Circle((0, 0), 0.025)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir=[1, 0, 0])

    plt.show()

    return 0 

main("data/test_plot.csv")


