import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_lifia(data_dir="lifiadata/data", nimages=6):
    """Load lifia data: 3D points, edges, and 2d points (with indexed)."""
    filename = '{}/pt_3D'.format(data_dir)
    with open(filename, 'r') as f:
        content = f.readlines()
    content = [line.strip() for line in content]
    npoints = int(content[0])
    points_3d = np.zeros((3, npoints))
    for i in range(npoints):
        line = content[i + 1].split()
        points_3d[:, i] = [float(line[1]),
                           float(line[2]),
                           float(line[3])]

    filename = '{}/edges'.format(data_dir)
    with open(filename) as f:
        content = f.readlines()
    content = [line.strip() for line in content if line.strip() != '']
    nedges = len(content)
    edges = np.zeros((nedges, 2), dtype=int)
    for i in range(nedges):
        line = content[i].split()
        edges[i, :] = [int(line[0]),
                       int(line[1])]

    image_data = []
    for im_number in range(nimages):
        filename = '{}/pt_2D{}'.format(data_dir, im_number + 1)
        with open(filename, 'r') as f:
            content = f.readlines()
        content = [line.strip() for line in content]
        npoints = int(content[0])
        points_2d = np.zeros((2, npoints))
        index = np.zeros(npoints, dtype=int)
        for i in range(npoints):
            line = content[i + 1].split()
            index[i] = int(line[0])
            points_2d[:, i] = [float(line[1]),
                               float(line[2])]

        image_data.append((index, points_2d))

    return points_3d, edges, image_data


def image_points_reorder(image_data):
    '''Return 3x38 matrix (for each image) from affine points+ordering index.'''
    points_2d = []
    for image in image_data:
        points_2d_image = np.zeros((2, 38))
        point_order = np.argsort(image[0])
        for j in range(38):
            points_2d_image[:, j] = image[1][:, point_order[j]]
        points_2d.append(points_2d_image)

    return points_2d


def full_load_lifia(views=(0, 1, 2, 3, 4, 5), plot=False):
    """Load lifia data. Return 3d_points (4x38 matrix) and
    2d_points (list of three 3x38 matrices)."""
    points_3d, edges, image_data = load_lifia()
    points_3d = points_3d[:, :38]  # use only 38 points
    points_3d = np.concatenate((points_3d, np.ones((1, 38))), axis=0)  # add 1 coordinates
    points_2d_a = image_points_reorder(image_data)
    points_2d = [np.concatenate((points_2d_image, np.ones((1, 38))), axis=0)
                 for points_2d_image in points_2d_a]  # add 1 coordinates

    points_2d = [points_2d[v] for v in views]

    if plot:
        plot_points_3d(points_3d, edges)
        plot_points_2d(points_2d[0], edges)

    return points_3d, points_2d, edges


def plot_points_3d(points_3d, edges, points_3d_reconstruct=None, display_basis=None):
    """Plot 3D points with edges."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[0, :], points_3d[1, :], points_3d[2, :], facecolors='none', edgecolors='r')

    for edge in edges:
        x_coords = [points_3d[0, edge[0] - 1], points_3d[0, edge[1] - 1]]
        y_coords = [points_3d[1, edge[0] - 1], points_3d[1, edge[1] - 1]]
        z_coords = [points_3d[2, edge[0] - 1], points_3d[2, edge[1] - 1]]
        ax.plot(x_coords, y_coords, z_coords, 'r', linewidth=1, alpha=0.7)

    if points_3d_reconstruct is not None:
        ax.scatter(points_3d_reconstruct[0, :], points_3d_reconstruct[1, :], points_3d_reconstruct[2, :],
                   facecolors='none', edgecolors='b')

        for edge in edges:
            x_coords = [points_3d_reconstruct[0, edge[0] - 1], points_3d_reconstruct[0, edge[1] - 1]]
            y_coords = [points_3d_reconstruct[1, edge[0] - 1], points_3d_reconstruct[1, edge[1] - 1]]
            z_coords = [points_3d_reconstruct[2, edge[0] - 1], points_3d_reconstruct[2, edge[1] - 1]]
            ax.plot(x_coords, y_coords, z_coords, 'b', linewidth=1, alpha=0.7)

    if display_basis is not None:
        ax.scatter(points_3d[0, display_basis], points_3d[1, display_basis],
                   points_3d[2, display_basis], c='r', s=30)

    ax.axis('off')
    ax.grid(b=None)
    # fig.savefig('recons10.pdf')
    plt.show()


def plot_points_2d(points_2d, edges, display_basis=()):
    """Plot 2D points with edges."""
    plt.figure()
    plt.scatter(points_2d[0, :], points_2d[1, :], c='y')

    for edge in edges:
        x_coords = [points_2d[0, edge[0] - 1], points_2d[0, edge[1] - 1]]
        y_coords = [points_2d[1, edge[0] - 1], points_2d[1, edge[1] - 1]]
        plt.plot(x_coords, y_coords, 'b')

    if len(display_basis) > 0:
        plt.scatter(points_2d[0, display_basis], points_2d[1, display_basis], c='r', s=30)

    plt.show()
