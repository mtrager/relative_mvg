from relative_sfm import *
from lifia import *
import argparse


def main():
    np.random.seed(0)
    parser = argparse.ArgumentParser(description='3D dual reconstruction with LIFIA data.')
    parser.add_argument('-n', '--ntests', type=int, default=50, help='number of points seventuples to test (default: 50)')
    args = parser.parse_args()

    points_3d, points_2d, edges = full_load_lifia()
    best_ref, cameras_final, error_final = find_best_recons_dual(points_2d, args.ntests, print_res=False)
    print("Number of point seventuples: {}".format(args.ntests))
    print("Best reference: {}".format(best_ref))
    print("Reprojection error: {:.4f}".format(error_final))
    points_3d_new = triangulate_points_norm(points_2d[0], points_2d[2], points_2d[4],
                                            cameras_final[0], cameras_final[2], cameras_final[4])
    points_3d_new = register_points(points_3d_new, points_3d)
    print("Reconstruction error: {:.2%}".format(reconstruction_error(points_3d_new, points_3d)))
    plot_points_3d(points_3d, edges, points_3d_new)


if __name__ == "__main__":
    main()
