from relative_sfm import *
from lifia import *
import argparse


def main():
    np.random.seed(0)
    parser = argparse.ArgumentParser(description='3D primal reconstruction with LIFIA data.')
    parser.add_argument('-n', '--ntests', type=int, default=50, help='number of quadruples to test (default: 50)')
    parser.add_argument('-v', '--views', type=int, nargs='+', default=[0, 3, 5], help='views for reconstruction (default: [0, 3, 5])')
    args = parser.parse_args()

    points_3d, points_2d, edges = full_load_lifia(args.views)
    best_basis, cameras_final, error_final = find_best_recons(points_2d, args.ntests, print_res=False)
    [P0_final, P1_final, P2_final] = cameras_final
    print("Number of point quadruples: {}".format(args.ntests))
    print("Best quadruple: {}".format(best_basis))
    print("Reprojection error: {:.4f}".format(error_final))
    points_3d_new = triangulate_points_norm(points_2d[0], points_2d[1], points_2d[2],
                                            cameras_final[0], cameras_final[1], cameras_final[2])
    points_3d_new = register_points(points_3d_new, points_3d)
    print("Reconstruction error: {:.2%}".format(reconstruction_error(points_3d_new, points_3d)))
    plot_points_3d(points_3d, edges, points_3d_new)


if __name__ == "__main__":
    main()
