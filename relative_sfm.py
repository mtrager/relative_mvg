"""
Quasi-linear algorithm for SFM.
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def canonical_transform_2d(P):
    """Return 3x3 transformation that turns columns of P (that is 3x4) into basis."""
    U = P[:, :3]
    U_inv = np.linalg.pinv(U)
    k = U_inv.dot(P[:, 3])
    T = np.array([U_inv[0] / k[0], U_inv[1] / k[1], U_inv[2] / k[2]])
    return T  # / T.mean()  # normalize in some way?


def transform_points(points_2d, basis=(0, 1, 2, 3)):
    """Transform all points so specified points are basis."""
    nimages = len(points_2d)
    new_points_2d = []
    t_list = []
    for i in range(nimages):
        P = points_2d[i][:, basis]
        T = canonical_transform_2d(P)
        new_image_points = T.dot(points_2d[i])
        # new_image_points = new_image_points / new_image_points.mean()
        new_points_2d.append(new_image_points)
        t_list.append(T)
    return new_points_2d, t_list


def trilinearities_equations(point_triple, matrix):
    """Fill 4x12 matrix with coefficients of trilinearities"""
    u1 = point_triple[0][0]
    u2 = point_triple[0][1]
    u3 = point_triple[0][2]
    v1 = point_triple[1][0]
    v2 = point_triple[1][1]
    v3 = point_triple[1][2]
    t1 = point_triple[2][0]
    t2 = point_triple[2][1]
    t3 = point_triple[2][2]
    uu1 = u3 - u2
    uu2 = u1 - u3
    uu3 = u2 - u1
    vv1 = v3 - v2
    vv2 = v1 - v3
    vv3 = v2 - v1
    tt1 = t3 - t2
    tt2 = t1 - t3
    tt3 = t2 - t1
    matrix += np.array([[0, 0, 0, 0, v3 * t2 * uu1, -u2 * v3 * tt1, 0, -v2 * t3 * uu1, u3 * v2 * tt1, 0, u2 * t3 * vv1, -u3 * t2 * vv1],
                        [0, -v3 * t1 * uu2, u1 * v3 * tt2, 0, 0, 0, v1 * t3 * uu2, 0, -u3 * v1 * tt2, -u1 * t3 * vv2, 0, u3 * t1 * vv2],
                        [v2 * t1 * uu3, 0, -u1 * v2 * tt3, -v1 * t2 * uu3, 0, u2 * v1 * tt3, 0, 0, 0, u1 * t2 * vv3, -u2 * t1 * vv3, 0],
                        [-uu3 * vv1 * tt2, uu2 * vv1 * tt3, 0, uu3 * vv2 * tt1, -uu1 * vv2 * tt3, 0, -uu2 * vv3 * tt1, uu1 * vv3 * tt2, 0, 0, 0, 0]])


def trilinearities_system(points_2d_triples):
    """Return 4npoints x 12 matrix describing the linear equations for e."""
    npoints = (points_2d_triples[0]).shape[1]
    mat = np.zeros((4 * npoints, 12))
    for i in range(npoints):
        point_triple = (points_2d_triples[0][:, i], points_2d_triples[1][:, i], points_2d_triples[2][:, i])
        trilinearities_equations(point_triple, mat[4 * i:4 * (i + 1), :])
    return mat


def trilinearities_system_dual(points_2d, x_ref):
    """Return 4npoints x 12 matrix describing the linear equations for e."""

    nviews = len(points_2d)
    mat = np.zeros((4 * nviews, 12))
    for i in range(nviews):
        point_triple = (points_2d[i][:, x_ref[0]], points_2d[i][:, x_ref[1]], points_2d[i][:, x_ref[2]])
        trilinearities_equations(point_triple, mat[4 * i:4 * (i + 1), :])
    return mat


def camera_centers_from_e_vector(e):
    """Solve over-determined linear system for c',c'' given the vector e."""
    E1 = np.array([[e[4] - e[5], -e[1] + e[2], 0, 0],
                   [e[7] - e[8], 0, -e[0] + e[2], 0],
                   [e[10] - e[11], 0, 0, -e[0] + e[1]],
                   [0, e[6] - e[8], -e[3] + e[5], 0],
                   [0, -e[11] + e[9], 0, -e[3] + e[4]],
                   [0, 0, -e[10] + e[9], -e[6] + e[7]]])
    E2 = np.array([[-e[10] + e[7], -e[6] + e[9], 0, 0],
                   [-e[11] + e[4], 0, -e[3] + e[9], 0],
                   [e[5] - e[8], 0, 0, -e[3] + e[6]],
                   [0, e[1] - e[11], -e[0] + e[10], 0],
                   [0, e[2] - e[8], 0, -e[0] + e[7]],
                   [0, 0, e[2] - e[5], -e[1] + e[4]]])
    U, S, V = np.linalg.svd(E1)
    cc1 = V[-1, :]
    U, S, V = np.linalg.svd(E2)
    cc2 = V[-1, :]
    c1 = 1 / cc1
    c1 = c1 / c1[3]
    c2 = 1 / cc2
    c2 = c2 / c2[3]
    return c1, c2


def points_from_e_vector(e):
    """Solve over-determined linear system for c',c'' given the vector e."""
    E1 = np.array([[e[4] - e[5], -e[1] + e[2], 0, 0],
                   [e[7] - e[8], 0, -e[0] + e[2], 0],
                   [e[10] - e[11], 0, 0, -e[0] + e[1]],
                   [0, e[6] - e[8], -e[3] + e[5], 0],
                   [0, -e[11] + e[9], 0, -e[3] + e[4]],
                   [0, 0, -e[10] + e[9], -e[6] + e[7]]])
    E2 = np.array([[-e[10] + e[7], -e[6] + e[9], 0, 0],
                   [-e[11] + e[4], 0, -e[3] + e[9], 0],
                   [e[5] - e[8], 0, 0, -e[3] + e[6]],
                   [0, e[1] - e[11], -e[0] + e[10], 0],
                   [0, e[2] - e[8], 0, -e[0] + e[7]],
                   [0, 0, e[2] - e[5], -e[1] + e[4]]])
    U, S, V = np.linalg.svd(E1)
    x1 = V[-1, :]
    U, S, V = np.linalg.svd(E2)
    x2 = V[-1, :]
    x1 = x1 / x1[3]  # maybe not necessary to normalize
    x2 = x2 / x2[3]
    return x1, x2


def reduced_camera_from_center(c):
    """Return 3x4 reduced projection matrix with center c."""
    return np.array([[1 / c[0], 0, 0, -1 / c[3]], [0, 1 / c[1], 0, -1 / c[3]], [0, 0, 1 / c[2], -1 / c[3]]])


def camera_projection(P, X):
    u = P.dot(X)
    return u / u[2]


def camera_center(P):
    U, S, V = np.linalg.svd(P)
    c = V[-1, :]
    return c


def hartley_norm_2d(points_2d):
    """Hartley normalization of points_2d (a list of 3xnpoints matrices).
    newpoints = T*oldpoints"""
    points_2d_normalized = []
    t_list = []
    for U in points_2d:
        npoints = U.shape[1]
        U = U[:2, :]
        mean = U.mean(axis=1).reshape([-1, 1])
        U1 = (U - (mean.repeat(npoints, axis=1)))
        scale = np.mean(np.sqrt(np.sum(U1**2, axis=1)))
        scale = np.sqrt(2) / scale
        U1 = scale * U1
        U1 = np.concatenate((U1, np.ones((1, npoints))))
        T = np.array([[scale, 0, -scale * mean[0, 0]],
                      [0, scale, -scale * mean[1, 0]],
                      [0, 0, 1]])
        points_2d_normalized.append(U1)
        t_list.append(T)
    return points_2d_normalized, t_list


def triangulate_point(u0, u1, u2, P0, P1, P2):
    """Triangulate one triple of image points views using least squares (return 4-vector of homogeneous coords)."""
    M = np.zeros((9, 7))
    M[:3, :4] = P0
    M[3:6, :4] = P1
    M[6:9, :4] = P2
    M[:3, 4] = u0
    M[3:6, 5] = u1
    M[6:9, 6] = u2
    U, S, V = np.linalg.svd(M)
    X = V[-1, :4]
    return X


def triangulate_points(points_2d_0, points_2d_1, points_2d_2, P0, P1, P2):
    """Triangulate multiple triples of image poitns using least squares
    (return 4-vector of homogeneous coords)."""
    npoints = np.min((points_2d_0.shape[1], points_2d_1.shape[1], points_2d_2.shape[1]))
    points_3d = np.zeros((4, npoints))
    for i in range(npoints):
        X = triangulate_point(points_2d_0[:, i], points_2d_1[:, i], points_2d_2[:, i], P0, P1, P2)
        points_3d[:, i] = X.T
    return points_3d


def triangulate_points_norm(points_2d_0, points_2d_1, points_2d_2, P0, P1, P2):
    """Triangulate multiple triples of image poitns using least squares
    (return 4-vector of homogeneous coords)."""
    npoints = np.min((points_2d_0.shape[1], points_2d_1.shape[1], points_2d_2.shape[1]))
    points_3d = np.zeros((4, npoints))
    points_2d_normalized, t_list = hartley_norm_2d((points_2d_0, points_2d_1, points_2d_2))
    points_2d_normalized_0 = points_2d_normalized[0]
    points_2d_normalized_1 = points_2d_normalized[1]
    points_2d_normalized_2 = points_2d_normalized[2]
    P0_normalized = (t_list[0]).dot(P0)
    P1_normalized = (t_list[1]).dot(P1)
    P2_normalized = (t_list[2]).dot(P2)
    for i in range(npoints):
        X = triangulate_point(points_2d_normalized_0[:, i],
                              points_2d_normalized_1[:, i], points_2d_normalized_2[:, i],
                              P0_normalized, P1_normalized, P2_normalized)
        points_3d[:, i] = X.T
    return points_3d


def register_points(points_3d_0, points_3d_1):
    """Linear projective registration."""
    npoints = points_3d_0.shape[1]
    mat = np.zeros((6 * npoints, 16))
    for i in range(npoints):
        x0, x1, x2, x3 = points_3d_0[:, i]
        y0, y1, y2, y3 = points_3d_1[:, i]
        mat[6 * i:6 * (i + 1), :] = np.array([
            [x0 * y1, x1 * y1, x2 * y1, x3 * y1, -x0 * y0, -x1 * y0, -x2 * y0, -x3 * y0, 0, 0, 0, 0, 0, 0, 0, 0],
            [x0 * y2, x1 * y2, x2 * y2, x3 * y2, 0, 0, 0, 0, -x0 * y0, -x1 * y0, -x2 * y0, -x3 * y0, 0, 0, 0, 0],
            [x0 * y3, x1 * y3, x2 * y3, x3 * y3, 0, 0, 0, 0, 0, 0, 0, 0, -x0 * y0, -x1 * y0, -x2 * y0, -x3 * y0],
            [0, 0, 0, 0, x0 * y2, x1 * y2, x2 * y2, x3 * y2, -x0 * y1, -x1 * y1, -x2 * y1, -x3 * y1, 0, 0, 0, 0],
            [0, 0, 0, 0, x0 * y3, x1 * y3, x2 * y3, x3 * y3, 0, 0, 0, 0, -x0 * y1, -x1 * y1, -x2 * y1, -x3 * y1],
            [0, 0, 0, 0, 0, 0, 0, 0, x0 * y3, x1 * y3, x2 * y3, x3 * y3, -x0 * y2, -x1 * y2, -x2 * y2, -x3 * y2]])
    u, s, v = np.linalg.svd(mat)
    q_flat = v[-1, :]
    q = q_flat.reshape((4, 4))
    points_3d_new = q.dot(points_3d_0)
    return points_3d_new / points_3d_new[3, :]


def projection_errors(points_3d, points_2d, cameras):
    """Compute squared distance of projected points to image points."""
    nimages = len(cameras)
    errors_vectors = []
    for i in range(nimages):
        points_2d_projected = camera_projection(cameras[i], points_3d)
        diff = points_2d_projected - points_2d[i]
        errors_image = np.sum(diff**2, axis=0)
        errors_vectors.append(errors_image)
    return errors_vectors


def reprojection_errors(points_2d, cameras):
    """Compute squared distance of projection of triangulated points to image points."""

    # points_3d = triangulate_points(points_2d[0], points_2d[1], points_2d[2],
    #                                cameras[0], cameras[1], cameras[2])
    points_3d = triangulate_points_norm(points_2d[0], points_2d[1], points_2d[2],
                                        cameras[0], cameras[1], cameras[2])
    return projection_errors(points_3d, points_2d, cameras)


def mean_reprojection_error(points_2d, cameras):
    """Return mean *squared* reprojection error."""
    errors = reprojection_errors(points_2d, cameras)
    return np.mean(np.concatenate(errors))


def reconstruction_error(points_3d_r, points_3d_true):
    """Mean reconstruction error, relative to scene radius."""
    npoints = points_3d_true.shape[1]
    center = points_3d_true.mean(axis=1).reshape([-1, 1])
    radius = np.sqrt(((points_3d_true - center.repeat(npoints, axis=1))**2).sum(axis=0).max())
    mean_error = np.sqrt(np.mean(((points_3d_r - points_3d_true)**2).sum(axis=0)))
    return mean_error / radius


def camera_resection(X, U):
    """Camera resectioning. X are 3D points and U are 2D points (homogeneous coordinates)."""
    npoints = X.shape[1]
    mat = np.zeros((3 * npoints, 12))
    for i in range(npoints):
        x0, x1, x2, x3 = X[:, i]
        u0, u1, u2 = U[:, i]
        mat[3 * i:3 * (i + 1), :] = np.array(
            [[-x0 * u1, -x1 * u1, -x2 * u1, -x3 * u1, x0 * u0, x1 * u0, x2 * u0, x3 * u0, 0, 0, 0, 0],
             [-x0 * u2, -x1 * u2, -x2 * u2, -x3 * u2, 0, 0, 0, 0, x0 * u0, x1 * u0, x2 * u0, x3 * u0],
             [0, 0, 0, 0, -x0 * u2, -x1 * u2, -x2 * u2, -x3 * u2, x0 * u1, x1 * u1, x2 * u1, x3 * u1]])

    u, s, v = np.linalg.svd(mat)
    P_flat = v[-1, :]
    P = P_flat.reshape((3, 4))
    return P


def reconstruct_cameras(points_2d, basis):
    """Use relative reconstruction method with respect to the image points specified by basis."""
    new_points, t_list = transform_points(points_2d, basis)
    mat = trilinearities_system(new_points)
    U, S, V = np.linalg.svd(mat)
    e = V[-2, :]
    c1, c2 = camera_centers_from_e_vector(e)
    c0 = np.ones(4)
    P0 = reduced_camera_from_center(c0)
    P1 = reduced_camera_from_center(c1)
    P2 = reduced_camera_from_center(c2)
    # use original coordinates
    P0 = np.linalg.inv(t_list[0]).dot(P0)
    P1 = np.linalg.inv(t_list[1]).dot(P1)
    P2 = np.linalg.inv(t_list[2]).dot(P2)
    return P0, P1, P2


def reconstruct_cameras_dual(points_2d, ref_points):
    """Use dual relative reconstruction method with respect to the image points specified by basis."""
    nviews = len(points_2d)
    new_points, t_list = transform_points(points_2d, ref_points[:4])
    mat = trilinearities_system_dual(new_points, ref_points[4:])
    U, S, V = np.linalg.svd(mat)
    e = V[-2, :]
    x1, x2 = points_from_e_vector(e)
    x0 = np.ones(4).reshape([-1, 1])
    x1 = x1.reshape([-1, 1])
    x2 = x2.reshape([-1, 1])
    X = np.concatenate((np.identity(4), x0, x1, x2), axis=1)
    cameras = []
    for i in range(nviews):
        u4 = np.ones(3).reshape([-1, 1])
        v0 = new_points[i][:, ref_points[4]].reshape([-1, 1])
        v1 = new_points[i][:, ref_points[5]].reshape([-1, 1])
        v2 = new_points[i][:, ref_points[6]].reshape([-1, 1])
        U = np.concatenate((np.identity(3), u4, v0, v1, v2), axis=1)
        new_camera = camera_resection(X, U)
        new_camera = np.linalg.inv(t_list[i]).dot(new_camera)  # use original coordinates
        cameras.append(new_camera)
    return cameras


def find_best_recons(points_2d, ntests, print_res=False):
    """Find best quadruple for relative reconstruction method."""
    npoints = points_2d[0].shape[1]
    min_error = np.inf
    best_basis = [0, 1, 2, 3]
    best_cam = []
    for i in range(ntests):
        basis = np.random.choice(npoints, 4, replace=False)
        P0, P1, P2 = reconstruct_cameras(points_2d, basis)
        error = mean_reprojection_error(points_2d, (P0, P1, P2))
        if print_res:
            print(i + 1, basis, "error:", np.sqrt(error))
        if error < min_error:
            best_basis = basis
            min_error = error
            best_cam = [P0, P1, P2]
    return best_basis, best_cam, np.sqrt(min_error)


def find_best_recons_dual(points_2d, ntests, print_res=False):
    """Find best seventuple for dual relative reconstruction method."""
    npoints = points_2d[0].shape[1]
    min_error = np.inf
    best_ref = list(range(7))
    best_cam = []
    for i in range(ntests):
        ref_points = np.random.choice(list(range(38)), 7, replace=False)  # first four are basis
        cameras = reconstruct_cameras_dual(points_2d, ref_points)
        error = mean_reprojection_error(points_2d, cameras)
        if print_res:
            print(i + 1, ref_points, "error:", np.sqrt(error))
        if error < min_error:
            best_ref = ref_points
            min_error = error
            best_cam = cameras
    return best_ref, best_cam, np.sqrt(min_error)


def find_best_recons_partial_data(points_2d, points_2d_all, ntests, print_res=False):
    npoints = points_2d[0].shape[1]
    min_error = np.inf
    best_basis = [0, 1, 2, 3]
    best_cam = []
    for i in range(ntests):
        basis = np.random.choice(npoints, 4, replace=False)
        P0, P1, P2 = reconstruct_cameras(points_2d, basis)
        error = mean_reprojection_error(points_2d_all, (P0, P1, P2))
        if print_res:
            print(i + 1, basis, "error:", np.sqrt(error))
        if error < min_error:
            best_basis = basis
            min_error = error
            best_cam = [P0, P1, P2]
    return best_basis, best_cam, np.sqrt(min_error)
