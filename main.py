import copy
import numpy as np
from sklearn.neighbors import KDTree
import open3d as o3d
from timeit import default_timer as timer


def fit_transform(pcd_a, pcd_b):
    centroid_pcd_a = np.mean(pcd_a, axis=0)
    centroid_pcd_b = np.mean(pcd_b, axis=0)
    diff_pcd_a = pcd_a - centroid_pcd_a
    diff_pcd_b = pcd_b - centroid_pcd_b

    to_svd = np.dot(diff_pcd_a.T, diff_pcd_b)
    vec_u, _, vec_t = np.linalg.svd(to_svd)
    rotation_matrix = np.dot(vec_t.T, vec_u.T)

    if np.linalg.det(rotation_matrix) < 0:
        vec_t[pcd_a.shape[1] - 1, :] *= -1
        rotation_matrix = np.dot(vec_t.T, vec_u.T)

    translation = centroid_pcd_b.T - np.dot(rotation_matrix, centroid_pcd_a.T)

    transformation_matrix = np.identity(pcd_a.shape[1] + 1)
    transformation_matrix[:pcd_a.shape[1], :pcd_a.shape[1]] = rotation_matrix
    transformation_matrix[:pcd_a.shape[1], pcd_a.shape[1]] = translation

    return transformation_matrix


def nearest_neighbor_kd_tree(src, dst):
    neigh = KDTree(dst, leaf_size=1)
    distances, indices = neigh.query(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(pcd_a, pcd_b, max_iterations=2000, tolerance=0.00001):
    src = np.ones((pcd_a.shape[1] + 1, pcd_a.shape[0]))
    dst = np.ones((pcd_a.shape[1] + 1, pcd_b.shape[0]))
    src[:pcd_a.shape[1], :] = np.copy(pcd_a.T)
    dst[:pcd_a.shape[1], :] = np.copy(pcd_b.T)

    prev_error = 0
    i = 0
    for i in range(max_iterations):
        distances_array, indices_array = nearest_neighbor_kd_tree(src[:pcd_a.shape[1], :].T, dst[:pcd_a.shape[1], :].T)
        transformation_matrix = fit_transform(src[:pcd_a.shape[1], :].T, dst[:pcd_a.shape[1], indices_array].T)
        src = np.dot(transformation_matrix, src)

        curr_error = np.mean(distances_array)
        if np.abs(prev_error - curr_error) < tolerance:
            break
        prev_error = curr_error

    transformation_matrix = fit_transform(pcd_a, src[:pcd_a.shape[1], :].T)

    return transformation_matrix, i


# https://www.researchgate.net/publication/3974183_The_Trimmed_Iterative_Closest_Point_algorithm
# https://github.com/nyakasko/icp_tricp/blob/main/src/main.cpp
def tricp(pcd_a, pcd_b, n_iterations=2000, tolerance=0.00001):
    src = np.ones((pcd_a.shape[1] + 1, pcd_a.shape[0]))
    dst = np.ones((pcd_a.shape[1] + 1, pcd_b.shape[0]))
    src[:pcd_a.shape[1], :] = np.copy(pcd_a.T)
    dst[:pcd_a.shape[1], :] = np.copy(pcd_b.T)

    prev_error = 0
    i = 0
    for i in range(n_iterations):
        distances, indices = nearest_neighbor_kd_tree(src[:pcd_a.shape[1], :].T, dst[:pcd_a.shape[1], :].T)
        paired_points = [(distances[i], indices[i], i) for i in range(0, len(distances))]
        paired_points = sorted(paired_points)
        iter_number = int(0.4 * len(paired_points))
        trimmed_src = np.zeros((3, iter_number))
        trimmed_dst = np.zeros((3, iter_number))
        for k in range(0, iter_number):
            trimmed_src[0:3, k] = src[0:3, paired_points[k][2]]
            trimmed_dst[0:3, k] = dst[0:3, paired_points[k][1]]

        transformation_matrix = fit_transform(trimmed_src[:, :].T, trimmed_dst[:, :].T)
        src = np.dot(transformation_matrix, src)

        curr_error = np.mean(distances[0:iter_number])
        if np.abs(prev_error - curr_error) < tolerance:
            break
        prev_error = curr_error

    transformation_matrix = fit_transform(pcd_a, src[:pcd_a.shape[1], :].T)

    return transformation_matrix, i


# https://stackoverflow.com/questions/33480618/numpy-matrix-combination
def createTransformationMatrix(translation, angles):
    r = euler_to_rotMat(angles[0], angles[1], angles[2])  # yaw, pitch roll
    return np.vstack((np.hstack((np.array(r), translation[:, None])), [0, 0, 0, 1]))


# http://www.open3d.org/docs/0.12.0/tutorial/pipelines/robust_kernels.html
def apply_noise(pcd, mu, sigma):  # mean and standard deviation
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


# from practice homeworks
def euler_to_rotMat(yaw, pitch, roll):
    yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]])
    pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]])
    roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]])
    return np.dot(yaw, np.dot(pitch, roll))


def rotationMatrixToEulerAngles(t_matrix):
    sy = np.sqrt(t_matrix[0, 0] * t_matrix[0, 0] + t_matrix[1, 0] * t_matrix[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(t_matrix[2, 1], t_matrix[2, 2])
        y = np.arctan2(-t_matrix[2, 0], sy)
        z = np.arctan2(t_matrix[1, 0], t_matrix[0, 0])
    else:
        x = np.arctan2(-t_matrix[1, 2], t_matrix[1, 1])
        y = np.arctan2(-t_matrix[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def doAnalysis(pcd_1, pcd_2, euler_vector, translation_vector, noise_mu, noise_sigma, tolerance,
               visualize=True):
    pcd_a = o3d.io.read_point_cloud(pcd_1)
    pcd_b = o3d.io.read_point_cloud(pcd_2)
    if visualize:
        o3d.io.write_point_cloud('original.ply', (pcd_a + pcd_b))
        o3d.visualization.draw_geometries([pcd_a + pcd_b])
    generated_transformation_matrix = createTransformationMatrix(translation_vector, euler_vector)
    print("Applying:")
    print("Rotation for euler angles:")
    print(euler_vector)
    print("Translation:")
    print(translation_vector)
    print("Transformation Matrix:")
    print(generated_transformation_matrix)
    pcd_a = pcd_a.transform(generated_transformation_matrix)
    if visualize:
        o3d.io.write_point_cloud('transformed.ply', (pcd_a + pcd_b))
        o3d.visualization.draw_geometries([pcd_a + pcd_b])

    pcd_a_noise = apply_noise(pcd_a, noise_mu, noise_sigma)
    pcd_b_noise = apply_noise(pcd_b, noise_mu, noise_sigma)

    if visualize:
        o3d.io.write_point_cloud('noised.ply', (pcd_a_noise + pcd_b_noise))
        o3d.visualization.draw_geometries([pcd_a_noise + pcd_b_noise])

    np_pcd_a = np.asarray(pcd_a_noise.points)
    np_pcd_b = np.asarray(pcd_b_noise.points)

    print('\n')
    print('\n')
    print("[TR ICP] B -> A")
    start = timer()
    transformation_matrix_trimmed, iteration_number = tricp(np_pcd_b, np_pcd_a, tolerance=tolerance)
    end = timer()
    print("[TR ICP] Time elapsed: ", end - start)
    print("[TR ICP] Angle error: ", euler_vector - rotationMatrixToEulerAngles(transformation_matrix_trimmed))
    print("[TR ICP] Translation error: ", translation_vector - transformation_matrix_trimmed[0:3, 3])
    print("[TR ICP] Transformation matrix:")
    print(transformation_matrix_trimmed)
    print("[TR ICP] Tolerance: ", tolerance)
    print("[TR ICP] Convergence at iteration: ", iteration_number)
    if visualize:
        pcd_b_tricp = copy.deepcopy(pcd_b).transform(transformation_matrix_trimmed)
        o3d.io.write_point_cloud('tricp.ply', (pcd_a + pcd_b_tricp))
        o3d.visualization.draw_geometries([pcd_a + pcd_b_tricp])
    print('\n')
    print('\n')
    print("[ICP] B -> A")
    start = timer()
    transformation_matrix, iteration_number = icp(np_pcd_b, np_pcd_a, tolerance=tolerance)
    end = timer()
    print("[ICP] Time elapsed: ", end - start)
    print("[ICP] Angle error: ", euler_vector - rotationMatrixToEulerAngles(transformation_matrix))
    print("[ICP] Translation error: ", translation_vector - transformation_matrix[0:3, 3])
    print("[ICP] Transformation matrix:")
    print(transformation_matrix)
    print("[ICP] Tolerance: ", tolerance)
    print("[ICP] Convergence at iteration: ", iteration_number)
    if visualize:
        pcd_b_icp = copy.deepcopy(pcd_b).transform(transformation_matrix)
        o3d.io.write_point_cloud('icp.ply', (pcd_a + pcd_b_icp))
        o3d.visualization.draw_geometries([pcd_a + pcd_b_icp])
    print('\n')
    print('\n')
    print("Difference between ICP and TRICP: ")
    print("Difference in transformation matrix: ", transformation_matrix_trimmed - transformation_matrix)
    print("Difference in euler angles: ", rotationMatrixToEulerAngles(transformation_matrix_trimmed) -
          rotationMatrixToEulerAngles(transformation_matrix))
    print("Difference in translation: ", transformation_matrix_trimmed[0:3, 3] - transformation_matrix[0:3, 3])
    print('\n')
    print('\n')
    print('\n')
    print('\n')


# doAnalysis('fountain_a.ply', 'fountain_b.ply',
#            np.array([0, 0, 0]),
#            np.array([0.6, 0.3, 0]),
#            0.02,
#            0.01,
#            0.0000000000001)

# list_of_files = ['1.ply', '2.ply', '3.ply', '4.ply', '5.ply', '6.ply']

def reconstruct(list_of_files):
    base_pcd = o3d.io.read_point_cloud(list_of_files[0])
    for idx in range(1, len(list_of_files)):
        print(idx, len(list_of_files))
        curr_pcd = o3d.io.read_point_cloud(list_of_files[idx])
        np_pcd_a = np.asarray(base_pcd.points)
        np_pcd_b = np.asarray(curr_pcd.points)
        transformation_matrix, _ = tricp(np_pcd_a, np_pcd_b, tolerance=0.0001)
        base_pcd = curr_pcd + base_pcd.transform(transformation_matrix)
        if idx == 5:
            o3d.visualization.draw_geometries([base_pcd])
    o3d.visualization.draw_geometries([base_pcd])


# reconstruct(list_of_files)
