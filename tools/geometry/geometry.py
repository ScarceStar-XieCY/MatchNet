import numpy as np

from .ransac import RansacEstimator


class Procrustes:
    """Orthogonal Procrustes problem [1].

    References:
        [1]: https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    """

    def __init__(self, transform=None):
        self._transform = transform

    def __call__(self, xyz):
        return Procrustes.transform_xyz(xyz, self._transform)

    @classmethod
    def transform_xyz(cls, xyz, transform):
        """Applies a rigid transform to an (N, 3) point cloud.
        """
        xyz_h = np.hstack([xyz, np.ones((len(xyz), 1))])  # homogenize 3D pointcloud
        xyz_t_h = (transform @ xyz_h.T).T  # apply transform
        return xyz_t_h[:, :3]

    def estimate(self, X, Y):
        # find centroids
        X_c = np.mean(X, axis=0)
        Y_c = np.mean(Y, axis=0)

        # shift
        X_s = X - X_c
        Y_s = Y - Y_c

        # compute SVD of covariance matrix
        cov = Y_s.T @ X_s
        u, _, vt = np.linalg.svd(cov)

        # determine rotation
        rot = u @ vt
        if np.linalg.det(rot) < 0.0:
            vt[2, :] *= -1
            rot = u @ vt

        # determine optimal translation
        trans = Y_c - rot @ X_c

        if self._transform is None:
            self._transform = np.eye(4)
        self._transform[:3, :3] = rot
        self._transform[:3, 3] = trans

    def residuals(self, X, Y):
        Y_est = self(X)  # apply estimated rigid transform to dest
        return np.linalg.norm(Y_est - Y, axis=1)

    @property
    def params(self):
        return self._transform


def estimate_rigid_transform(X, Y, use_ransac=True):
    """Determine the best rigid transform between two point clouds.

    Args:
        X, Y (ndarray): Source and target point clouds of shape
            (N, 3).
        use_ransac (bool): Whether to use RANSAC. Makes it slower but
            more robust to outliers.

    Returns:
        transform (ndarray): The estimated 4x4 rigid transform.
        mse (float): The mean squared error between the transformed
            source point cloud X according to the estimated transform
            and the target point cloud Y.
  """
    model = Procrustes()
    if use_ransac:
        ransac = RansacEstimator(min_samples=3, residual_threshold=0.001, max_trials=1000,)
        ret = ransac.fit(model, [X, Y])
        transform = ret["best_params"]
        mse = ret["best_residual"]
    else:
        model.estimate(X, Y)
        mse = (model.residuals(X, Y) ** 2).mean()
        transform = model.params
    return transform, mse
