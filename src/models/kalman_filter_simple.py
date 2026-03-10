"""
Simple Kalman Filter implementation without external dependencies.
Replaces filterpy.kalman.KalmanFilter for tracking 3D bounding boxes.
"""

import numpy as np


class KalmanFilter:
    """
    Simple Kalman Filter for linear systems.

    Attributes:
        dim_x: int - Dimension of state vector
        dim_z: int - Dimension of measurement vector
        x: ndarray - State estimate [dim_x, 1]
        P: ndarray - State covariance matrix [dim_x, dim_x]
        F: ndarray - State transition matrix [dim_x, dim_x]
        H: ndarray - Measurement function [dim_z, dim_x]
        R: ndarray - Measurement noise covariance [dim_z, dim_z]
        Q: ndarray - Process noise covariance [dim_x, dim_x]
    """

    def __init__(self, dim_x, dim_z):
        """
        Initialize Kalman Filter.

        Args:
            dim_x: int - Dimension of state vector
            dim_z: int - Dimension of measurement vector
        """
        self.dim_x = dim_x
        self.dim_z = dim_z

        # State estimate
        self.x = np.zeros((dim_x, 1))

        # State covariance matrix
        self.P = np.eye(dim_x)

        # State transition matrix (identity by default)
        self.F = np.eye(dim_x)

        # Measurement function (identity by default)
        self.H = np.eye(dim_z, dim_x)

        # Measurement noise covariance
        self.R = np.eye(dim_z)

        # Process noise covariance
        self.Q = np.eye(dim_x)

    def predict(self):
        """
        Predict next state.

        Updates:
            x = F @ x
            P = F @ P @ F.T + Q
        """
        # Predict state
        self.x = self.F @ self.x

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        Update state estimate with measurement.

        Args:
            z: ndarray [dim_z] or [dim_z, 1] - Measurement vector
        """
        # Ensure z is column vector
        if z.ndim == 1:
            z = z.reshape((-1, 1))

        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state estimate
        self.x = self.x + K @ y

        # Update covariance estimate (Joseph form for numerical stability)
        I_KH = np.eye(self.dim_x) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
