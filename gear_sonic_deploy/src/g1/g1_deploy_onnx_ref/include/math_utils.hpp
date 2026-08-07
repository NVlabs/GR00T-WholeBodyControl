/**
 * @file math_utils.hpp
 * @brief Quaternion, rotation, and vector math utilities used throughout the
 *        G1 deployment stack.
 *
 * All quaternion functions use the **wxyz** (scalar-first) convention:
 *   q = [w, x, y, z]
 *
 * Most functions come in both float and double versions.  The double versions
 * are suffixed with `_d` (e.g. `quat_mul_d`, `quat_rotate_d`).  Float
 * versions are kept for compatibility with the robot SDK interface which
 * uses 32-bit floats.
 *
 * Key function groups:
 *  - **Conversion**: float_to_double / double_to_float, wxyz ↔ xyzw.
 *  - **Gravity**: GetGravityOrientation – project gravity into body frame.
 *  - **Rotation**: quat_rotate – rotate a 3D vector by a quaternion.
 *  - **Quaternion algebra**: quat_mul, quat_conjugate, quat_unit, quat_slerp.
 *  - **Heading**: calc_heading, calc_heading_quat, calc_heading_quat_inv,
 *    euler_z_to_quat – extract / construct yaw-only quaternions.
 *  - **Axis-angle**: quat_from_angle_axis, quat_to_angle_axis.
 *  - **Matrix ↔ quat**: quat_to_rotation_matrix, rotation_matrix_to_quat.
 *  - **Vector**: normalize_vector.
 */

#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <array>
#include <cmath>
#include <algorithm>

// ============================================================================
// CONVERSION UTILITIES BETWEEN FLOAT AND DOUBLE
// ============================================================================

// Convert float array to double array
template <size_t N> inline std::array<double, N> float_to_double(const std::array<float, N>& f_arr) {
  std::array<double, N> d_arr;
  for (size_t i = 0; i < N; i++) { d_arr[i] = static_cast<double>(f_arr[i]); }
  return d_arr;
}

// Convert double array to float array
template <size_t N> inline std::array<float, N> double_to_float(const std::array<double, N>& d_arr) {
  std::array<float, N> f_arr;
  for (size_t i = 0; i < N; i++) { f_arr[i] = static_cast<float>(d_arr[i]); }
  return f_arr;
}

// ============================================================================
// LEGACY FLOAT FUNCTIONS (for compatibility with robot interface)
// ============================================================================

// Quaternion format conversion utilities (float versions)
inline std::array<float, 4> wxyz_to_xyzw(const std::array<float, 4>& quaternion) {
  return {quaternion[1], quaternion[2], quaternion[3], quaternion[0]}; // qx, qy, qz, qw
}

inline std::array<float, 4> xyzw_to_wxyz(const std::array<float, 4>& quaternion) {
  return {quaternion[3], quaternion[0], quaternion[1], quaternion[2]}; // qw, qx, qy, qz
}

// ============================================================================
// HIGH PRECISION DOUBLE FUNCTIONS (for internal computations)
// ============================================================================

// Quaternion format conversion utilities (double versions)
inline std::array<double, 4> wxyz_to_xyzw_d(const std::array<double, 4>& quaternion) {
  return {quaternion[1], quaternion[2], quaternion[3], quaternion[0]}; // qx, qy, qz, qw
}

inline std::array<double, 4> xyzw_to_wxyz_d(const std::array<double, 4>& quaternion) {
  return {quaternion[3], quaternion[0], quaternion[1], quaternion[2]}; // qw, qx, qy, qz
}

// Calculate gravity orientation from quaternion (float version)
inline std::array<float, 3> GetGravityOrientation(const std::array<float, 4>& quaternion) {
  float qw = quaternion[0];
  float qx = quaternion[1];
  float qy = quaternion[2];
  float qz = quaternion[3];

  std::array<float, 3> gravity_orientation = {0.0f, 0.0f, 0.0f};

  gravity_orientation[0] = 2.0f * (-qz * qx + qw * qy);
  gravity_orientation[1] = -2.0f * (qz * qy + qw * qx);
  gravity_orientation[2] = 1.0f - 2.0f * (qw * qw + qz * qz);

  return gravity_orientation;
}

// Calculate gravity orientation from quaternion (double version)
inline std::array<double, 3> GetGravityOrientation_d(const std::array<double, 4>& quaternion) {
  double qw = quaternion[0];
  double qx = quaternion[1];
  double qy = quaternion[2];
  double qz = quaternion[3];

  std::array<double, 3> gravity_orientation = {0.0, 0.0, 0.0};

  gravity_orientation[0] = 2.0 * (-qz * qx + qw * qy);
  gravity_orientation[1] = -2.0 * (qz * qy + qw * qx);
  gravity_orientation[2] = 1.0 - 2.0 * (qw * qw + qz * qz);

  return gravity_orientation;
}

// Rotate a 3D vector by a quaternion (quaternion in wxyz format) - float version
template<typename T>
inline std::array<T, 3> quat_rotate(const std::array<T, 4>& q, const std::array<T, 3>& v) {
  T q_w = q[0];
  std::array<T, 3> q_vec = {q[1], q[2], q[3]};

  // a = v * (2.0 * q_w^2 - 1.0)
  T scale_a = 2 * q_w * q_w - 1;
  std::array<T, 3> a = {v[0] * scale_a, v[1] * scale_a, v[2] * scale_a};

  // b = cross(q_vec, v) * q_w * 2.0
  std::array<T, 3> cross_qv = {q_vec[1] * v[2] - q_vec[2] * v[1], q_vec[2] * v[0] - q_vec[0] * v[2],
                                   q_vec[0] * v[1] - q_vec[1] * v[0]};
  std::array<T, 3> b = {cross_qv[0] * q_w * 2, cross_qv[1] * q_w * 2, cross_qv[2] * q_w * 2};

  // c = q_vec * dot(q_vec, v) * 2.0
  T dot_qv = q_vec[0] * v[0] + q_vec[1] * v[1] + q_vec[2] * v[2];
  std::array<T, 3> c = {q_vec[0] * dot_qv * 2, q_vec[1] * dot_qv * 2, q_vec[2] * dot_qv * 2};

  return {a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]};
}

// Rotate a 3D vector by a quaternion (quaternion in wxyz format) - double version
inline std::array<double, 3> quat_rotate_d(const std::array<double, 4>& q, const std::array<double, 3>& v) {
  double q_w = q[0];
  std::array<double, 3> q_vec = {q[1], q[2], q[3]};

  // a = v * (2.0 * q_w^2 - 1.0)
  double scale_a = 2.0 * q_w * q_w - 1.0;
  std::array<double, 3> a = {v[0] * scale_a, v[1] * scale_a, v[2] * scale_a};

  // b = cross(q_vec, v) * q_w * 2.0
  std::array<double, 3> cross_qv = {q_vec[1] * v[2] - q_vec[2] * v[1], q_vec[2] * v[0] - q_vec[0] * v[2],
                                    q_vec[0] * v[1] - q_vec[1] * v[0]};
  std::array<double, 3> b = {cross_qv[0] * q_w * 2.0, cross_qv[1] * q_w * 2.0, cross_qv[2] * q_w * 2.0};

  // c = q_vec * dot(q_vec, v) * 2.0
  double dot_qv = q_vec[0] * v[0] + q_vec[1] * v[1] + q_vec[2] * v[2];
  std::array<double, 3> c = {q_vec[0] * dot_qv * 2.0, q_vec[1] * dot_qv * 2.0, q_vec[2] * dot_qv * 2.0};

  return {a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]};
}

// Quaternion multiplication (wxyz format: w, x, y, z) - float version
template<typename T>
inline std::array<T, 4> quat_mul(const std::array<T, 4>& a, const std::array<T, 4>& b) {
  T w1 = a[0], x1 = a[1], y1 = a[2], z1 = a[3];
  T w2 = b[0], x2 = b[1], y2 = b[2], z2 = b[3];

  // Python optimized implementation
  T ww = (z1 + x1) * (x2 + y2);
  T yy = (w1 - y1) * (w2 + z2);
  T zz = (w1 + y1) * (w2 - z2);
  T xx = ww + yy + zz;
  T qq = (xx + (z1 - x1) * (x2 - y2)) / 2;

  T w = qq - ww + (z1 - y1) * (y2 - z2);
  T x = qq - xx + (x1 + w1) * (x2 + w2);
  T y = qq - yy + (w1 - x1) * (y2 + z2);
  T z = qq - zz + (z1 + y1) * (w2 - x2);

  return {w, x, y, z};
}

// Quaternion multiplication (wxyz format: w, x, y, z) - double version
inline std::array<double, 4> quat_mul_d(const std::array<double, 4>& a, const std::array<double, 4>& b) {
  double w1 = a[0], x1 = a[1], y1 = a[2], z1 = a[3];
  double w2 = b[0], x2 = b[1], y2 = b[2], z2 = b[3];

  // Python optimized implementation
  double ww = (z1 + x1) * (x2 + y2);
  double yy = (w1 - y1) * (w2 + z2);
  double zz = (w1 + y1) * (w2 - z2);
  double xx = ww + yy + zz;
  double qq = 0.5 * (xx + (z1 - x1) * (x2 - y2));

  double w = qq - ww + (z1 - y1) * (y2 - z2);
  double x = qq - xx + (x1 + w1) * (x2 + w2);
  double y = qq - yy + (w1 - x1) * (y2 + z2);
  double z = qq - zz + (z1 + y1) * (w2 - x2);

  return {w, x, y, z};
}

// Extract heading angle from quaternion (returns angle in radians) - float version
inline float calc_heading(const std::array<float, 4>& q) {
  // Create reference direction [1, 0, 0]
  std::array<float, 3> ref_dir = {1.0f, 0.0f, 0.0f};

  // Rotate reference direction by quaternion
  std::array<float, 3> rot_dir = quat_rotate(q, ref_dir);

  // Compute heading as atan2(y, x) of rotated direction
  return std::atan2(rot_dir[1], rot_dir[0]);
}

// Extract heading angle from quaternion (returns angle in radians) - double version
inline double calc_heading_d(const std::array<double, 4>& q) {
  // Create reference direction [1, 0, 0]
  std::array<double, 3> ref_dir = {1.0, 0.0, 0.0};

  // Rotate reference direction by quaternion
  std::array<double, 3> rot_dir = quat_rotate_d(q, ref_dir);

  // Compute heading as atan2(y, x) of rotated direction
  return std::atan2(rot_dir[1], rot_dir[0]);
}


// Normalize a vector (matches Python normalize function) - float version
template<typename T>
inline std::array<T, 3> normalize_vector(const std::array<T, 3>& v, T eps = 1e-9f) {
  T norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  norm = std::max(norm, eps);
  return {v[0] / norm, v[1] / norm, v[2] / norm};
}

// Normalize a vector (matches Python normalize function) - double version
inline std::array<double, 3> normalize_vector_d(const std::array<double, 3>& v, double eps = 1e-12) {
  double norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  norm = std::max(norm, eps);
  return {v[0] / norm, v[1] / norm, v[2] / norm};
}

template<typename T>
inline std::tuple<T, std::array<T, 3> > quat_to_angle_axis(const std::array<T, 4> &q)
{

    std::array<T, 3> axis = {q[1], q[2], q[3]};
    auto shalf = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    T angle = 2 * std::asin(shalf);

    axis = normalize_vector(axis);

    return {angle, axis};
}

// Normalize a quaternion (matches Python quat_unit function) - float version
template<typename T>
inline std::array<T, 4> quat_unit(const std::array<T, 4>& q, T eps = 1e-9f) {
  T norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  norm = std::max(norm, eps);
  return {q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm};
}

// Normalize a quaternion (matches Python quat_unit function) - double version
inline std::array<double, 4> quat_unit_d(const std::array<double, 4>& q, double eps = 1e-12) {
  double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  norm = std::max(norm, eps);
  return {q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm};
}

// Create quaternion from angle and axis (matches Python quat_from_angle_axis) - float version
template<typename T>
std::array<T, 4> quat_from_angle_axis(T angle, const std::array<T, 3>& axis) {
  T theta = angle / 2.;
  T sin_theta = std::sin(theta);
  T cos_theta = std::cos(theta);

  // Normalize axis
  auto normalized_axis = normalize_vector(axis);

  // Create quaternion and normalize it (matches Python quat_unit call)
  std::array<T, 4> quat = {cos_theta, normalized_axis[0] * sin_theta, normalized_axis[1] * sin_theta,
                               normalized_axis[2] * sin_theta};
  return quat_unit(quat); // w, x, y, z
}

// Create quaternion from angle and axis (matches Python quat_from_angle_axis) - double version
inline std::array<double, 4> quat_from_angle_axis_d(double angle, const std::array<double, 3>& axis) {
  double theta = angle / 2.0;
  double sin_theta = std::sin(theta);
  double cos_theta = std::cos(theta);

  // Normalize axis
  auto normalized_axis = normalize_vector_d(axis);

  // Create quaternion and normalize it (matches Python quat_unit call)
  std::array<double, 4> quat = {cos_theta, normalized_axis[0] * sin_theta, normalized_axis[1] * sin_theta,
                                normalized_axis[2] * sin_theta};
  return quat_unit_d(quat); // w, x, y, z
}

// Calculate heading quaternion from full quaternion (wxyz format) - float version
inline std::array<float, 4> calc_heading_quat(const std::array<float, 4>& q) {
  float heading = calc_heading(q);
  std::array<float, 3> axis = {0.0f, 0.0f, 1.0f}; // z-axis
  return quat_from_angle_axis(heading, axis);
}

// Calculate heading quaternion from full quaternion (wxyz format) - double version
inline std::array<double, 4> calc_heading_quat_d(const std::array<double, 4>& q) {
  double heading = calc_heading_d(q);
  std::array<double, 3> axis = {0.0, 0.0, 1.0}; // z-axis
  return quat_from_angle_axis_d(heading, axis);
}

// Calculate inverse heading quaternion (matches Python implementation) - float version
inline std::array<float, 4> calc_heading_quat_inv(const std::array<float, 4>& q) {
  float heading = calc_heading(q);
  std::array<float, 3> axis = {0.0f, 0.0f, 1.0f}; // z-axis
  return quat_from_angle_axis(-heading, axis); // Note: negative heading
}

// Calculate inverse heading quaternion (matches Python implementation) - double version
inline std::array<double, 4> calc_heading_quat_inv_d(const std::array<double, 4>& q) {
  double heading = calc_heading_d(q);
  std::array<double, 3> axis = {0.0, 0.0, 1.0}; // z-axis
  return quat_from_angle_axis_d(-heading, axis); // Note: negative heading
}

// Create quaternion from Euler Z rotation (returns wxyz format) - float version
inline std::array<float, 4> euler_z_to_quat(float angle_rad) {
  std::array<float, 3> z_axis = {0.0f, 0.0f, 1.0f};
  return quat_from_angle_axis(angle_rad, z_axis);
}

// Create quaternion from Euler Z rotation (returns wxyz format) - double version
inline std::array<double, 4> euler_z_to_quat_d(double angle_rad) {
  std::array<double, 3> z_axis = {0.0, 0.0, 1.0};
  return quat_from_angle_axis_d(angle_rad, z_axis);
}

// Quaternion conjugate (negate x, y, z components) - float version
inline std::array<float, 4> quat_conjugate(const std::array<float, 4>& quat) {
  return {quat[0], -quat[1], -quat[2], -quat[3]}; // w, -x, -y, -z
}

// Quaternion conjugate (negate x, y, z components) - double version
inline std::array<double, 4> quat_conjugate_d(const std::array<double, 4>& quat) {
  return {quat[0], -quat[1], -quat[2], -quat[3]}; // w, -x, -y, -z
}

// Spherical Linear Interpolation (SLERP) for quaternions - double version
inline std::array<double, 4> quat_slerp_d(const std::array<double, 4>& q0, const std::array<double, 4>& q1, double t) {
  // Compute dot product (cos of half angle between quaternions)
  double dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
  
  // If dot is negative, slerp along the shorter path by using -q1
  std::array<double, 4> q1_adjusted = q1;
  if (dot < 0.0) {
    q1_adjusted = {-q1[0], -q1[1], -q1[2], -q1[3]};
    dot = -dot;
  }
  
  // Use linear interpolation for very close quaternions to avoid division by near-zero
  if (dot > 0.9995) {
    // Linear interpolation
    std::array<double, 4> result = {
      q0[0] + t * (q1_adjusted[0] - q0[0]),
      q0[1] + t * (q1_adjusted[1] - q0[1]),
      q0[2] + t * (q1_adjusted[2] - q0[2]),
      q0[3] + t * (q1_adjusted[3] - q0[3])
    };
    return quat_unit_d(result);
  } else {
    // Spherical linear interpolation
    double theta = std::acos(std::abs(dot));
    double sin_theta = std::sin(theta);
    double factor0 = std::sin((1.0 - t) * theta) / sin_theta;
    double factor1 = std::sin(t * theta) / sin_theta;
    
    return {
      factor0 * q0[0] + factor1 * q1_adjusted[0],
      factor0 * q0[1] + factor1 * q1_adjusted[1], 
      factor0 * q0[2] + factor1 * q1_adjusted[2],
      factor0 * q0[3] + factor1 * q1_adjusted[3]
    };
  }
}

// Convert quaternion to rotation matrix (wxyz format input) - float version
inline std::array<std::array<float, 3>, 3> quat_to_rotation_matrix(const std::array<float, 4>& quat) {
  float w = quat[0], x = quat[1], y = quat[2], z = quat[3];

  // Normalize quaternion
  float norm = std::sqrt(w * w + x * x + y * y + z * z);
  w /= norm;
  x /= norm;
  y /= norm;
  z /= norm;

  // Compute rotation matrix elements
  std::array<std::array<float, 3>, 3> R;

  R[0][0] = 1 - 2 * (y * y + z * z);
  R[0][1] = 2 * (x * y - w * z);
  R[0][2] = 2 * (x * z + w * y);

  R[1][0] = 2 * (x * y + w * z);
  R[1][1] = 1 - 2 * (x * x + z * z);
  R[1][2] = 2 * (y * z - w * x);

  R[2][0] = 2 * (x * z - w * y);
  R[2][1] = 2 * (y * z + w * x);
  R[2][2] = 1 - 2 * (x * x + y * y);

  return R;
}

// Convert quaternion to rotation matrix (wxyz format input) - double version
inline std::array<std::array<double, 3>, 3> quat_to_rotation_matrix_d(const std::array<double, 4>& quat) {
  double w = quat[0], x = quat[1], y = quat[2], z = quat[3];

  // Normalize quaternion
  double norm = std::sqrt(w * w + x * x + y * y + z * z);
  w /= norm;
  x /= norm;
  y /= norm;
  z /= norm;

  // Compute rotation matrix elements
  std::array<std::array<double, 3>, 3> R;

  R[0][0] = 1 - 2 * (y * y + z * z);
  R[0][1] = 2 * (x * y - w * z);
  R[0][2] = 2 * (x * z + w * y);

  R[1][0] = 2 * (x * y + w * z);
  R[1][1] = 1 - 2 * (x * x + z * z);
  R[1][2] = 2 * (y * z - w * x);

  R[2][0] = 2 * (x * z - w * y);
  R[2][1] = 2 * (y * z + w * x);
  R[2][2] = 1 - 2 * (x * x + y * y);

  return R;
}

// Convert 3x3 rotation matrix to quaternion (wxyz format output) - float version
inline std::array<float, 4> rotation_matrix_to_quat(const std::array<std::array<float, 3>, 3>& R) {
  float trace = R[0][0] + R[1][1] + R[2][2];
  float w, x, y, z;
  
  if (trace > 0.0f) {
    float s = 0.5f / std::sqrt(trace + 1.0f);
    w = 0.25f / s;
    x = (R[2][1] - R[1][2]) * s;
    y = (R[0][2] - R[2][0]) * s;
    z = (R[1][0] - R[0][1]) * s;
  } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
    float s = 2.0f * std::sqrt(1.0f + R[0][0] - R[1][1] - R[2][2]);
    w = (R[2][1] - R[1][2]) / s;
    x = 0.25f * s;
    y = (R[0][1] + R[1][0]) / s;
    z = (R[0][2] + R[2][0]) / s;
  } else if (R[1][1] > R[2][2]) {
    float s = 2.0f * std::sqrt(1.0f + R[1][1] - R[0][0] - R[2][2]);
    w = (R[0][2] - R[2][0]) / s;
    x = (R[0][1] + R[1][0]) / s;
    y = 0.25f * s;
    z = (R[1][2] + R[2][1]) / s;
  } else {
    float s = 2.0f * std::sqrt(1.0f + R[2][2] - R[0][0] - R[1][1]);
    w = (R[1][0] - R[0][1]) / s;
    x = (R[0][2] + R[2][0]) / s;
    y = (R[1][2] + R[2][1]) / s;
    z = 0.25f * s;
  }
  
  return {w, x, y, z};
}

// Convert 3x3 rotation matrix to quaternion (wxyz format output) - double version
inline std::array<double, 4> rotation_matrix_to_quat_d(const std::array<std::array<double, 3>, 3>& R) {
  double trace = R[0][0] + R[1][1] + R[2][2];
  double w, x, y, z;
  
  if (trace > 0.0) {
    double s = 0.5 / std::sqrt(trace + 1.0);
    w = 0.25 / s;
    x = (R[2][1] - R[1][2]) * s;
    y = (R[0][2] - R[2][0]) * s;
    z = (R[1][0] - R[0][1]) * s;
  } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
    double s = 2.0 * std::sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]);
    w = (R[2][1] - R[1][2]) / s;
    x = 0.25 * s;
    y = (R[0][1] + R[1][0]) / s;
    z = (R[0][2] + R[2][0]) / s;
  } else if (R[1][1] > R[2][2]) {
    double s = 2.0 * std::sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]);
    w = (R[0][2] - R[2][0]) / s;
    x = (R[0][1] + R[1][0]) / s;
    y = 0.25 * s;
    z = (R[1][2] + R[2][1]) / s;
  } else {
    double s = 2.0 * std::sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]);
    w = (R[1][0] - R[0][1]) / s;
    x = (R[0][2] + R[2][0]) / s;
    y = (R[1][2] + R[2][1]) / s;
    z = 0.25 * s;
  }
  
  return {w, x, y, z};
}

#endif // MATH_UTILS_HPP
