// File: fsiv_funcs.hpp
// (c) mjmarin

/**
 * @file fsiv_funcs.hpp
 * @brief Declarations for FSIV P3 helper functions.
 */

#ifndef FSIV_FUNCS_HPP
#define FSIV_FUNCS_HPP

#include <vector>
#include <opencv2/core.hpp>
struct FSIVParams
{
    int camera = 0;
    std::string video = "";
    int rows = 6;
    int cols = 9;
    float square = 25.0f;
    bool calibrate = false;
    std::string out = "camera_params.yml";
    bool run = false;
    std::string params = "";
    std::string draw = "axes";
};

/**
 * @brief Create the 3D points (object points) for a planar chessboard pattern.
 *
 * The points are generated on Z=0 plane with the origin at the top-left inner corner.
 * X grows with columns; Y grows with rows.
 *
 * @param pattern_size Number of inner corners (cols, rows).
 * @param square_size Size of one square in user units (e.g. millimeters).
 * @param object_points Output vector of 3D points.
 */
void fsiv_create_chessboard_3d_points(const cv::Size& pattern_size,
                                      float square_size,
                                      std::vector<cv::Point3f>& object_points);

/**
 * @brief Find and refine chessboard corners in a BGR or grayscale image.
 *
 * @param image BGR or grayscale input.
 * @param pattern_size Number of inner corners (cols, rows).
 * @param corners Output 2D corners (image coordinates).
 * @param fast_preview If true, uses FAST_CHECK and does not refine (quick preview).
 * @return true if corners were found.
 */
bool fsiv_find_chessboard_corners(const cv::Mat& image,
                                  const cv::Size& pattern_size,
                                  std::vector<cv::Point2f>& corners,
                                  bool fast_preview);

/**
 * @brief Run OpenCV camera calibration.
 *
 * @param object_points_list List of 3D points for each view.
 * @param image_points_list List of 2D corners for each view.
 * @param image_size Image size (width, height).
 * @param camera_matrix Output intrinsics.
 * @param dist_coeffs Output distortion coefficients.
 * @param rvecs Output rotation vectors (one per view).
 * @param tvecs Output translation vectors (one per view).
 * @return RMS re-projection error from calibrateCamera().
 */
double fsiv_calibrate_camera(const std::vector<std::vector<cv::Point3f> >& object_points_list,
                             const std::vector<std::vector<cv::Point2f> >& image_points_list,
                             const cv::Size& image_size,
                             cv::Mat& camera_matrix, cv::Mat& dist_coeffs,
                             std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs);

/**
 * @brief Compute mean reprojection error over all views (for reporting).
 *
 * @param object_points_list Object points per view.
 * @param image_points_list Image points per view.
 * @param rvecs Rotation vectors from calibration.
 * @param tvecs Translation vectors from calibration.
 * @param K Camera matrix.
 * @param dist Distortion coefficients.
 * @return Mean pixel error.
 */
double fsiv_compute_reprojection_error(const std::vector<std::vector<cv::Point3f> >& object_points_list,
                                       const std::vector<std::vector<cv::Point2f> >& image_points_list,
                                       const std::vector<cv::Mat>& rvecs,
                                       const std::vector<cv::Mat>& tvecs,
                                       const cv::Mat& K, const cv::Mat& dist);

/**
 * @brief Save calibration data to a YAML/XML file.
 *
 * @param path Output file path (.yml or .xml).
 * @param K Camera matrix.
 * @param dist Distortion coefficients.
 * @param image_size Image size used in calibration.
 * @param reproj_error Mean reprojection error from calibrateCamera().
 * @return true if saved successfully.
 */
bool fsiv_save_calibration(const std::string& path,
                           const cv::Mat& K, const cv::Mat& dist,
                           const cv::Size& image_size, double reproj_error);

/**
 * @brief Load calibration data from a YAML/XML file.
 *
 * @param path Input file path.
 * @param K Output camera matrix.
 * @param dist Output distortion coefficients.
 * @return true if loaded successfully and data look valid.
 */
bool fsiv_load_calibration(const std::string& path,
                           cv::Mat& K, cv::Mat& dist);

/**
 * @brief Prepare undistort rectify maps for fast remap() undistortion.
 *
 * @param K Camera matrix.
 * @param dist Distortion coefficients.
 * @param image_size Size of frames to be undistorted.
 * @param map1 Output map1.
 * @param map2 Output map2.
 */
void fsiv_prepare_undistort_maps(const cv::Mat& K, const cv::Mat& dist,
                                 const cv::Size& image_size,
                                 cv::Mat& map1, cv::Mat& map2);

/**
 * @brief Apply undistortion using precomputed maps.
 *
 * @param src Input frame.
 * @param dst Output undistorted frame.
 * @param map1 Map1 from initUndistortRectifyMap.
 * @param map2 Map2 from initUndistortRectifyMap.
 */
void fsiv_undistort_with_maps(const cv::Mat& src, cv::Mat& dst,
                              const cv::Mat& map1, const cv::Mat& map2);

/**
 * @brief Solve PnP pose estimation from 2D-3D correspondences.
 *
 * @param object_points 3D object points (single view).
 * @param image_points 2D image points (same order as object points).
 * @param K Camera matrix.
 * @param dist Distortion coefficients.
 * @param rvec Output rotation vector.
 * @param tvec Output translation vector.
 * @return true if solvePnP succeeded.
 */
bool fsiv_estimate_pose(const std::vector<cv::Point3f>& object_points,
                        const std::vector<cv::Point2f>& image_points,
                        const cv::Mat& K, const cv::Mat& dist,
                        cv::Mat& rvec, cv::Mat& tvec);

/**
 * @brief Draw 3D coordinate axes (X:red, Y:green, Z:blue) over the image.
 *
 * The axes are drawn starting at the chessboard origin (0,0,0).
 *
 * @param image BGR image to draw on.
 * @param K Camera matrix.
 * @param dist Distortion coefficients.
 * @param rvec Rotation vector.
 * @param tvec Translation vector.
 * @param axis_length Length of each axis (in the same units as the chessboard).
 */
void fsiv_draw_axes(cv::Mat& image, const cv::Mat& K, const cv::Mat& dist,
                    const cv::Mat& rvec, const cv::Mat& tvec, float axis_length);

/**
 * @brief Draw a wireframe cube whose base sits on the first chessboard square.
 *
 * The cube base is one square (square_size x square_size); height equals square_size.
 *
 * @param image BGR image to draw on.
 * @param K Camera matrix.
 * @param dist Distortion coefficients.
 * @param rvec Rotation vector.
 * @param tvec Translation vector.
 * @param square_size Size of one square (same units used in calibration).
 */
void fsiv_draw_cube(cv::Mat& image, const cv::Mat& K, const cv::Mat& dist,
                    const cv::Mat& rvec, const cv::Mat& tvec, float square_size);

#endif // FSIV_FUNCS_HPP
