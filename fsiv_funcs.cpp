// File: fsiv_funcs.cpp
// (c) mjmarin

/**
 * @file fsiv_funcs.cpp
 * @brief Definitions for FSIV P3 helper functions (classic, cross-platform).
 */

#include "fsiv_funcs.hpp"
#include <iostream>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

void fsiv_create_chessboard_3d_points(const cv::Size& pattern_size,
                                      float square_size,
                                      std::vector<cv::Point3f>& object_points)
{
    ;
}

bool fsiv_find_chessboard_corners(const cv::Mat& image,
                                  const cv::Size& pattern_size,
                                  std::vector<cv::Point2f>& corners,
                                  bool fast_preview)
{
    ;
}

double fsiv_calibrate_camera(const std::vector<std::vector<cv::Point3f> >& object_points_list,
                             const std::vector<std::vector<cv::Point2f> >& image_points_list,
                             const cv::Size& image_size,
                             cv::Mat& camera_matrix, cv::Mat& dist_coeffs,
                             std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs)
{
    ;
}

double fsiv_compute_reprojection_error(const std::vector<std::vector<cv::Point3f> >& object_points_list,
                                       const std::vector<std::vector<cv::Point2f> >& image_points_list,
                                       const std::vector<cv::Mat>& rvecs,
                                       const std::vector<cv::Mat>& tvecs,
                                       const cv::Mat& K, const cv::Mat& dist)
{
    ;
}

bool fsiv_save_calibration(const std::string& path,
                           const cv::Mat& K, const cv::Mat& dist,
                           const cv::Size& image_size,
                           double reproj_error)
{
;
}

bool fsiv_load_calibration(const std::string& path,
                           cv::Mat& K, cv::Mat& dist)
{
;
}

void fsiv_prepare_undistort_maps(const cv::Mat& K, const cv::Mat& dist,
                                 const cv::Size& image_size,
                                 cv::Mat& map1, cv::Mat& map2)
{
 ;
}

void fsiv_undistort_with_maps(const cv::Mat& src, cv::Mat& dst,
                              const cv::Mat& map1, const cv::Mat& map2)
{
 ;
}

bool fsiv_estimate_pose(const std::vector<cv::Point3f>& object_points,
                        const std::vector<cv::Point2f>& image_points,
                        const cv::Mat& K, const cv::Mat& dist,
                        cv::Mat& rvec, cv::Mat& tvec)
{
;
}

void fsiv_draw_axes(cv::Mat& image, const cv::Mat& K, const cv::Mat& dist,
                    const cv::Mat& rvec, const cv::Mat& tvec, float axis_length)
{
;
}

void fsiv_draw_cube(cv::Mat& image, const cv::Mat& K, const cv::Mat& dist,
                    const cv::Mat& rvec, const cv::Mat& tvec, float square_size)
{
;
}

