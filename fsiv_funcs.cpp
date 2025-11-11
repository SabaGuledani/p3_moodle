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
    object_points.clear();
    for (int row = 0; row < pattern_size.height; ++row)
    {
        for (int col = 0; col < pattern_size.width; ++col)
        {
            object_points.push_back(cv::Point3f(col * square_size, row * square_size, 0.0f));
        }
    }

}

bool fsiv_find_chessboard_corners(const cv::Mat& image,
                                  const cv::Size& pattern_size,
                                  std::vector<cv::Point2f>& corners,
                                  bool fast_preview)
{

    // set flags for chessboard detection if fast_preview is true
     int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
    if (fast_preview)
    {
        flags = cv::CALIB_CB_FAST_CHECK;
    }

    // find chessboard corners
    bool found = cv::findChessboardCorners(image, pattern_size, corners, flags);

    // If corners found and not in fast preview mode use subpix method
    if (found && !fast_preview)
    {
        // termination criteria 
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
        
        // refine corner positions to subpix accuracy
        cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
    }

    return found;
}


double fsiv_calibrate_camera(const std::vector<std::vector<cv::Point3f> >& object_points_list,
                             const std::vector<std::vector<cv::Point2f> >& image_points_list,
                             const cv::Size& image_size,
                             cv::Mat& camera_matrix, cv::Mat& dist_coeffs,
                             std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs)
{
    // intialize camera matrix and dist_coeffs to identuty and zeros matrxies
    camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    // intialize flags to 0
    int flags = 0;
    // run calibration method
    double rms = cv::calibrateCamera(
        object_points_list,  // 3D points for each view
        image_points_list,   // 2D corners for each view
        image_size,          // Image dimensions
        camera_matrix,       // Output: intrinsic matrix
        dist_coeffs,         // Output: distortion coefficients
        rvecs,               // Output: rotation vectors (one per view)
        tvecs,               // Output: translation vectors (one per view)
        flags                // Calibration flags
    );
    return rms;
    
}

double fsiv_compute_reprojection_error(const std::vector<std::vector<cv::Point3f> >& object_points_list,
    const std::vector<std::vector<cv::Point2f> >& image_points_list,
    const std::vector<cv::Mat>& rvecs,
    const std::vector<cv::Mat>& tvecs,
    const cv::Mat& K, const cv::Mat& dist)
{
    double total_error = 0.0;
    int total_points = 0;

    // iterate over all views
    for (size_t i = 0; i < object_points_list.size(); ++i){
        // we project 3D object points to 2D image points using the calibrated parameters
        std::vector<cv::Point2f> projected_points;
        cv::projectPoints(object_points_list[i], rvecs[i], tvecs[i], K, dist, projected_points);

        // then compute the error between projected and actual image points
        const std::vector<cv::Point2f>& actual_points = image_points_list[i];

        // sum the Euclidean distances for all points in this view
        for (size_t j = 0; j < projected_points.size(); ++j){
            cv::Point2f diff = projected_points[j] - actual_points[j];
            double error = cv::norm(diff);  // euclidean distance
            total_error += error * error;  // sum of squared errors
            }

        total_points += static_cast<int>(projected_points.size());
    }

    // return RMS
    if (total_points > 0){
        return std::sqrt(total_error / total_points);
    }
    return 0.0;
}

bool fsiv_save_calibration(const std::string& path,
    const cv::Mat& K, const cv::Mat& dist,
    const cv::Size& image_size,
    double reproj_error)
    {
    // open file (.yaml or .xml)
    cv::FileStorage fs(path, cv::FileStorage::WRITE);

    if (!fs.isOpened())
    {
        std::cerr << "Error: Could not open file for writing: " << path << std::endl;
        return false;
    }

    // write fields
    fs << "image_width" << image_size.width;
    fs << "image_height" << image_size.height;
    fs << "camera_matrix" << K;
    fs << "distortion_coefficients" << dist;
    fs << "error" << reproj_error;

    // close the file
    fs.release();

    return true;
}

bool fsiv_load_calibration(const std::string& path,
    cv::Mat& K, cv::Mat& dist)
{
    // open file (.yaml or .xml)
    cv::FileStorage fs(path, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        std::cerr << "Error: Could not open file for reading: " << path << std::endl;
        return false;
    }

    // read camera matrix and distortion coefficients
    fs["camera_matrix"] >> K;
    fs["distortion_coefficients"] >> dist;

    // Close the file
    fs.release();

    //////data validation//////

    // validate that data was loaded correctly
    if (K.empty() || dist.empty())
    {
        std::cerr << "Error: Failed to load calibration data from " << path << std::endl;
        return false;
    }

    // validate dimensions
    if (K.rows != 3 || K.cols != 3)
    {
        std::cerr << "Error: Invalid camera matrix dimensions (expected 3x3)" << std::endl;
        return false;
    }
    // validate distortion coefficients
    if (dist.rows != 1 || (dist.cols != 5 && dist.cols != 1))
    {
        std::cerr << "Error: Invalid distortion coefficients dimensions (expected 1x5 or 5x1)" << std::endl;
        return false;
    }

    return true;
}

// void fsiv_prepare_undistort_maps(const cv::Mat& K, const cv::Mat& dist,
//                                  const cv::Size& image_size,
//                                  cv::Mat& map1, cv::Mat& map2)
// {
//  ;
// }

// void fsiv_undistort_with_maps(const cv::Mat& src, cv::Mat& dst,
//                               const cv::Mat& map1, const cv::Mat& map2)
// {
//  ;
// }

// bool fsiv_estimate_pose(const std::vector<cv::Point3f>& object_points,
//                         const std::vector<cv::Point2f>& image_points,
//                         const cv::Mat& K, const cv::Mat& dist,
//                         cv::Mat& rvec, cv::Mat& tvec)
// {
// ;
// }

// void fsiv_draw_axes(cv::Mat& image, const cv::Mat& K, const cv::Mat& dist,
//                     const cv::Mat& rvec, const cv::Mat& tvec, float axis_length)
// {
// ;
// }

// void fsiv_draw_cube(cv::Mat& image, const cv::Mat& K, const cv::Mat& dist,
//                     const cv::Mat& rvec, const cv::Mat& tvec, float square_size)
// {
// ;
// }

