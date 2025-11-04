// File: fsiv_main_p3.cpp
// (c) mjmarin

/**
 * @file fsiv_main_p3.cpp
 * @brief FSIV P3 - From Calibration to Augmented Reality.
 *
 * Usage examples:
 *  Calibration (webcam 0), chessboard 9x6 inner corners, 25mm squares:
 *    fsiv_p3 --calibrate --camera=0 --rows=6 --cols=9 --square=25 --out=camera_params.yml
 *    # Press SPACE to grab good views; press 'c' to compute and save; ESC to quit.
 *
 *  AR (load params and draw cube):
 *    fsiv_p3 --run --camera=0 --params=camera_params.yml --rows=6 --cols=9 --square=25 --draw=cube
 *
 *  AR over a video file:
 *    fsiv_p3 --run --video=video.mp4 --params=camera_params.yml --rows=6 --cols=9 --square=25
 *
 * Keys in calibration mode:
 *   - SPACE: try to detect chessboard; if found, store this view
 *   - d: toggle drawing detected corners
 *   - c: run calibration with stored views and write parameters
 *   - r: reset stored views
 *   - ESC: exit
 *
 * Keys in AR mode:
 *   - a: draw axes
 *   - u: draw cube
 *   - s: save screenshot
 *   - ESC: exit
 */

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "fsiv_funcs.hpp"

static void print_help()
{
    std::cout
    << "FSIV P3 - From Calibration to Augmented Reality\n"
    << "Options:\n"
    << "  --calibrate                 Run calibration mode.\n"
    << "  --run                       Run AR mode (pose + overlay).\n"
    << "  --camera=<int>              Open camera index (default 0).\n"
    << "  --video=<path>              Open video file instead of camera.\n"
    << "  --rows=<int>                Checkerboard inner rows (e.g. 6).\n"
    << "  --cols=<int>                Checkerboard inner cols (e.g. 9).\n"
    << "  --square=<float>            Square size (e.g. 25.0 in mm or chosen units).\n"
    << "  --out=<file.yml>            Output calibration file (calibration mode).\n"
    << "  --params=<file.yml>         Input calibration file (AR mode).\n"
    << "  --draw=<axes|cube>          What to draw in AR mode (default axes).\n"
    << "  --help                      Print this help.\n"
    << std::endl;
}

// TODO: use cv::CommandLineParser

int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h ? |       | Show help }"
        "{camera   |       | Open a camera index }"
        "{video    |       | Open a video file instead of a live camera }"
        "{rows     |  5    | Number of inner rows of the chessboard }"
        "{cols     |  6    | Number of inner columns of the chessboard }"
        "{square   |  25.0 | Square size in user units. Must be > 0 }"
        "{calibrate|       | Enable calibration mode }"
        "{out      |  camera_params.yml  | Output path for calibration file }"
        "{run      |       | Enable AR mode (pose + overlay) }"
        "{params   |       | Path to an existing calibration file to load }"
        "{draw     |  axes | Overlay to draw in AR mode }";
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("FSIV Practical XXX");
    if (parser.has("help") || !parser.check()) {
        print_help();
    }

    FSIVParams params;
    params.camera = parser.get<int>("camera");
    params.video = parser.get<std::string>("video");
    params.use_video = parser.has("video");
    params.rows = parser.get<int>("rows");
    params.cols = parser.get<int>("cols");
    params.square = parser.get<float>("square");
    params.calibrate = parser.has("calibrate");
    params.out = parser.get<std::string>("out");
    params.run = parser.has("run");
    params.params = parser.get<std::string>("params");
    params.draw = parser.get<std::string>("draw");

    // Validate arguments
    if (!params.calibrate && !params.run) {
        std::cerr << "Error: Must specify either --calibrate or --run mode." << std::endl;
        print_help();
        return 1;
    }

    // Open input (camera or video)
    cv::VideoCapture cap;
    if (params.use_video) {
        cap.open(params.video);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file: " << params.video << std::endl;
            return 1;
        }
    } else {
        cap.open(params.camera);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera " << params.camera << std::endl;
            return 1;
        }
    }

    for (;;)
    {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty())
        {
            // End of file or camera error.
            if (params.use_video) break;
            else continue; // try again for cameras
        }

        if (params.calibrate)
        {
            // make frame grayscale
            cv::Mat gray;
            cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            // create the 3D points for the chessboard
            // first create pattern size from rows and cols parameters
            cv::Size pattern_size(params.cols, params.rows);
            // create empty vector of 3d points
            // std::vector<cv::Point3f> obj_pts;
            // fsiv_create_chessboard_3d_points(pattern_size, params.square, obj_pts);
            // create empty vector corners_tmp
            std::vector<cv::Point2f> corners_tmp;
            bool found = fsiv_find_chessboard_corners(gray, pattern_size, corners_tmp, true);
  
            cv::drawChessboardCorners(frame, pattern_size, corners_tmp, found);
            cv::imshow("Calibration", frame);

  /*
                double rms = fsiv_calibrate_camera(all_object_points, all_image_points,
                                                   image_size, camera_matrix, dist_coeffs,
                                                   rvecs, tvecs);
    */

//                double mean_err = fsiv_compute_reprojection_error(all_object_points, all_image_points,
//                                                                  rvecs, tvecs, camera_matrix, dist_coeffs);

//                fsiv_save_calibration(out_path, camera_matrix, dist_coeffs, image_size, mean_err);

        }
        else // AR mode
        {
            
//                fsiv_prepare_undistort_maps(camera_matrix, dist_coeffs, image_size, map1, map2);

//            fsiv_undistort_with_maps(frame, undist, map1, map2);


//            bool found = fsiv_find_chessboard_corners(undist, pattern_size, corners, false);

//                bool pnp_ok = fsiv_estimate_pose(obj_pts, corners, camera_matrix, dist_coeffs, rvec, tvec);

//                        fsiv_draw_cube(undist, camera_matrix, dist_coeffs, rvec, tvec, (float)square_size);
                    
//                        fsiv_draw_axes(undist, camera_matrix, dist_coeffs, rvec, tvec, (float)square_size*2.0f);

        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
