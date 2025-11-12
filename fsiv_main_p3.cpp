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
#include <ctime> 
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
        return 1;
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

    // validate arguments if user does not specify them
    if (!params.calibrate && !params.run) {
        std::cerr << "Error: Must specify either --calibrate or --run mode." << std::endl;
        print_help();
        return 1;
    }

    // open input (camera or video)
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
    // intialize variables for calibration mode
    std::vector<std::vector<cv::Point3f>> object_points_list; // list of 3D object points
    std::vector<std::vector<cv::Point2f>> image_points_list; // list of 2D image points
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    std::vector<cv::Mat> rvecs, tvecs; //rotation and translation vecttors 
    bool draw_corners = false;

    // AR mode variables
    cv::Mat map1, map2;  // undistort maps
    cv::Mat rvec, tvec;   //rotation and translation vecttors 
    cv::Mat zero_dist; // zero distortion coefficients for undistorted images
    bool draw_axes = false;   // flag to draw axes overlay
    bool draw_cube = false;   // flag to draw cube overlay

    // create pattern size and 3D object points (same for all views)
    cv::Size pattern_size(params.cols, params.rows);
    std::vector<cv::Point3f> object_points;
    if (params.calibrate) {
        fsiv_create_chessboard_3d_points(pattern_size, params.square, object_points);
    }
    // AR mode initialization
    if (params.run) {
        // validate that calibration file path is provided
        if (params.params.empty()) {
            std::cerr << "Error: AR mode requires --params=<file.yml> to load calibration parameters." << std::endl;
            print_help();
            cap.release();
            return 1;
        }
        
        // load calibration parameters from file
        if (!fsiv_load_calibration(params.params, camera_matrix, dist_coeffs)) {
            std::cerr << "Error: Failed to load calibration from " << params.params << std::endl;
            cap.release();
            return 1;
        }
        
        std::cout << "Calibration loaded successfully from: " << params.params << std::endl;

        // create 3D object points for the chessboard pattern (needed for pose estimation)
        fsiv_create_chessboard_3d_points(pattern_size, params.square, object_points);
        std::cout << "Created " << object_points.size() << " 3D object points for chessboard pattern" << std::endl;

        // prepare undistort maps once (for efficiency in real-time processing)
        cv::Size image_size = cv::Size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                                       static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        fsiv_prepare_undistort_maps(camera_matrix, dist_coeffs, image_size, map1, map2);
        std::cout << "Prepared undistort maps for image size: " << image_size.width << "x" << image_size.height << std::endl;
        zero_dist = cv::Mat::zeros(dist_coeffs.rows, dist_coeffs.cols, dist_coeffs.type());
        // initialize drawing state based on --draw parameter
        if (params.draw == "axes") {
            draw_axes = true;
        } else if (params.draw == "cube") {
            draw_cube = true;
        }
   
    }
    
    for (;;)
    {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty())
        {
            // end of file or camera error
            if (params.use_video) break;
            else continue;
        }

        if (params.calibrate)
        {
            // make frame grayscale
            cv::Mat gray;
            cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            // create empty vector corners_tmp
            std::vector<cv::Point2f> corners_tmp;
            bool found = fsiv_find_chessboard_corners(gray, pattern_size, corners_tmp, true);
            // if found and draw corners are true use drawChessboardCorners method
            if(draw_corners && found){
                cv::drawChessboardCorners(frame, pattern_size, corners_tmp, found);
            }
            int key = cv::waitKey(params.use_video ? 30 : 1) & 0xFF;
            if (key == 27){ //ESC: exit
                break;
            }else if (key == ' ' && found) {  // SPACE: store view
                // refine corners with subpix
                std::vector<cv::Point2f> refined_corners = corners_tmp;
                if (!refined_corners.empty()) {
                    cv::Mat gray_for_refine;
                    cvtColor(frame, gray_for_refine, cv::COLOR_BGR2GRAY);
                    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
                    cv::cornerSubPix(gray_for_refine, refined_corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
                }
                
                // store this view
                object_points_list.push_back(object_points);  // store 3d points for all views
                image_points_list.push_back(refined_corners);  // store 2d points for each view separately
                
                std::cout << "Stored view " << object_points_list.size() << std::endl;
            }else if (key == 'd') {  // toggle corner drawing
                draw_corners = !draw_corners;
            }else if (key == 'c') {  // run calibration
                if (object_points_list.size() < 8){
                    std::cerr << "calibration needs at least 8 views, current views: " << object_points_list.size() << std::endl;

                }else{
                    cv::Size img_size = frame.size();

                    // calibrate camera
                    double rms = fsiv_calibrate_camera(object_points_list, image_points_list,
                                                        img_size, camera_matrix, dist_coeffs, rvecs, tvecs);
                    // compute reprojection error
                    double mean_error = fsiv_compute_reprojection_error(object_points_list, image_points_list,
                        rvecs, tvecs, camera_matrix, dist_coeffs);

                    // print calibration results
                    std::cout << "calibration completed" << std::endl;
                    std::cout << "RMS error:" << rms << std::endl;
                    std::cout << "Mean reprojection error:" << mean_error << std::endl;

                    // save calibration to file
                    if (fsiv_save_calibration(params.out, camera_matrix, dist_coeffs, img_size, mean_error)) {
                        std::cout << "Calibration saved to: " << params.out << std::endl;
                    } else {
                        std::cerr << "Error: Failed to save calibration to " << params.out << std::endl;
                    }


                    
                }

            }else if (key == 'r') {  // reset stored views
                object_points_list.clear();
                image_points_list.clear();
                std::cout << "Reset: cleared all stored views" << std::endl;
            }

            
}
        else // AR mode
        {
            // undistort the frame with precomputed maps
            cv::Mat undist;
            fsiv_undistort_with_maps(frame, undist, map1, map2);
    
            cv::Mat undist_gray;
            cv::cvtColor(undist, undist_gray, cv::COLOR_BGR2GRAY);
    
            std::vector<cv::Point2f> corners;
            bool found = fsiv_find_chessboard_corners(undist_gray, pattern_size, corners, false);
            
            // if corners found, estimate pose and draw overlays
            if (found) {
                // estimate camera pose and check if it is ok
                bool pnp_ok = fsiv_estimate_pose(object_points, corners, camera_matrix, zero_dist, rvec, tvec);
                
                if (pnp_ok) {
                    // draw overlays (cube or axes)
                    // we use zero_dist since the image is already undistorted
                    if (draw_axes) {
                        fsiv_draw_axes(undist, camera_matrix, zero_dist, rvec, tvec, params.square * 2.0f);
                    }
                    if (draw_cube) {
                        fsiv_draw_cube(undist, camera_matrix, zero_dist, rvec, tvec, params.square);
                    }
                }
            }
            
            // copy undistorted frame to display
            frame = undist.clone();
        }
        cv::imshow("FSIV P3", frame);

        // handle keyboard input for AR mode
        if (params.run) {
            int key = cv::waitKey(params.use_video ? 30 : 1) & 0xFF;
            if (key == 27) { 
                break;
            } else if (key == 'a') {  // toggle/force draw axes
                draw_axes = !draw_axes;
                draw_cube = false;  //turn off cube
                std::cout << "Axes overlay: " << (draw_axes ? "ON" : "OFF") << std::endl;
            } else if (key == 'u') {  // toggle/force draw cube
                draw_cube = !draw_cube;
                draw_axes = false;  //turn off axes
                std::cout << "Cube overlay: " << (draw_cube ? "ON" : "OFF") << std::endl;
            } else if (key == 's') {  // save screenshot
                std::string filename = "screenshot_" + std::to_string(time(nullptr)) + ".png";
                cv::imwrite(filename, frame);
                std::cout << "Screenshot saved: " << filename << std::endl;
            }
        } else {

            if (cv::waitKey(params.use_video ? 30 : 1) == 27) {
                break;
            }
        }
        
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
