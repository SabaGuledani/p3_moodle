# Camera Calibration and Augmented Reality

A computer vision project implementing camera calibration and augmented reality using OpenCV. This project demonstrates camera calibration techniques using chessboard patterns and real-time 3D object overlay in augmented reality.

## Portfolio Summary

This project showcases my implementation of camera calibration and augmented reality systems. I developed a C++ application using OpenCV that calibrates cameras using chessboard patterns and overlays 3D objects (axes and cubes) in real-time. The system handles camera distortion correction, pose estimation, and real-time rendering of virtual objects on physical markers.

## Course Information

**University:** University of Córdoba (UCO)  
**Course:** FSIV (Fundamentos de Sistemas de Información Visual)  
**Project:** P3 - From Calibration to Augmented Reality

## Features

- **Camera Calibration**: Calibrate cameras using chessboard patterns to compute intrinsic parameters and distortion coefficients
- **Distortion Correction**: Real-time undistortion of camera frames
- **Pose Estimation**: Estimate camera pose relative to chessboard markers using PnP (Perspective-n-Point) algorithm
- **Augmented Reality**: Overlay 3D objects (coordinate axes and cubes) on detected chessboard patterns
- **Real-time Processing**: Process live camera feed or video files in real-time
- **Interactive Controls**: Keyboard controls for calibration and AR visualization

## Requirements

- C++17 compatible compiler
- CMake (version 3.5 or higher)
- OpenCV library

## Building the Project

```bash
mkdir build
cd build
cmake ..
make
```

The executable `fsiv_calibAR` will be created in the build directory.

## Usage

### Camera Calibration

Calibrate your camera using a chessboard pattern:

```bash
./fsiv_calibAR --calibrate --camera=0 --rows=6 --cols=9 --square=25 --out=camera_params.yml
```

**Controls:**
- `SPACE`: Capture a view when chessboard is detected
- `d`: Toggle drawing detected corners
- `c`: Compute calibration (requires at least 8 views)
- `r`: Reset stored views
- `ESC`: Exit

### Augmented Reality Mode

Run AR mode with pre-calibrated camera parameters:

```bash
./fsiv_calibAR --run --camera=0 --params=camera_params.yml --rows=6 --cols=9 --square=25 --draw=cube
```

**Controls:**
- `a`: Toggle axes overlay
- `u`: Toggle cube overlay
- `s`: Save screenshot
- `ESC`: Exit

### Using Video Files

You can also process video files instead of live camera:

```bash
./fsiv_calibAR --run --video=video.mp4 --params=camera_params.yml --rows=6 --cols=9 --square=25
```

## Technical Details

The project implements:
- Chessboard corner detection with subpixel refinement
- Camera calibration using multiple views
- Reprojection error calculation for calibration quality assessment
- Undistortion using precomputed remap tables for efficiency
- PnP pose estimation for 6-DOF camera pose
- 3D object rendering using OpenCV's projection functions

## Project Structure

- `fsiv_main_p3.cpp`: Main application with calibration and AR modes
- `fsiv_funcs.cpp`: Implementation of calibration and AR functions
- `fsiv_funcs.hpp`: Function declarations and data structures
- `CMakeLists.txt`: Build configuration
- `tablero_params.yml`: Example calibration parameters file

## License

This project was developed as part of coursework at the University of Córdoba.
