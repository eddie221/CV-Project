# CV Project -- Stereo Image for Depth Map
## TODO:
- [X] Find Essential matrix or Fundamental matrix (in func.py) (version 0.2)
- [X] Find translation vector (T) (in func.py) (version 0.4)
- [X] Find rotation matrix (R) (in func.py) (version 0.6) (using openCV to solve)
- [X] Rectified two images (in rectified.py) (version 0.8) (using openCV to solve)
- [X] Get the Disparity Map (in correspondence.py) (version 0.9)
- [ ] Get the Depth Map (in func.py) (version 1.0)
- The unstable reason because for using RANSAC to get  F

## Dataset:
 - https://vision.middlebury.edu/stereo/data/scenes2001/

## Reference:
1. Get T and R from Essential Matrix. 
 - https://en.wikipedia.org/wiki/Essential_matrix
 - https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf
2. Similar work
 - https://github.com/savnani5/Depth-Estimation-using-Stereovision
