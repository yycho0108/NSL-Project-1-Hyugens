#include "apriltag_opencv.h"
#include "apriltag_family.h"
extern "C"{
#include "common/homography.h"
}
#include "getopt.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

void getRelativeTransform(
        double tag_size,
        double fx, double fy,
        double px, double py,
        double* p, 
        double* tx, double* ty, double* tz
        ) {

  std::vector<cv::Point3f> objPts;
  std::vector<cv::Point2f> imgPts;
  double s = tag_size/2.;
  objPts.push_back(cv::Point3f(-s,-s, 0));
  objPts.push_back(cv::Point3f( s,-s, 0));
  objPts.push_back(cv::Point3f( s, s, 0));
  objPts.push_back(cv::Point3f(-s, s, 0));

  //std::pair<float, float> p1 = p[0];
  //std::pair<float, float> p2 = p[1];
  //std::pair<float, float> p3 = p[2];
  //std::pair<float, float> p4 = p[3];
  imgPts.push_back(cv::Point2f(p[0], p[1]));
  imgPts.push_back(cv::Point2f(p[2], p[3]));
  imgPts.push_back(cv::Point2f(p[4], p[5]));
  imgPts.push_back(cv::Point2f(p[6], p[7]));

  cv::Mat rvec, tvec;
  cv::Matx33f cameraMatrix(
                           fx, 0, px,
                           0, fy, py,
                           0,  0,  1);
  cv::Vec4f distParam(0,0,0,0); // all 0?
  cv::solvePnP(objPts, imgPts, cameraMatrix, distParam, rvec, tvec);

  *tx = tvec.at<double>(0);
  *ty = tvec.at<double>(1);
  *tz = tvec.at<double>(2);

  //cv::Matx33d r;
  //cv::Rodrigues(rvec, r);
  //Eigen::Matrix3d wRo;
  //wRo << r(0,0), r(0,1), r(0,2), r(1,0), r(1,1), r(1,2), r(2,0), r(2,1), r(2,2);

  //Eigen::Matrix4d T; 
  //T.topLeftCorner(3,3) = wRo;
  //T.col(3).head(3) << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
  //T.row(3) << 0,0,0,1;

  //return T;
}

int main(int argc, char** argv) {

  getopt_t *getopt = getopt_create();

  getopt_add_bool(getopt, 'h', "help", 0, "Show this help");
  getopt_add_bool(getopt, 'q', "quiet", 0, "Reduce output");
  getopt_add_string(getopt, 'f', "family", "tag36h11", "Tag family to use");
  getopt_add_int(getopt, '\0', "border", "1", "Set tag family border size");
  getopt_add_int(getopt, 't', "threads", "4", "Use this many CPU threads");
  getopt_add_double(getopt, 'x', "decimate", "1.0", "Decimate input image by this factor");
  getopt_add_double(getopt, 'b', "blur", "0.0", "Apply low-pass blur to input");
  getopt_add_bool(getopt, '0', "refine-edges", 1, "Spend more time trying to align edges of tags");
  getopt_add_bool(getopt, '1', "refine-decode", 0, "Spend more time trying to decode tags");
  getopt_add_bool(getopt, '2', "refine-pose", 0, "Spend more time trying to precisely localize tags");
  getopt_add_bool(getopt, 'c', "contours", 0, "Use new contour-based quad detection");

  if (!getopt_parse(getopt, argc, argv, 1) || getopt_get_bool(getopt, "help")) {
    printf("Usage: %s [options] Camera index or movie file\n", argv[0]);
    getopt_do_usage(getopt);
    exit(0);
  }

  const char *famname = getopt_get_string(getopt, "family");
  apriltag_family_t *tf = apriltag_family_create(famname);

  if (!tf) {
    printf("Unrecognized tag family name. Use e.g. \"tag36h11\".\n");
    exit(-1);
  }

  tf->black_border = getopt_get_int(getopt, "border");

  apriltag_detector_t *td = apriltag_detector_create();
  apriltag_detector_add_family(td, tf);

  if (getopt_get_bool(getopt, "contours")) {
    apriltag_detector_enable_quad_contours(td, 1);
  }
    
  td->quad_decimate = getopt_get_double(getopt, "decimate");
  td->quad_sigma = getopt_get_double(getopt, "blur");
  td->nthreads = getopt_get_int(getopt, "threads");
  td->debug = 0;
  td->refine_edges = getopt_get_bool(getopt, "refine-edges");
  td->refine_decode = getopt_get_bool(getopt, "refine-decode");
  td->refine_pose = getopt_get_bool(getopt, "refine-pose");

  const zarray_t *inputs = getopt_get_extra_args(getopt);

  int camera_index = 0;
  const char* movie_file = NULL;
  
  if (zarray_size(inputs) > 1) {
    printf("Usage: %s [options] Camera index or movie file\n", argv[0]);
    exit(-1);
  } else if (zarray_size(inputs)) {
    char* input;
    zarray_get(inputs, 0, &input);
    char* endptr;
    camera_index = strtol(input, &endptr, 10);
    if (!endptr || *endptr) {
      movie_file = input;
    }
  }

  cv::VideoCapture* cap;

  if (movie_file) {
        cap = new cv::VideoCapture(movie_file);
        int length = int(cap->get(CV_CAP_PROP_FRAME_COUNT));
        printf("Video Length : %d\n" , length);
  } else {
    cap = new cv::VideoCapture(camera_index);
  }

  const char* window = "Camera";

  cv::Mat frame;

  cv::namedWindow(window);

  while (1) {
    bool ok = cap->read(frame);
    if (!ok) { break; }
    cv::imshow(window, frame);

    Mat8uc1 gray;
    
    if (frame.channels() == 3) {
      cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
    } else {
      frame.copyTo(gray);
    }
    
    image_u8_t* im8 = cv2im8_copy(gray);
    
    zarray_t *detections = apriltag_detector_detect(td, im8);
    
    //printf("detected %d tags\n", zarray_size(detections));

    cv::Mat display = detectionsImage(detections, frame.size(), frame.type());
    //printf("size : %d\n", zarray_size(detections));
    double tx, ty, tz;
    for (int i = 0; i < zarray_size(detections); i++) {
      apriltag_detection_t *det;
      zarray_get(detections, i, &det);
      getRelativeTransform(
              29.5, //20mm
              28.0, //26mm
              28.0, //26mm
              gray.cols / 2.0,
              gray.rows / 2.0,
              (double*)det->p,
              &tx, &ty, &tz);

      //printf("detection %3d: id (%2dx%2d)-%-4d, hamming %d, "
      //       "goodness %8.3f, margin %8.3f\n",
      //       i, det->family->d*det->family->d, det->family->h,
      //       det->id, det->hamming, det->goodness, det->decision_margin);

      //matd_t* p = homography_to_pose(det->H, 26.0, 26.0, gray.cols/2.0, gray.rows/2.0);
      
      //double x = MATD_EL(p, 0,3);
      //double y = MATD_EL(p, 1,3);
      //double z = MATD_EL(p, 2,3);
      //
      //matd_destroy(p);
      printf("%d,%.9f,%.9f,%.9f|", det->id, tx,ty,tz);
    }

    printf("\n");

    apriltag_detections_destroy(detections);

    display = 0.5*display + 0.5*frame;
    cv::imshow(window, display);
    image_u8_destroy(im8);
    
    int k = cv::waitKey(1);
    if (k == 27) { break; }
    
  }
  
  return 0;

}
