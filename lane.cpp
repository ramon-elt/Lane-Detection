/*  Teste para a detecção de caminho
 *  05.02  vesão 0.0
 *                                   */

#include <sstream>
#include <iostream>
#include <cstring>
#include <vector>
#include <stdio.h>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui.hpp"
//#include <opencv2/xfeatures2d.hpp>

#include <sys/time.h>

using namespace std;
using namespace cv;

#ifndef pi
const double pi = 3.14159265358979323846;
#endif

const char* windowName1 = "Image";
const char* windowName2 = "Detected Lanes";
const char* windowName3 = "Middle steps";



// Utility function to provide current system time (used below in
// determining frame rate at which images are being processed)
double tic() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return ((double)t.tv_sec + ((double)t.tv_usec)/1000000.);
}



class Lane{
    int im_width;                       // image size in pixels
    int im_height;                      // image size in pixels
    int im_deviceId; // camera id (in case of multiple cameras)
    cv::VideoCapture im_cap;                 // For USB cam only

  public:
    // Default constructor
    Lane():
    // Default settings
    im_width(640),
    im_height(480),
    im_deviceId(0)
    {}




    /* Calculates distances among points of a Matrix of points        */
    void Dist(const cv::Mat& P, cv::Mat& D){
        cv::Mat Aux1, Mt, M, D_i;
        // Matrix that calculates differences between elements
        cv::Mat T = Mat::ones(1, P.rows-1, CV_64F);
        cv::Mat T2 = Mat::eye(P.rows-1, P.rows-1, CV_64F);
        T2 = -T2;
        T.push_back(T2);
        cv::Mat T_t, T_b;

        // Loop to calculate differences Matrix (D_i)
        Mt = P.t()*T;
        M = Mt.t();
        Aux1 = M*M.t();
        sqrt(Aux1.diag(),D_i);
        // Appends D_i to D
        Aux1 = D_i.t();
        D.push_back(Aux1);
        for(int i=0; i<P.rows-2; i++){
            // Delocates vertically the vector T and copies to T_t
            T.rowRange(0,i).copyTo(T_t);
            T.rowRange(i+2,T.rows).copyTo(T_b);
            T_t.push_back(T.row(i+1));
            T_t.push_back(T.row(i));
            T_t.push_back(T_b);
            // Refreshes T
            T = T_t;
            // Calculates differences Matrix (D_i)
            Mt = P.t()*T;
            M = Mt.t();
            Aux1 = M*M.t();
            sqrt(Aux1.diag(),D_i);
            // Appends D_i to D
            Aux1 = D_i.t();
            D.push_back(Aux1);
        }

        Aux1 = D.t();
        D = Aux1;
    }// Dist()


    /* Creates windows and configures devices                         */
    void Setup(){
      // Prepare window for drawing the camera images
      cv::namedWindow(windowName1, 1);
      // Prepare window for drawing detected lanes
      cv::namedWindow(windowName2, 2);
      // Prepare window for drawing detected lanes
      cv::namedWindow(windowName3, 3);

      // Find and open a USB camera (built in laptop camera, web cam etc)
      im_cap = cv::VideoCapture(im_deviceId);
      if(!im_cap.isOpened()) {
          cerr << "ERROR: Can't find video device " << im_deviceId << "\n";
          exit(1);
      }
      im_cap.set(CAP_PROP_FRAME_WIDTH, im_width);
      im_cap.set(CAP_PROP_FRAME_HEIGHT, im_height);
      cout << "Camera successfully opened (ignore error messages above...)" << endl;
      cout << "Actual resolution: "
           << im_cap.get(CAP_PROP_FRAME_WIDTH) << "x"
           << im_cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
    }// Setup()


    /* Image is acquired, and processed, and lanes are detected       */
    void ProcessImage(const cv::Mat& image, cv::Mat& thres_image) {
        cv::Mat thres_image1, thres_image10, detected_edges;

        // Turn rgb image into a grayscale image------------------------
        cv::cvtColor(image, thres_image, COLOR_BGR2GRAY);
        cv::imshow(windowName1, image);                // Displays image
        // Threshold ---------------------------------------------------
        // Uncomment to chose between Ostsu and Gaussian threshold
        // Threshold Gaussian;
//        cv::adaptiveThreshold(image_gray, thres_image, 255, 0, 0, 5, 0);
        // Threshold Otsu;
        cv::threshold(thres_image, thres_image, 0, 255, 8);

        thres_image10 = thres_image;

        // Morphological operations in order to ignore great blobs -----
        int morph_size = 5;                   // must be an even integer
        cv::Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2*morph_size+1,2*morph_size+1),Point(morph_size, morph_size) );
        // first an openning (erode + dilate)
        cv::morphologyEx(thres_image, thres_image, 2, element);
        // now an closing (dilate + erode)
        cv::morphologyEx(thres_image, thres_image, 3, element);
        // Gaussian blur to avoid noise --------------------------------
        int kernel_size = 3;
        cv::GaussianBlur(thres_image, thres_image, Size(kernel_size,kernel_size),0,0,0);
        // Canny detector ----------------------------------------------
        int lowThreshold = 100;
        int ratio = 5;
        cv::Canny(thres_image, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
        //Hough Line Transform -----------------------------------------
        vector<Vec4i> lines;
        thres_image1 = Mat::zeros(thres_image.size(), image.type());
        cv::Mat thres_image2 = Mat::zeros(thres_image.size(), image.type());

        HoughLinesP(detected_edges, lines, 1, CV_PI/180, 20, 20, 20 );
        // Variables of size 'lines.size()'
        cv::Mat Xc = Mat::zeros( lines.size(), 2, CV_64F);
        cv::Mat X_perto = Mat::zeros( lines.size(), lines.size(), CV_64F);
        cv::Mat D;
        double m[lines.size()];

        // Loop to plot lines detected by Hough's algorithm
        for( size_t i = 0; i < lines.size(); i++ ){
          Vec4i l = lines[i];
          // Plot lines detected by Hough's algorithm
          line(thres_image1, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(00,0,255), 3, 8, 0);
          // In order to detect smooth curves calculates the center of lines and the slope(m)
          Xc.at<double>(i, 1) = min(l[2],l[0]) + abs(l[2] - l[0]);
          Xc.at<double>(i, 2) = min(l[3],l[1]) + abs(l[3] - l[1]);
          // To avoid indeterminations in slope
          if(((double)l[3] - (double)l[1])!=0){
              m[i] = ((double)l[2] - (double)l[0])/((double)l[3] - (double)l[1]);
          }else{
              if(((double)l[2] - (double)l[0])==0){
                  m[i] = -0.00001;
              }else{
                  m[i] = 1000;
              }
          }
          // Plot center points between detection
          circle(thres_image1, Point(Xc.at<double>(i, 1), Xc.at<double>(i, 2)), 10, Scalar(100,0,255), 1, 8, 0);
        }

        Dist(Xc,D);            // Calculates distance matrix
        // Tries to improve detection
        for( int i = 0; i < lines.size(); i++ ){
            int j=0;
            for( int ii = i; ii < lines.size()-1; ii++){
                if(D.at<double>(i,ii)<100){                   // If centers are close
                    if(abs(m[i]-m[ii])<0.2*m[i]){             // And slopes are similar
                        X_perto.at<double>(i,j+1) = ii + 1;    // ii+1 reffers to the element in Xc matrix
                        j++;
                    }//else{
//                        r[i][ii] = (Xc.at<double>(ii, 2) - Xc.at<double>(i, 2))/(Xc.at<double>(ii, 1) - Xc.at<double>(i, 1));
//                    }
//                }/*else{

                }
            }
            X_perto.at<double>(i,0) = j;                    // Saves number of similar points

cout <<"----------------X_perto "<< X_perto << endl;
        }

        for( int i = 0; i < lines.size(); i++ ){
            cv::Mat Aux;
            Aux.push_back(X_c.row(i));
            // Checks if there are points close to the points close to point i
            for( int ii = 1; ii <=X_perto.at<double>(i,0); ii++ ){
                Aux.push_back(X_c.row(X_perto.at<double>(i,ii)));
                if(X_perto.at<double>(ii,0)!=0){

                }
                // Plot lines detected by 'a roubada' algorithm
//                line(thres_image2, Point(px[i][ii], py[i][ii]), Point(px[i][ii+1], py[i][ii+1]), Scalar(100,0,255), 3, 8, 0);
            }
        }

        cv::imshow(windowName2, thres_image1);                // Displays detected lines
        cv::imshow(windowName3, thres_image10);                // Displays detected lines

    }// ProcessImage



    /* The processing loop where images are retrieved, lane detected,
       and information about detections generated                     */
    void Loop(){
        // Image Matrix
        cv::Mat image,image_gray;

        int frame = 0;
        double last_t = tic();
        while(true) {
            // Capture frame
            im_cap >> image;
            ProcessImage(image, image_gray);

            // Print out the frame rate at which image frames are being processed
            frame++;
            if (frame % 10 == 0) {
                double t = tic();
                cout << "  " << 10./(t-last_t) << " fps" << endl;
                last_t = t;
            }
            // Exit if any key is pressed
            if (cv::waitKey(1) >= 0)
                break;
        }
    } //Loop
}; //Lane


// Here is were everything begins
int main(int argc, char* argv[]) {
  Lane lane;

  lane.Setup();
  lane.Loop();

  return 0;
}

//Beizer fitting
//bwconncomp
