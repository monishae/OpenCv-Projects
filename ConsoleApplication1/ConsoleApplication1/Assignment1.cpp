/**
Assignment_1
mxe160530

**/

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

double alpha =2.0;
int beta = 50;
float  h,s,v,h_new;

void check_program_arguments(int argc) {
	if(argc != 2) {
		cout << "Error! Program usage:" << endl;
		cout << "./fruits.jpg path" << endl;
		exit(-1);
	}	
}

void check_if_image_exist(const Mat &img, const string &path) {
	if(img.empty()) {
		cout << "Error! Unable to load image: " << path << endl;
		exit(-1);
	}	
}
//Increase the brightness with the mathematical function, g(x) = a.f(x) + b;
void colorBrightness(Mat image){
	Mat new_image = Mat :: zeros(image.size(),image.type());

	for(int y=0;y< image.rows;y++){
				for(int x=0;x<image.cols;x++){
	
					for(int c=0;c<3;c++){
						new_image.at<Vec3b>(y,x)[c] =saturate_cast<uchar>(alpha * (image.at<Vec3b>(y,x)[c])+beta);
					}
				}
			}

	namedWindow("Original Image",1);
	namedWindow("Question_1: Increase Brightness",1);
	
	imshow("Original Image",image);
	imshow("Question_1: Increase Brightness",new_image);

	waitKey();
	}

void convertRGBtoHSV(Mat image_rgb){
	Mat image_hsv;
	float min,max,delta,r,g,b;
	for(int y=0;y< image_rgb.rows;y++){
		for(int x=0;x<image_rgb.cols;x++){
			r = image_rgb.at<Vec3b>(y,x)[2];
			g = image_rgb.at<Vec3b>(y,x)[1];
			b = image_rgb.at<Vec3b>(y,x)[0];
			min = MIN(r,g,b);
			max= MAX(r,g,b);
			v = max;
			delta = max-min;
				if(max!=0)
					s = delta/max;
					else{
						s=0;
						h=-1;
						break;
						}
					if( r == max )
						
						h_new =  int(( g - b ) / delta) % 6;	
					else if( g == max )

						h_new = 2 + ( b - r) / delta;
					else
						h_new = 4 + ( r- g ) / delta;
						h = 60  * h_new;				// degrees
				image_rgb.at<Vec3b>(y,x)[2] =uchar(h);
				image_rgb.at<Vec3b>(y,x)[1] = uchar(s);
				image_rgb.at<Vec3b>(y,x)[0]= uchar(v);
	}
}

	imshow("Question_2: RGB to HSV",image_rgb);
	waitKey();
}

void detectObject(Mat bgr_image){

	Mat orig_image = bgr_image.clone();

	medianBlur(bgr_image, bgr_image, 3);

	// Convert input image to HSV
	Mat hsv_image;
	cvtColor(bgr_image, hsv_image, COLOR_BGR2HSV);

	// Threshold the HSV image, keep only the red pixels
	Mat lower_red_hue_range;
	Mat upper_red_hue_range;
	inRange(hsv_image, Scalar(0, 100, 100), Scalar(10, 255, 255), lower_red_hue_range);
	inRange(hsv_image, Scalar(160, 100, 100), Scalar(179, 255, 255), upper_red_hue_range);

	// Combine the above two images
	Mat red_hue_image;
	addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);

	GaussianBlur(red_hue_image, red_hue_image, Size(9, 9), 2, 2);

	// Use the Hough transform to detect circles in the combined threshold image
	vector<Vec3f> circles1;
	HoughCircles(red_hue_image, circles1, CV_HOUGH_GRADIENT, 1, red_hue_image.rows/8, 120, 30, 0, 0);

	// Loop over all detected circles and outline them on the original image
	if(circles1.size() == 0) exit(-1);
	for(size_t current_circle = 0; current_circle < circles1.size(); ++current_circle) {
		Point center(ceil(circles1[current_circle][0]), ceil(circles1[current_circle][1]));
		int radius = ceil(circles1[current_circle][2]);

		circle(orig_image, center, radius*2, Scalar(0, 255, 0),5);
		circle(red_hue_image, center, radius*2, Scalar(0, 255, 0),5);
	}
	// Show images
	namedWindow("Threshold lower image", WINDOW_AUTOSIZE);
	imshow("Threshold lower image", lower_red_hue_range);
	namedWindow("Detected red circles on the input image", WINDOW_AUTOSIZE);
	imshow("Detected red circles on the input image", orig_image);

	waitKey(0);
}

void colorChange(Mat original){

	int w = original.size().width;
	int h = original.size().height;
   
	
	for(int y=0;y< original.rows;y++){
		for(int x=0;x<original.cols;x++){
	
			
				original.at<Vec3b>(y,x)[0]=original.at<Vec3b>(y *w +x)[0];
				original.at<Vec3b>(y,x)[1]=original.at<Vec3b>(y *w +x)[2];
				original.at<Vec3b>(y,x)[2]=171;
			
		}
	}
	
    namedWindow( "Original image", CV_WINDOW_AUTOSIZE ); 
    imshow( "Question_4: Color Change", original ); 
	waitKey(0);                        

}
int main(int argc,char** argv)
{
	Mat image = imread(argv[1]);
	image = imread(argv[1],IMREAD_COLOR);
	check_program_arguments(argc);
	
		// Check if the image can be loaded
	check_if_image_exist(image, argv[1]);

	Mat colorImg= image.clone();
	Mat hsvImage = image.clone();
	Mat objectImage = image.clone();

	colorBrightness(image);
	convertRGBtoHSV(hsvImage);
	detectObject(objectImage);
	colorChange(colorImg);

}

