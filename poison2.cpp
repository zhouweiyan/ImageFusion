// Ex1: Poisson Blending
// Ax=b; 
// A is the poisson matrix.
// x is the vector that represents value of each pixel of roi, whose length is roi.height*roi.width.
// b is the vector that contains divergence term and constraint condition.
// Calculate A,b,respectively. Then the equation Ax = b is calculated.
// zhouweiyan	20180517

#include<opencv2/opencv.hpp>
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

#define elif else if
#define ATD at<double>


// positive-going calculation of the horizontal gradients of an image, i.e., img(i, j + 1) - img(i, j)
// img: the image that you want to calculate its gradients
Mat getGradientXp(Mat &img)
{
	int height = img.rows;
	int width = img.cols;
	Mat casimg = repeat(img, 1, 2);	// repeat the input img for two times, they are horizontally arranged.

	Rect roi = Rect(1, 0, width, height);
	Mat roimat = casimg(roi);		// roimat is the version of img that the first column is in the last one, the other columns move one column forward.
	return roimat - img;			// img(i, j + 1) - img(i, j)
}

//  positive-going calculation of the vertical gradients of an image, i.e., img(i + 1, j) - img(i, j)
// img: the image that you want to calculate its gradients
Mat getGradientYp(Mat &img)
{
	int height = img.rows;
	int width = img.cols;
	Mat casimg = repeat(img, 2, 1);	// repeat the input img for two times, they are vertically arranged.

	Rect roi = Rect(0, 1, width, height);
	Mat roimat = casimg(roi);		// roimat is the version of img that the first row is in the last one, the other rows move one row forward.
	return roimat - img;			// img(i + 1, j) - img(i, j)
}

// negative-going calculation of the horizontal gradients of an image, i.e., img(i, j - 1) - img(i, j)
// img: the image that you want to calculate its gradients
Mat getGradientXn(Mat &img){
	int height = img.rows;
	int width = img.cols;
	Mat casimg = repeat(img, 1, 2);	// repeat the input img for two times, they are horizontally arranged.

	Rect roi = Rect(width - 1, 0, width, height);
	Mat roimat = casimg(roi);		// roimat is the version of img that the last column is in the first one, the other columns move one column backward.
	return roimat - img;			// img(i, j - 1) - img(i, j)
}

// negative-going calculation of the vertical gradients of an image, i.e., img(i - 1, j) - img(i, j)
// img: the image that you want to calculate its gradients
Mat getGradientYn(Mat &img)
{
	int height = img.rows;
	int width = img.cols;
	Mat casimg = repeat(img, 2, 1);	// repeat the input img for two times, they are vertically arranged.

	Rect roi = Rect(0, height - 1, width, height);
	Mat roimat = casimg(roi);		// roimat is the version of img that the last row is in the first one, the other rows move one row backward.
	return roimat - img;			// img(i - 1, j) - img(i, j)
}

int getLabel(int i, int j, int height, int width)
{
	return i * width + j;			// i is in height axis; j is in width axis.
}

// get Matrix A
Mat getA(int height, int width)
{
	Mat A = Mat::eye(height * width, height * width, CV_64FC1);
	A *= -4;

	// M: the label matrix of roi from im2; divede M into three parts, 0,1,2, respectively.
	// 0: the corners of roi
	// 1: boundaries but not corners
	// 2: inner part of roi
	// different parts represent different methods to asssign A values
	Mat M = Mat::zeros(height, width, CV_64FC1);
	Mat temp = Mat::ones(height, width - 2, CV_64FC1);
	Rect roi = Rect(1, 0, width - 2, height);
	Mat roimat = M(roi);
	temp.copyTo(roimat);
	temp = Mat::ones(height - 2, width, CV_64FC1);
	roi = Rect(0, 1, width, height - 2);
	roimat = M(roi);
	temp.copyTo(roimat);
	temp = Mat::ones(height - 2, width - 2, CV_64FC1);
	temp *= 2;
	roi = Rect(1, 1, width - 2, height - 2);
	roimat = M(roi);
	temp.copyTo(roimat);

	// get Matrix A
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			// get index of M(i,j)
			int label = getLabel(i, j, height, width);

			// when M(i,j) is the corner, assign coeffients to A
			if (M.ATD(i, j) == 0)
			{
				if (i == 0)  A.ATD(getLabel(i + 1, j, height, width), label) = 1;				// left upper
				elif(i == height - 1)   A.ATD(getLabel(i - 1, j, height, width), label) = 1;	// right upper
				if (j == 0)  A.ATD(getLabel(i, j + 1, height, width), label) = 1;				// left lower
				elif(j == width - 1)   A.ATD(getLabel(i, j - 1, height, width), label) = 1;		// right lower
			}
			// when M(i,j) is the boundary but not corner, assign coeffients to A
			elif(M.ATD(i, j) == 1)
			{
				if (i == 0){
					A.ATD(getLabel(i + 1, j, height, width), label) = 1;
					A.ATD(getLabel(i, j - 1, height, width), label) = 1;
					A.ATD(getLabel(i, j + 1, height, width), label) = 1;
				}elif(i == height - 1){
					A.ATD(getLabel(i - 1, j, height, width), label) = 1;
					A.ATD(getLabel(i, j - 1, height, width), label) = 1;
					A.ATD(getLabel(i, j + 1, height, width), label) = 1;
				}
				if (j == 0){
					A.ATD(getLabel(i, j + 1, height, width), label) = 1;
					A.ATD(getLabel(i - 1, j, height, width), label) = 1;
					A.ATD(getLabel(i + 1, j, height, width), label) = 1;
				}elif(j == width - 1){
					A.ATD(getLabel(i, j - 1, height, width), label) = 1;
					A.ATD(getLabel(i - 1, j, height, width), label) = 1;
					A.ATD(getLabel(i + 1, j, height, width), label) = 1;
				}
			}
			// when M(i,j) is the inner point, assign coeffients to A
			else{
				A.ATD(getLabel(i, j - 1, height, width), label) = 1;
				A.ATD(getLabel(i, j + 1, height, width), label) = 1;
				A.ATD(getLabel(i - 1, j, height, width), label) = 1;
				A.ATD(getLabel(i + 1, j, height, width), label) = 1;
			}
		}
	}
	return A;
}

// Calculate b
// using getGradient functions.
Mat getB2(Mat &img1, Mat &img2, int posX, int posY, Rect ROI)
{
	// calculate the divergence of img1
	Mat MergeGradXp = getGradientXp(img1);
	Mat GradXp2 = getGradientXp(img2);
	Mat MergeGradXpROI = MergeGradXp(ROI);
	GradXp2.copyTo(MergeGradXpROI);

	Mat MergeGradXn = getGradientXn(img1);
	Mat GradXn2 = getGradientXn(img2);
	Mat MergeGradXnROI = MergeGradXn(ROI);
	GradXn2.copyTo(MergeGradXnROI);

	Mat MergeGradYp = getGradientYp(img1);
	Mat GradYp2 = getGradientYp(img2);
	Mat MergeGradYpROI = MergeGradYp(ROI);
	GradYp2.copyTo(MergeGradYpROI);

	Mat MergeGradYn = getGradientYn(img1);
	Mat GradYn2 = getGradientYn(img2);
	Mat MergeGradYnROI = MergeGradYn(ROI);
	GradYn2.copyTo(MergeGradYnROI);

	Mat grad = MergeGradXp + MergeGradXn + MergeGradYp + MergeGradYn;

	int roiheight = ROI.height;
	int roiwidth = ROI.width;
	Mat B = Mat::zeros(roiheight * roiwidth, 1, CV_64FC1);
	for (int i = 0; i < roiheight; i++){
		for (int j = 0; j < roiwidth; j++){
			double temp = 0.0;
			temp += grad.ATD(i + ROI.y, j + ROI.x);
			if (i == 0)              temp -= img2.ATD(i - 1 + posY, j + posX);
			if (i == roiheight - 1)  temp -= img2.ATD(i + 1 + posY, j + posX);
			if (j == 0)              temp -= img2.ATD(i + posY, j - 1 + posX);
			if (j == roiwidth - 1)   temp -= img2.ATD(i + posY, j + 1 + posX);
			B.ATD(getLabel(i, j, roiheight, roiwidth), 0) = temp;
		}
	}
	return B;
}

// Solve equation and reshape it back to the right height and width.
Mat
getResult(Mat &A, Mat &B, Rect &ROI){
	Mat result;
	solve(A, B, result);
	result = result.reshape(0, ROI.height);
	return  result;
}



// img1: 3-channel image, we wanna move something in it into img2.
// img2: 3-channel image, dst image.
// ROI: the position and size of the block we want to move in img1.
// posX, posY: where we want to move the block to in img2

Mat poisson_blending(Mat &img1, Mat &img2, Rect ROI, int posX, int posY)
{
	int roiheight = ROI.height;
	int roiwidth = ROI.width;
	Mat A = getA(roiheight, roiwidth);

	// we must do the poisson blending to each channel.
	vector<Mat> rgb1;
	split(img1, rgb1);
	vector<Mat> rgb2;
	split(img2, rgb2);

	vector<Mat> result;
	Mat merged, res, Br, Bg, Bb;

	Br = getB2(rgb1[0], rgb2[0], posX, posY, ROI);
	res = getResult(A, Br, ROI);
	result.push_back(res);
	cout << "R channel finished..." << endl;
	Bg = getB2(rgb1[1], rgb2[1], posX, posY, ROI);
	res = getResult(A, Bg, ROI);
	result.push_back(res);
	cout << "G channel finished..." << endl;
	Bb = getB2(rgb1[2], rgb2[2], posX, posY, ROI);
	res = getResult(A, Bb, ROI);
	result.push_back(res);
	cout << "B channel finished..." << endl;

	// merge the 3 gray images into a 3-channel image 
	merge(result, merged);
	return merged;
}

int main(int argc, char** argv)
{
	Mat img1, img2;

	Mat in1 = imread("sourceV4.jpg");
	Mat in2 = imread("destinationsV3.jpg");
	imshow("src", in1);
	imshow("dst", in2);
	in1.convertTo(img1, CV_64FC3);
	in2.convertTo(img2, CV_64FC3);

	Rect rc = Rect(35, 53, 70, 68);//the part in im2 that we wanna move into im1

	// img1: 3-channel image, we wanna move something in it into img2.
	// img2: 3-channel image, dst image.
	// ROI: the position and size of the block we want to move in img1.
	// posX, posY: where we want to move the block to in img2
	Mat result = poisson_blending(img1, img2, rc, 70, 60);
	result.convertTo(result, CV_8UC1);
	Rect rc2 = Rect(70, 60, 70, 68);
	Mat roimat = in2(rc2);
	result.copyTo(roimat);

	imshow("roi", result);
	imshow("result", in2);
	imwrite("resultV3_V4.jpg", in2);
	waitKey(0);
	return 0;
}