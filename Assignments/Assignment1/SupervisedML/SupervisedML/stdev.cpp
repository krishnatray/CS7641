#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat mat2gray(const Mat& src)
{
    Mat dst;
    normalize(src, dst, 0.0, 1.0, NORM_MINMAX);
    return dst;
}

int main()
{
    Mat image = imread("coke-can.jpg", 0);

    Mat image32f;
    image.convertTo(image32f, CV_32F);

    Mat mu;
    blur(image32f, mu, Size(3, 3));

    Mat mu2;
    blur(image32f.mul(image32f), mu2, Size(3, 3));

    Mat sigma;
    cv::sqrt(mu2 - mu.mul(mu), sigma);

    imshow("coke", mat2gray(image32f));
    imshow("mu", mat2gray(mu));
    imshow("sigma",mat2gray(sigma));
    waitKey();
    return 0;
}