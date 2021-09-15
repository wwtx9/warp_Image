#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

using namespace std;
void computeHomography(const vector<vector<cv::Point2f>> &inliers, cv::Mat &H_l, cv::Mat &H_r)
{
    int num_points = inliers.size();
    cv::Mat A_l = cv::Mat::ones(num_points, 2, CV_32F);
    cv::Mat A_r = cv::Mat::ones(num_points, 2, CV_32F);
    cv::Mat b_l = cv::Mat::ones(num_points, 1, CV_32F);
    cv::Mat b_r = cv::Mat::ones(num_points, 1, CV_32F);
    cv::Mat x_l, x_r;

    float y_avg, y_left, y_right;
    for (unsigned int i = 0; i < num_points; i++){
        y_left = inliers[i][0].y;
        y_right = inliers[i][1].y;
        y_avg = (y_left + y_right)/2.0;

        A_l.at<float>(i,0) = inliers[i][0].y;
        b_l.at<float>(i,0) = y_avg;

        A_r.at<float>(i,0) = inliers[i][1].y;
        b_r.at<float>(i,0) = y_avg;
    }
    solve(A_l, b_l, x_l, cv::DECOMP_SVD);
    solve(A_r, b_r, x_r, cv::DECOMP_SVD);

    //build homography matrices
    H_l = cv::Mat::zeros(3,3, CV_32F);
    H_r = cv::Mat::zeros(3,3, CV_32F);
    H_l.at<float>(0,0) = 1;
    H_r.at<float>(0,0) = 1;
    H_l.at<float>(2,2) = 1;
    H_r.at<float>(2,2) = 1;

    H_l.at<float>(1,1) = x_l.at<float>(0,0);
    H_l.at<float>(1,2) = x_l.at<float>(1,0);

    H_r.at<float>(1,1) = x_r.at<float>(0,0);
    H_r.at<float>(1,2) = x_r.at<float>(1,0);

    cout<<"homography matrix for left image: "<<H_l<<endl;
    cout<<"homography matrix for right image: "<<H_r<<endl;
}

int main(int argc, char **argv) {
    cv::Mat H_l, H_r;
    vector<string> filenames{"1532199773957276199","1532199774006098099", "1532199774054927299", "1532199774154619199",
                             "1532199774254377099","1532199774354066499", "1532199774402911899","1532199774453827299",
                             "1532199774502676199","1532199774553546399","1532199774602401899","1532199774651256599",
                             "1532199774702130599","1532199774750985699"};
    string data_path = argv[1];
    //rectify image
    for(int ni  = 0; ni < (int)filenames.size(); ++ni)
    {
        string left_path = data_path + "left_origin/" + filenames[ni]+".png";
        cv::Mat imLeft = cv::imread(left_path,-1);


        string right_path = data_path + "right_origin/" + filenames[ni]+".png";
        cv::Mat imRight = cv::imread(right_path,-1);

        if(ni == 0)
        {
            //get homography matrix
            vector<vector<cv::Point2f> > inliners(4, vector<cv::Point2f>(2));
            inliners[0][0] = cv::Point2f(230.0, 256.0);
            inliners[0][1] = cv::Point2f(150.0, 259.0);

            inliners[1][0] = cv::Point2f(270.0, 269.0);
            inliners[1][1] = cv::Point2f(290.0, 272.0);

            inliners[2][0] = cv::Point2f(189.0, 342.0);
            inliners[2][1] = cv::Point2f(105.0, 345.0);

            inliners[3][0] = cv::Point2f(231.0, 356.0);
            inliners[3][1] = cv::Point2f(147.0, 359.0);

            computeHomography(inliners, H_l, H_r);
        }


        cv::Mat warpLeftIm, warpRightIm;
        cv::Size dsize = imLeft.size();

        warpPerspective(imLeft, warpLeftIm, H_l, dsize, cv::INTER_LINEAR , cv::BORDER_CONSTANT, 0); //+ WARP_INVERSE_MAP
        warpPerspective(imRight, warpRightIm, H_r, dsize, cv::INTER_LINEAR , cv::BORDER_CONSTANT, 0); //+ WARP_INVERSE_MAP

        imwrite("/home/wangweihan/Documents/my_project/underwater_project/dataset/800_600/left/"+filenames[ni]+".png", warpLeftIm);
        imwrite("/home/wangweihan/Documents/my_project/underwater_project/dataset/800_600/right/"+filenames[ni]+".png", warpRightIm);

    }
    return 0;
}
