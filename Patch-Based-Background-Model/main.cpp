//
//  main.cpp
//  Patch-Based-Background-Model
//
//  Created by Pumpkin Lee on 1/6/14.
//  Copyright (c) 2014 Pumpkin Lee. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define SAMPLE_FILE "/Users/pumpkin/Documents/project/Patch-Based Background Model/sample/320x240.mov"

#define BLOCK_SIZE 8
#define N_CHANNEL 3
#define N_COEFF 4

#define T1 0.5
#define T2 0.5

int main(int argc, const char * argv[])
{
    namedWindow("Video");
    
    VideoCapture cap(SAMPLE_FILE);
    if (!cap.isOpened()) {
        cerr << "Sample Not Found." << endl;
        return EXIT_FAILURE;
    }
    
    int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    int nFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);
    
    vector<Rect> blocks;
    for (int r=0; r<height; r+=BLOCK_SIZE) {
        for (int c=0; c<width; c+=BLOCK_SIZE) {
            Rect block;
            block.width = BLOCK_SIZE;
            block.height = BLOCK_SIZE;
            block.x = c;
            block.y = r;
            blocks.push_back(block);
        }
    }
    
    size_t nBlock = blocks.size();
    vector<vector<int>> features;
    vector<EM> gmms;
    vector<bool> flags(nBlock);
    for (int i=0; i<nBlock; ++i) {
        features.push_back(vector<int>(N_CHANNEL * N_COEFF));
        gmms.push_back(EM(1));
    }

    int64 beginTime = getTickCount();
    for(int idx = 0; idx < nFrame; ++idx)
    {
        Mat frame;
        cap >> frame;
        imshow("Video", frame);
        
        Mat frameF = Mat::zeros(frame.size(), CV_32FC3);
        frame.convertTo(frameF, CV_32FC3);
        vector<Mat> channels;
        split(frameF, channels);
        
        // Feature Extracion
        for (int i=0; i<3; ++i) {
            for (int idx=0; idx<nBlock; ++idx) {
                Mat patch = channels[i](blocks[idx]);
                Mat dctFeature;
                dct(patch, dctFeature);
                
                vector<int> &feature = features[idx];
                feature[i * N_COEFF] = dctFeature.at<float>(0, 0);
                feature[i * N_COEFF + 1] = dctFeature.at<float>(0, 1);
                feature[i * N_COEFF + 2] = dctFeature.at<float>(1, 0);
                feature[i * N_COEFF + 3] = dctFeature.at<float>(1, 1);
            }
        }
        
        // Block Classify
        for (<#initialization#>; <#condition#>; <#increment#>) {
            <#statements#>
        }
        
//        imshow("Demo", output);
//        waitKey();

        if (waitKey(1) == 27) break;
    }
    
    int64 endTime = getTickCount();
    cout << "FPS: " << nFrame / ((endTime - beginTime) / getTickFrequency()) << endl;
    
    return EXIT_SUCCESS;
}

