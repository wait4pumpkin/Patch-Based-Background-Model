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

#define N_FRAME_TRAIN 20

#define BLOCK_SIZE 8
#define N_CHANNEL 3
#define N_COEFF 4

#define T1 0.6
#define T2 0.4

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
    
    // Block Initialize
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
    
    // Variable Initialize
    size_t nBlock = blocks.size();
    vector<Mat> features;
    vector<EM> gmms;
    vector<int> gmmFlags(nBlock);
    vector<bool> flagsPrev(nBlock);
    vector<bool> flags(nBlock);
    for (int i=0; i<nBlock; ++i) {
        features.push_back(Mat::zeros(1, N_CHANNEL * N_COEFF, CV_64FC1));
        gmms.push_back(EM(2));
    }
    
    // Get Train Data
    int idxFrame = 0;
    vector<Mat> trainData;
    for (int i=0; i<nBlock; ++i) {
        trainData.push_back(Mat::zeros(N_FRAME_TRAIN, N_CHANNEL * N_COEFF, CV_64FC1));
    }
    for (; idxFrame < N_FRAME_TRAIN; ++idxFrame) {
        Mat frame;
        cap >> frame;
        
        Mat frameF = Mat::zeros(frame.size(), CV_64FC3);
        frame.convertTo(frameF, CV_64FC3);
        vector<Mat> channels;
        split(frameF, channels);
        
        // Feature Extracion
        for (int i=0; i<3; ++i) {
            for (int idx=0; idx<nBlock; ++idx) {
                Mat patch = channels[i](blocks[idx]);
                Mat dctFeature;
                dct(patch, dctFeature);
                
                Mat &feature = trainData[idx];
                feature.at<double>(idxFrame, i * N_COEFF) = dctFeature.at<double>(0, 0);
                feature.at<double>(idxFrame, i * N_COEFF + 1) = dctFeature.at<double>(0, 1);
                feature.at<double>(idxFrame, i * N_COEFF + 2) = dctFeature.at<double>(1, 0);
                feature.at<double>(idxFrame, i * N_COEFF + 3) = dctFeature.at<double>(1, 1);
            }
        }
    }
    
    // Train GMM
    for (int i=0; i<nBlock; ++i) {
        EM &gmm = gmms[i];
        Mat &feature = trainData[i];
        gmm.train(feature);
        
        Mat weight = gmm.getMat("weights");
        if (abs(weight.at<double>(0, 0) > weight.at<double>(0, 1)) > 0.5) {
            if (weight.at<double>(0, 0) > weight.at<double>(0, 1)) {
                gmmFlags[i] = 0;
            } else {
                gmmFlags[i] = 1;
            }
        } else {
            gmm.set("nclusters", 1);
            gmm.train(feature);
            gmmFlags[i] = 0;
        }
    }
    
    // Cleaning
    trainData.clear();

    int64 beginTime = getTickCount();
    for(; idxFrame < nFrame; ++idxFrame)
    {
        Mat frame;
        cap >> frame;
        imshow("Video", frame);
        
        Mat frameF = Mat::zeros(frame.size(), CV_64FC3);
        frame.convertTo(frameF, CV_64FC3);
        vector<Mat> channels;
        split(frameF, channels);
        
        // Feature Extracion
        for (int i=0; i<3; ++i) {
            for (int idx=0; idx<nBlock; ++idx) {
                Mat patch = channels[i](blocks[idx]);
                Mat dctFeature;
                dct(patch, dctFeature);
                
                Mat &feature = features[idx];
                feature.at<double>(0, i * N_COEFF) = dctFeature.at<double>(0, 0);
                feature.at<double>(0, i * N_COEFF + 1) = dctFeature.at<double>(0, 1);
                feature.at<double>(0, i * N_COEFF + 2) = dctFeature.at<double>(1, 0);
                feature.at<double>(0, i * N_COEFF + 3) = dctFeature.at<double>(1, 1);
            }
        }
        
        // Block Classify
        for (int i=0; i<nBlock; ++i) {
            bool isBackground = false;
            EM &gmm = gmms[i];
            Mat &feature = features[i];
            
            Vec2d probs = gmm.predict(feature);
            if (probs[gmmFlags[i]] > T1) {
                isBackground = true;
            }
            
            if (!isBackground) {
                Mat means = gmm.getMat("means");
                Mat similarity = 1 - feature * means.row(gmmFlags[i]).t() / (norm(feature) * norm(means));
                if (similarity.at<double>(0, 0) < T2) {
                    isBackground = true;
                } else if (flagsPrev[i] && similarity.at<double>(0, 0) < 0.5 * T2) {
                    isBackground = true;
                }
            }
            
            flags[i] = isBackground;
        }
        flagsPrev = flags;
        
        // Result Generating
        Mat result = Mat::zeros(height, width, CV_8UC1);
        for (int i=0; i<nBlock; ++i) {
            if (flags[i]) {
                rectangle(result, blocks[i], Scalar::all(255.), CV_FILLED);
            }
        }
        imshow("Result", result);

        if (waitKey(1) == 27) break;
    }
    
    int64 endTime = getTickCount();
    cout << "FPS: " << (nFrame - N_FRAME_TRAIN) / ((endTime - beginTime) / getTickFrequency()) << endl;
    
    return EXIT_SUCCESS;
}

