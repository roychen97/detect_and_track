#ifndef face_ssd_h
#define face_ssd_h


#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory>
#include <vector>
#include <numeric>

#define _R_MEAN 123.68
#define _G_MEAN 116.78
#define _B_MEAN 103.94

using namespace std;

class FaceSSD
{
public:
    FaceSSD();  
    ~FaceSSD();
    void init_model(const char * model_path);
    void run_face_detection(unsigned char * image, int w, int h, vector<float> &score, vector<vector<int>> &boxes);

private:
    const int input_size = 96*96*3;
    const int output_size = 380*6;
    int img_width;
    int img_height;
    float *pDataIn;
    float *pDataOut;
    int total_anchor_num;
    vector<int> img_shape;
    vector<vector<int>> layers_shapes;
    vector<vector<float>> anchor_scales;
    vector<vector<float>> extra_anchor_scales;
    vector<vector<float>> anchor_ratios;
    vector<int> layer_steps;
    vector<int> layer_size;
    vector<int> layer_depth;
    vector<float> prior_scaling;
    // float anchor_offset = 0.5 * layers_shapes.size();
    void* dl_model1;
    void preprocessing(unsigned char * image, int width, int height, float * pDataIn, int size, const char * input_format, const char * output_format);
    void select_anchor_pred(vector<float> &pred_score, vector<vector<float>> &pred_loc, float threshold);
    void pred_to_box(vector<vector<float>> &pred_loc, vector<vector<int>> &pred_box);
    vector<float> decoder(int index, vector<float> pred_loc);
    vector<int> non_max_suppression(vector<vector<int>> boxes, vector<float> scores, int nms_topk, float threshold);
    float iou(vector<int> box1, vector<int> box2);
    vector<int> sort_indexes(const vector<float> &scores) ;    
};
#endif