#include "face_ssd.h"
#include "dl_interface.h"
using namespace std;

FaceSSD::FaceSSD(){
    pDataIn = new float[input_size];
    pDataOut = new float[output_size];    
    img_shape = {96, 96};
    layers_shapes = {{12,12},{6,6},{3,3},{1,1}};
    anchor_scales = {{0.15},{0.3},{0.45},{0.6}};
    extra_anchor_scales = {{0.2},{0.4},{0.5},{0.6}};
    anchor_ratios = {{1.},{1.},{1.},{1.}};
    layer_steps = {8,16,32,96};
    prior_scaling={0.1, 0.1, 0.2, 0.2};
    total_anchor_num = 380;
    for (int layer = 0; layer < layers_shapes.size(); layer++)
    {
        int depth = anchor_scales[layer].size() * anchor_ratios[layer].size() + extra_anchor_scales[layer].size();
        int size = layers_shapes[layer][0] * layers_shapes[layer][1] * depth;
        layer_depth.push_back(depth);
        layer_size.push_back(size);
    }


}    

FaceSSD::~FaceSSD(){
    delete pDataIn;
    delete pDataOut;
    dl_delete(dl_model1);
}

void FaceSSD::init_model(const char * model_path)
{
    dl_model1 = dl_new("./face_ssd_model_anchor.tflite");

    if(dl_model1 == nullptr)
        printf("!!! ptr 1 is null !!!\n");
    else printf("!!! ptr 1 init OK !!!\n");


}
void FaceSSD::run_face_detection(unsigned char * image, int w, int h, vector<float> &score, vector<vector<int>> &boxes)
{

    img_width = w;
    img_height = h;
    preprocessing( image, w, h, pDataIn, 96, "bgr", "bgr");


    if(!dl_run(dl_model1, pDataIn, (int)input_size, pDataOut, output_size))
    {
        printf("run fail\n");
        // return -1;
    }
   
    vector<vector<float>> pred_loc;
    vector<float> pred_score;
    vector<vector<int>> pred_box;
    // select perdiction box with prob > 0.5
    select_anchor_pred(pred_score, pred_loc, 0.5);
    pred_to_box(pred_loc, pred_box);
    vector<int> nms_index = non_max_suppression(pred_box, pred_score, 10, 0.5);
	for (int i=0; i<nms_index.size(); i++)
	{
        score.push_back(pred_score[nms_index[i]]);
        boxes.push_back(pred_box[nms_index[i]]);
	}
}

void FaceSSD::preprocessing(unsigned char * image, int width, int height, float * pDataIn, int size, const char * input_format, const char * output_format)
{
    int ri=0,gi=1,bi=2,ro=0,go=1,bo=2;
    if (strcmp (input_format, "bgr")==0)
    {
        ri = 2; bi = 0;
    }
    if (strcmp (output_format, "bgr")==0)
    {
        ro = 2; bo = 0;
    }


    for(int y = 0; y < size; y++)
    {
        int j = int((float)y / (float)size * (float)height);
        for(int x = 0; x < size; x++)
        {
            
            int i = int((float)x / (float)size * (float)width);
            pDataIn[(y * size + x) * 3 + ro] = (float)image[(j * width + i) * 3 + ri] - _R_MEAN;
            pDataIn[(y * size + x) * 3 + go] = (float)image[(j * width + i) * 3 + gi] - _G_MEAN;
            pDataIn[(y * size + x) * 3 + bo] = (float)image[(j * width + i) * 3 + bi] - _B_MEAN;
        }
    }
    
}
vector<float> FaceSSD::decoder(int index, vector<float> pred_loc)
{
    int total_index = index + 1;
    int layer_index = 0;
    float offset = 0.5;
    while(total_index > layer_size[layer_index])
    {
        total_index-=layer_size[layer_index];
        layer_index++;
    }
    total_index--;
    int y_index =  total_index / layers_shapes[layer_index][1] / layer_depth[layer_index];
    int x_index =  (total_index / layer_depth[layer_index] ) % layers_shapes[layer_index][1] ;
    int ratio_index = total_index % layer_depth[layer_index];

    float anchor_cy = (y_index + offset) * layer_steps[layer_index] / img_shape[0];
    float anchor_cx = (x_index + offset) * layer_steps[layer_index] / img_shape[1];
    float anchor_h = 0;
    float anchor_w = 0;

    if( ratio_index < extra_anchor_scales[layer_index].size())
    {
        anchor_h = extra_anchor_scales[layer_index][ratio_index];
        anchor_w = extra_anchor_scales[layer_index][ratio_index];
    }
    else
    {
        int l_index = ratio_index- extra_anchor_scales[layer_index].size();
        anchor_h = anchor_scales[layer_index][l_index] / sqrtf(anchor_ratios[layer_index][l_index]);
        anchor_w = anchor_scales[layer_index][l_index] * sqrtf(anchor_ratios[layer_index][l_index]);
    }

    float pred_h = exp(pred_loc[2] * prior_scaling[2]) * anchor_h;
    float pred_w = exp(pred_loc[3] * prior_scaling[3]) * anchor_w;


    float pred_cy = pred_loc[0] * prior_scaling[0] * anchor_h + anchor_cy;
    float pred_cx = pred_loc[1] * prior_scaling[1] * anchor_w + anchor_cx;

    return {pred_cy, pred_cx, pred_h, pred_w};
}


vector<int> FaceSSD::sort_indexes(const vector<float> &scores) {

  // initialize original index locations
  vector<int> idx(scores.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&scores](size_t i1, size_t i2) {return scores[i1] > scores[i2];});

  return idx;
}

float FaceSSD::iou(vector<int> box1, vector<int> box2)
{

    int xs1 = max(box1[0], box2[0]);
    int ys1 = max(box1[1], box2[1]);
    int xs2 = min(box1[2], box2[2]);
    int ys2 = min(box1[3], box2[3]);
    int intersections = max(ys2-ys1, 0) * max(xs2-xs1, 0);
    int unions = (box1[2]-box1[0]) * (box1[3]-box1[1]) + 
                 (box2[2]-box2[0]) * (box2[3]-box2[1]) - intersections;

    return (float)intersections / (float)unions;

}

vector<int> FaceSSD::non_max_suppression(vector<vector<int>> boxes, vector<float> scores, int nms_topk, float threshold)
{
    // if (boxes.size() != scores.size())
    //     return vector<int> keep_index;
    vector<int> areas;
    for (int b = 0; b < scores.size(); ++b)
    {
        areas.push_back((boxes[b][2]-boxes[b][0])*(boxes[b][3]-boxes[b][1]));

    }
    
    vector<int> sort_score_index = sort_indexes((const vector<float>)scores);
    vector<int> keep_index;
    while(sort_score_index.size()>0)
    {
        int index = sort_score_index.front();
        sort_score_index.erase(sort_score_index.begin());
        keep_index.push_back(index);

        if(sort_score_index.size() == 0)
            break;


        vector<int> erase_list;
        for(int idx = 0; idx < sort_score_index.size(); ++idx)
        {
            if (iou(boxes[index], boxes[sort_score_index[idx]]) > threshold)
            {
                erase_list.push_back(idx);
            }
        }

        while(erase_list.size() > 0)
        {
            sort_score_index.erase(sort_score_index.begin() + erase_list.back());
            erase_list.pop_back();
        }


    }

    return keep_index;

}

void FaceSSD::select_anchor_pred(vector<float> &pred_score, vector<vector<float>> &pred_loc, float threshold)
{
    int box_cnt = 0;
    for(int i = 0; i < total_anchor_num; ++i) {
        float score ;
        if (pDataOut[i * 6 ] > pDataOut[i * 6 + 1])
            {continue;}
        score = exp(pDataOut[i * 6 + 1]) / (exp(pDataOut[i * 6 + 1]) + exp(pDataOut[i * 6 ]));
        if (score < threshold)
            {continue;}

        pred_loc.push_back(decoder(i, {pDataOut[i * 6 + 2], 
            pDataOut[i * 6 + 3], pDataOut[i * 6 + 4], pDataOut[i * 6 + 5]}));
        pred_score.push_back(score);
        box_cnt++;
    }
}

void FaceSSD::pred_to_box(vector<vector<float>> &pred_loc, vector<vector<int>> &pred_box)
{

    for(int i = 0; i< pred_loc.size(); ++i)
    {
        vector<float> loc = pred_loc[i];
        float cy = loc[0] * (float)img_height;
        float cx = loc[1] * (float)img_width;
        float h = loc[2] * (float)img_height;
        float w = loc[3] * (float)img_width;
        int x1 = int(cx - w/2);
        int y1 = int(cy - h/2);
        int x2 = int(cx + w/2);
        int y2 = int(cy + h/2);
        pred_box.push_back({x1, y1, x2, y2});
    }
   
}

