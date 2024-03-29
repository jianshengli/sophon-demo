//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "json.hpp"
#include "opencv2/opencv.hpp"
#include "ff_decode.hpp"
#include "yolov5.hpp"
using json = nlohmann::json;
using namespace std;
#define WITH_ENCODE 1
int main(int argc, char *argv[]){
  cout.setf(ios::fixed);
  // get params
  const char *keys="{bmodel | ../../models/BM1684/yolov5s_v6.1_3output_fp32_1b.bmodel | bmodel file path}"
    "{dev_id | 0 | TPU device id}"
    "{conf_thresh | 0.001 | confidence threshold for filter boxes}"
    "{nms_thresh | 0.6 | iou threshold for nms}"
    "{help | 0 | print help information.}"
    "{input | ../../datasets/test | input path, images direction or video file path}"
    "{classnames | ../../datasets/coco.names | class names file path}"
    "{use_cpu_opt | false | accelerate cpu postprocess}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }
  string bmodel_file = parser.get<string>("bmodel");
  string input = parser.get<string>("input");
  int dev_id = parser.get<int>("dev_id");
  bool use_cpu_opt = parser.get<bool>("use_cpu_opt");

  // check params
  struct stat info;
  if (stat(bmodel_file.c_str(), &info) != 0) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }
  string coco_names = parser.get<string>("classnames");
  if (stat(coco_names.c_str(), &info) != 0) {
    cout << "Cannot find classnames file." << endl;
    exit(1);
  }
  if (stat(input.c_str(), &info) != 0){
    cout << "Cannot find input path." << endl;
    exit(1);
  }

  // creat handle
  BMNNHandlePtr handle = make_shared<BMNNHandle>(dev_id);
  cout << "set device id: "  << dev_id << endl;
  bm_handle_t h = handle->handle();

  // load bmodel
  shared_ptr<BMNNContext> bm_ctx = make_shared<BMNNContext>(handle, bmodel_file.c_str());

  // initialize net
  YoloV5 yolov5(bm_ctx, use_cpu_opt);
  CV_Assert(0 == yolov5.Init(
        parser.get<float>("conf_thresh"),
        parser.get<float>("nms_thresh"),
        coco_names));

  // profiling
  TimeStamp yolov5_ts;
  TimeStamp *ts = &yolov5_ts;
  yolov5.enableProfile(&yolov5_ts);

  // get batch_size
  int batch_size = yolov5.batch_size();

  // creat save path
  if (access("results", 0) != F_OK)
    mkdir("results", S_IRWXU);
  if (access("results/images", 0) != F_OK)
    mkdir("results/images", S_IRWXU);
  
    // get files
    vector<string> files_vector;
    files_vector.push_back(input);
    std::sort(files_vector.begin(), files_vector.end());

    vector<bm_image> batch_imgs;
    vector<string> batch_names;
    vector<YoloV5BoxVec> boxes;
    vector<json> results_json;
    int cn = files_vector.size();
    int id = 0;
    for (vector<string>::iterator iter = files_vector.begin(); iter != files_vector.end(); iter++){
      string img_file = *iter; 
       auto  img = cv::imread(img_file.c_str());
       cout<<"开始推理前,img.cols="<<img.cols<<",img.rows="<<img.rows<<std::endl;
      yolov5.Detect(img,boxes);
       vector<json> bboxes_json;
          for (auto bbox1 : boxes) {
            for(auto bbox:bbox1){
    #if 1
                cout << "  class id=" << bbox.class_id << ", score = " << bbox.score << " (x=" << bbox.x << ",y=" << bbox.y << ",w=" << bbox.width << ",h=" << bbox.height << ")" << endl;
    #endif
    //             // draw image
    //               yolov5.draw_bmcv(h, bbox.class_id, bbox.score, bbox.x, bbox.y, bbox.width, bbox.height, batch_imgs[i]);

    //             // save result

                json bbox_json;
                bbox_json["category_id"] = bbox.class_id;
                bbox_json["score"] = bbox.score;
                bbox_json["bbox"] = {bbox.x, bbox.y, bbox.width, bbox.height};
                bboxes_json.push_back(bbox_json);
            }

          }
          json res_json;
          // res_json["image_name"] = batch_names[i];
          res_json["bboxes"] = bboxes_json;
          results_json.push_back(res_json);      


    }
    
    // save results
    size_t index = input.rfind("/");
    if(index == input.length() - 1){
      input = input.substr(0, input.length() - 1);
      index = input.rfind("/");
    }
    string dataset_name = input.substr(index + 1);
    index = bmodel_file.rfind("/");
    string model_name = bmodel_file.substr(index + 1);
    string json_file = "results/" + model_name + "_" + dataset_name + "_bmcv_cpp" + "_result.json";
    cout << "================" << endl;
    cout << "result saved in " << json_file << endl;
    ofstream(json_file) << std::setw(4) << results_json;


  return 0;
}
