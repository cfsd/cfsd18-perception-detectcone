#include "tiny_dnn/tiny_dnn.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <string>

void CNN(const std::string& dictionary, tiny_dnn::network<tiny_dnn::sequential>& model) {
  using conv    = tiny_dnn::convolutional_layer;
  using fc      = tiny_dnn::fully_connected_layer;
  using tanh    = tiny_dnn::tanh_layer;
  using relu    = tiny_dnn::relu_layer;
  using softmax = tiny_dnn::softmax_layer;

  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

  model << conv(64, 64, 4, 3, 16, tiny_dnn::padding::valid, true, 2, 2, 1, 1, backend_type) << tanh()                                                   
        << conv(31, 31, 3, 16, 16, tiny_dnn::padding::valid, true, 2, 2, 1, 1, backend_type) << tanh() 
        << conv(15, 15, 3, 16, 32, tiny_dnn::padding::valid, true, 2, 2, 1, 1, backend_type) << tanh() 
        << conv(7, 7, 3, 32, 32, tiny_dnn::padding::valid, true, 2, 2, 1, 1, backend_type) << tanh()                    
        << fc(3 * 3 * 32, 128, true, backend_type) << relu()  
        << fc(128, 4, true, backend_type) << softmax(4); 

  std::ifstream ifs(dictionary.c_str());
  if (!ifs.good()){
    std::cout << "CNN model does not exist!" << std::endl;
    return;
  }
  ifs >> model;
}

void convertImage(cv::Mat img, int w, int h, tiny_dnn::vec_t& data){
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(w, h));
  data.resize(w * h * 3);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
       data[c * w * h + y * w + x] =
         float(resized.at<cv::Vec3b>(y, x)[c] / 255.0);
      }
    }
  }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: ./test-cnn [input-img-path] [model-name]" << std::endl;
        return 1;
    }

    tiny_dnn::network<tiny_dnn::sequential> model;
    CNN(argv[2], model);

    std::vector<tiny_dnn::tensor_t> inputs;
    tiny_dnn::vec_t data;
    cv::Mat patchImg = cv::imread(argv[1]);
    convertImage(patchImg, 64, 64, data);
    inputs.push_back({data});

    auto prob = model.predict(inputs);

    size_t maxIndex = 0;
    double maxProb = prob[0][0][0];
    for(size_t j = 1; j < 4; j++){
        if(prob[0][0][j] > maxProb){
            maxIndex = j;
            maxProb = prob[0][0][j];
        }
    }

    std::cout << "class (1-blue, 2-yellow, 3-orange): " << maxIndex << ", probability: " << maxProb << std::endl;
}