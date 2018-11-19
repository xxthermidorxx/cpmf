#ifndef CPMF_CONFIG_HPP_
#define CPMF_CONFIG_HPP_

#include <string>

namespace cpmf {

struct BaseParams {
  BaseParams() : max_iter(10), num_threads(4), calc_rmse(true) {};

  int max_iter, num_threads;
  bool calc_rmse;
};

struct ModelParams {
  ModelParams() :
    read_model(false), write_model(true),
    dim(40), lp(1), lq(1), step_size(0.005), output_path("./model.bin") {}

  bool read_model, write_model;
  int dim;
  float lp, lq, step_size;
  std::string output_path;
};

struct DataParams {
  DataParams() :
    randomize(false), num_user_blocks(1), num_item_blocks(1),
    input_path("./input/testdata"), output_path("./output/testdata.bin") {}

  bool randomize;
  int num_user_blocks, num_item_blocks;
  std::string input_path, test_path, output_path;
};

struct Config {
  Config() :
    base_params(BaseParams()), model_params(ModelParams()),
    data_params(DataParams()) {}

  BaseParams base_params;
  ModelParams model_params;
  DataParams data_params;
};

} // namespace cpmf

#endif // CPMF_CONFIG_HPP_
