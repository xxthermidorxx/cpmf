#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <memory>
#include <cstdlib>

#include <picojson.h>

#include "config.hpp"
#include "common/common.hpp"
#include "utils/utils.hpp"
#include "parallel/switch.hpp"


namespace cpmf {

std::unique_ptr<cpmf::Config> parse_config_json(std::ifstream &conf_ifs) {
  std::istreambuf_iterator<char> first(conf_ifs), last;
  std::string buffer(first, last), err;
  const char * buffer_ptr = buffer.c_str();
  picojson::value v;
  picojson::parse(v, buffer_ptr, buffer_ptr+strlen(buffer_ptr), &err);
  if (!err.empty()) {
    std::cerr << err << std::endl;
    return std::unique_ptr<cpmf::Config>(nullptr);
  }

  std::unique_ptr<cpmf::Config> config(new cpmf::Config);
  cpmf::BaseParams bp;
  cpmf::ModelParams mp;
  cpmf::DataParams dp;
  const picojson::value::object &obj = v.get<picojson::object>();
  for (auto i = obj.begin(), i_end = obj.end(); i != i_end; ++i) {
    std::string type = i->first;
    const picojson::value::object &subobj = i->second.get<picojson::object>();
    for (auto j = subobj.begin(), j_end = subobj.end(); j != j_end; ++j) {
      std::string key = j->first;
      const picojson::value &val = j->second;

      if (type == "base") {
        if (key == "max_iter") {
          bp.max_iter = static_cast<int>( val.get<double>() );
        } else if(key == "num_threads") {
          bp.num_threads = static_cast<int>( val.get<double>() );
        } else if(key == "calc_rmse") {
          bp.calc_rmse = static_cast<int>( val.get<bool>() );
        }

      } else if (type == "model") {
        if (key == "dimension") {
          mp.dim = static_cast<int>( val.get<double>() );
        } else if (key == "step_size") {
          mp.step_size = static_cast<float>( val.get<double>() );
        } else if (key == "regularization_cost_for_P") {
          mp.lp = static_cast<float>( val.get<double>() );
        } else if (key == "regularization_cost_for_Q") {
          mp.lq = static_cast<float>( val.get<double>() );
        } else if (key == "read_model") {
          mp.read_model = val.get<bool>();
        } else if (key == "write_model") {
          mp.write_model = val.get<bool>();
        } else if (key == "output_path") {
          mp.output_path = val.get<std::string>();
        }

      } else if (type == "data") {
        if (key == "num_user_blocks") {
          dp.num_user_blocks = static_cast<float>( val.get<double>() );
        } else if (key == "num_item_blocks") {
          dp.num_item_blocks = static_cast<float>( val.get<double>() );
        } else if (key == "input_path") {
          dp.input_path = val.get<std::string>();
        } else if (key == "test_path") {
          dp.test_path = val.get<std::string>();
        } else if (key == "output_path") {
          dp.output_path = val.get<std::string>();
        } else if (key == "randomize") {
          dp.randomize = val.get<bool>();
        }
      }
    }
  }
  config->base_params = bp;
  config->model_params = mp;
  config->data_params = dp;

  return config;
}

} // namespace cpmf


void main_help() {
  printf(
    "Usage: ./cpmf <command> <json path>\n"
    "\n"
    "Available commands are listed below:\n"
    "    train     Train a model\n"
  );
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    main_help();
    return EXIT_FAILURE;
  }
  if (argc == 2) {
    std::cerr << "FileReadError: " << "Where is a config json ?" << std::endl;
    return EXIT_FAILURE;
  }

  const char * conf_path = argv[2];
  std::ifstream conf_ifs(conf_path);
  if (conf_ifs.fail()) {
    std::cerr << "FileReadError: Cannot open " << conf_path << std::endl;
  }
  std::unique_ptr<cpmf::Config> config = cpmf::parse_config_json(conf_ifs);
  if (!config) { return EXIT_FAILURE; }

  cpmf::utils::Timer timer;

  // parse input_data
  timer.start("Now parsing input data...");
  std::shared_ptr<cpmf::common::Matrix>
    R(new cpmf::common::Matrix(config->data_params));
  timer.stop("ends.");
  R->show_info("==== MATRIX INFO ====");

  // initialize model
  timer.start("Now initializing model...");
  std::shared_ptr<cpmf::common::Model>
    model(new cpmf::common::Model(config->model_params, R));
  timer.stop("ends.");
  model->show_info("==== MODEL INFO ====");

  // begin training
  //cpmf::parallel::train(R, model, config->base_params);
  cpmf::parallel::train(R, model, config->base_params, config->data_params);

  // write model to disk
  timer.start("Writing model...");
  model->write_to_disk();
  timer.stop("ends.");

  return EXIT_SUCCESS;
}
