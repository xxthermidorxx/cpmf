#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>

#include "common.hpp"

namespace cpmf {
namespace common {

Model::Model(const cpmf::ModelParams &model_params,
             const std::shared_ptr<Matrix> R)
    : params_(model_params), num_users_(R->num_users), num_items_(R->num_items),
      num_blocks_(R->blocks.size()),
      P(new float[num_users_ * params_.dim]),
      Q(new float[num_items_ * params_.dim]) {
  if (params_.read_model) {
    read_from_disk();
  } else {
    fill_with_random_value(P, num_users_ * params_.dim);
    fill_with_random_value(Q, num_items_ * params_.dim);
  }
  set_initial_losses(R->blocks);
}

float Model::calc_rmse() {
  float sum = 0;
  long num_ratings = 0;
  for (const auto &x : losses_) {
    sum += std::accumulate(x.begin(), x.end(), 0.0);
    num_ratings += x.size();
  }
  return std::sqrt(sum/num_ratings);
}

void Model::calc_test_rmse(const std::shared_ptr<cpmf::common::Matrix> R, const std::string &test_path){
  
  std::ifstream test_ifs(test_path.c_str());
  if (test_ifs.fail()) {
    std::cerr << "FileReadError: Cannot open " << test_path << std::endl;
  }
  std::string line_buf;
  int num_ratings = 0;
  float test_rmse = 0.0;
  const int dim = params_.dim;
  while (getline(test_ifs, line_buf)) {
    int user_id, item_id;
    float rating;
    sscanf(line_buf.data(),
           "%d %d %f\n", &user_id, &item_id, &rating);
    ++num_ratings;
    float * p = P.get() + (R->copy_user_mapping[user_id-1]-1) * dim;
    float * q = Q.get() + (R->copy_item_mapping[item_id-1]-1) * dim;
    float error = rating - std::inner_product(p, p+dim, q, 0.0);
    test_rmse += error * error;
  }
  std::cerr << "Test RMSE: " << std::sqrt(test_rmse/num_ratings) << std::endl;
}

void Model::fill_with_random_value(std::unique_ptr<float> &uniq_p,
                                   const int &size) {
  std::random_device rd;
  std::mt19937 mt(rd());
  float * raw_p = uniq_p.get();
  for (int i = 0; i < size; i++) { raw_p[i] = mt() % 1000 / 1000.0; }
}

void Model::set_initial_losses(const std::vector<Block> &blocks) {
  const int dim = params_.dim;
  losses_.resize(num_blocks_, std::vector<float> (0.0));
  for (int bid = 0; bid < num_blocks_; bid++) {
    const int num_nodes = blocks[bid].nodes.size();
    losses_[bid].resize(num_nodes, 0.0);
    for (int nid = 0; nid < num_nodes; nid++) {
      const Node &node = blocks[bid].nodes[nid];
      // TODO: duplicate calculation
      float * p = P.get() + (node.user_id - 1) * dim;
      float * q = Q.get() + (node.item_id - 1) * dim;
      float error = node.rating - std::inner_product(p, p+dim, q, 0.0);
      losses_[bid][nid] = error * error;
    }
  }
}

void Model::read_from_disk() {
  int dim, num_u, num_i, num_blks;
  std::ifstream fin(params_.output_path, std::ios::in | std::ios::binary);
  // TODO: exception has to be thrown
  if (fin.fail()) {
    std::cerr << "FileReadError: Cannot open " << params_.output_path
      << std::endl;
  }
  fin.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  fin.read(reinterpret_cast<char*>(&num_u), sizeof(num_users_));
  fin.read(reinterpret_cast<char*>(&num_i), sizeof(num_items_));
  fin.read(reinterpret_cast<char*>(&num_blks), sizeof(num_blocks_));

  // check wheather meta data of binary file and Model::params_ are equal
  assert(dim == params_.dim);
  assert(num_users_ == num_u);
  assert(num_items_ == num_i);
  assert(num_blocks_ == num_blks);

  float * p = P.get();
  float * q = Q.get();
  for (int i = 0; i < num_u * params_.dim; i++) {
    fin.read(reinterpret_cast<char*>(&p[i]), sizeof(float));
  }
  for (int i = 0; i < num_i * params_.dim; i++) {
    fin.read(reinterpret_cast<char*>(&q[i]), sizeof(float));
  }
}

void Model::write_to_disk() {
  std::ofstream fout(params_.output_path, std::ios::out | std::ios::binary);
  // TODO: exception has to be thrown
  if (fout.fail()) {
    std::cerr << "FileReadError: Cannot open " << params_.output_path
      << std::endl;
  }
  float * p = P.get();
  float * q = Q.get();

  fout.write(reinterpret_cast<const char*>(&params_.dim), sizeof(num_users_));
  fout.write(reinterpret_cast<const char*>(&num_users_), sizeof(num_users_));
  fout.write(reinterpret_cast<const char*>(&num_items_), sizeof(num_items_));
  fout.write(reinterpret_cast<const char*>(&num_blocks_), sizeof(num_blocks_));
  for (int i = 0, size = num_users_ * params_.dim; i < size; i++) {
    fout.write(reinterpret_cast<char*>(&p[i]), sizeof(float));
  }
  for (int i = 0, size = num_items_ * params_.dim; i < size; i++) {
    fout.write(reinterpret_cast<char*>(&q[i]), sizeof(float));
  }
}

void Model::show_info(const std::string &message) {
  std::string info = message + "\n";
  info += "  dimension        : " + std::to_string(params_.dim) + "\n";
  info += "  step size        : " + std::to_string(params_.step_size) + "\n";
  info += "  regularizer of P : " + std::to_string(params_.lp) + "\n";
  info += "  regularizer of Q : " + std::to_string(params_.lq) + "\n";
  info += "  output path      : " + params_.output_path + "\n";
  std::cout << info << std::endl;
}

} // namespace common
} // namespace cpmf
