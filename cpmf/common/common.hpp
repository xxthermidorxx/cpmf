#ifndef CPMF_COMMON_HPP_
#define CPMF_COMMON_HPP_

#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "../config.hpp"

namespace cpmf {
namespace common {

struct Node {
  Node()
    : user_id(0), item_id(0), orig_user_id(0), orig_item_id(0), rating(0.0) {}
  int user_id, item_id, orig_user_id, orig_item_id;
  float rating;
};


struct Block {
  Block(const int &block_user_id, const int &block_item_id)
    : user_id(block_user_id), item_id(block_item_id), nodes(0) {}
  int user_id, item_id;
  std::vector<Node> nodes;
};


class Matrix {
 public:
  Matrix(const cpmf::DataParams &data_params);
  void show_info(const std::string &message);

  long num_ratings;
  int num_users, num_items, num_user_blocks, num_item_blocks;
  std::vector<Block> blocks;
  std::vector<int> copy_user_mapping;
  std::vector<int> copy_item_mapping;

 private:
  void initialize_blocks();
  void read(std::vector<Node> * temp_nodes);
  void generate_mapping_vector(std::vector<int> * mapping_vec, bool randomize);
  void assign_nodes(std::vector<Node> * temp_nodes,
                    const std::vector<int> &user_mapping,
                    const std::vector<int> &item_mapping);
  void sort_nodes_by_user_id();

  std::string input_path_;
  std::string output_path_;
};


class Model {
 public:
  Model(const cpmf::ModelParams &model_params, const std::shared_ptr<Matrix> R);

  inline void sgd(const int &block_id, const Block &block);
  float calc_rmse();
  void calc_test_rmse(const std::shared_ptr<cpmf::common::Matrix> R, const std::string &test_path);
  void write_to_disk();
  void show_info(const std::string &message);

 private:
  void fill_with_random_value(std::unique_ptr<float> &uniq_p, const int &size);
  void set_initial_losses(const std::vector<Block> &blocks);
  void read_from_disk();

  cpmf::ModelParams params_;
  int num_users_, num_items_, num_blocks_;
  std::vector<std::vector<float>> losses_;
  std::unique_ptr<float> P, Q;
};

inline void Model::sgd(const int &block_id, const Block &block) {
  const int dim = params_.dim;
  const float step_size = params_.step_size;

  for (int nid = 0, num_nodes = block.nodes.size(); nid < num_nodes; nid++) {
    const auto &node = block.nodes[nid];
    // TODO: duplicate calculation
    float * p = P.get() + (node.user_id - 1) * dim;
    float * q = Q.get() + (node.item_id - 1) * dim;
    float error = node.rating - std::inner_product(p, p+dim, q, 0.0);
    losses_[block_id][nid] = error * error;
    for (int d = 0; d < dim; d++) {
      float temp = p[d];
      p[d] += (error * q[d] - params_.lp * p[d]) * step_size;
      q[d] += (error * temp - params_.lq * q[d]) * step_size;
    }
  }
}

} // namespace common
} // namespace cpmf

#endif // CPMF_COMMON_HPP_
