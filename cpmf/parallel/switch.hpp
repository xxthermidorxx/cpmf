#ifndef CPMF_PARALLEL_SWITCH_HPP_
#define CPMF_PARALLEL_SWITCH_HPP_

#include "../common/common.hpp"
#include "../utils/utils.hpp"

#if defined TP_BASED
#include "tp_based/tp_based.hpp"
using namespace cpmf::parallel::tp_based;

#elif defined FPSGD
#include "fpsgd/fpsgd.hpp"
using namespace cpmf::parallel::fpsgd;

#endif

namespace cpmf {
namespace parallel {

void train(const std::shared_ptr<cpmf::common::Matrix> R,
           std::shared_ptr<cpmf::common::Model> model,
           const cpmf::BaseParams &base_params,
	   const cpmf::DataParams &data_params);

} // namespace parallel
} // namespace cpmf

#endif // CPMF_PARALLEL_SWITCH_HPP_
