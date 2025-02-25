
#include <filesystem>
#include <vector>

#include "unitree_go1_interface/NNPolicy.hpp"
#include "unitree_go1_interface/Node.hpp"

namespace crl::unitree_go1_interface {

Node::Node(
    const crl::unitree::commons::UnitreeRobotModel &model,
    const std::shared_ptr<crl::unitree::commons::UnitreeLeggedRobotData> &data,
    const std::string &nodeName)
    : crl::unitree::commons::ControllerNode<NNPolicy>(model, data, nodeName) {
  auto paramDescription = rcl_interfaces::msg::ParameterDescriptor{};
  paramDescription.description = "Parameters for controlling the unitree go1.";
  paramDescription.read_only = true;
  this->declare_parameter<std::string>("policy_path", "data/model.onnx",
                                       paramDescription);
  this->declare_parameter<std::vector<double>>(
      "default_pose",
      {0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8},
      paramDescription);
  this->declare_parameter("action_scale", 0.5, paramDescription);
  const auto modelPath =
      std::filesystem::path(this->get_parameter("policy_path").as_string());
  controller_->setup(modelPath,
                     this->get_parameter("default_pose").as_double_array(),
                     this->get_parameter("action_scale").as_double());
}
} // namespace crl::unitree_go1_interface