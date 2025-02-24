
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
  this->declare_parameter<std::string>("policyPath", paramDescription);
  // TODO (yarden): get a better default
  this->declare_parameter<std::vector<double>>(
      "defaultPose", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, paramDescription);
  const auto modelPath =
      std::filesystem::path(this->get_parameter("policyPath").as_string());
  controller_->loadModelFromFile(modelPath);
}
} // namespace crl::unitree_go1_interface