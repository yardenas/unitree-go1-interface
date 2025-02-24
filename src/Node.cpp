
#include <filesystem>

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
  this->declare_parameter<std::string>("data_folder", ".", paramDescription);
  this->declare_parameter<std::string>("model", "/model/go1/model.json",
                                       paramDescription);
  const auto modelPath =
      std::filesystem::path(this->get_parameter("model").as_string());
  const auto dataPath =
      std::filesystem::path(this->get_parameter("data_folder").as_string());
  controller_->loadModelFromFile((dataPath / modelPath).string());
}
} // namespace crl::unitree_go1_interface