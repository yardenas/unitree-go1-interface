#include <boost/program_options.hpp>
#include <boost/program_options/errors.hpp>
#include <crl_fsm/fsm.h>
#include <crl_unitree_commons/RobotData.h>
#include <crl_unitree_commons/nodes/BaseStateEstimatorNode.h>
#include <crl_unitree_commons/nodes/CommNode.h>
#include <crl_unitree_commons/nodes/ContactStateEstimatorNode.h>
#include <crl_unitree_commons/nodes/EstopNode.h>
#include <crl_unitree_commons/nodes/GaitNode.h>
#include <crl_unitree_commons/nodes/StarterNode.h>
#include <crl_unitree_go1/Go1Node.h>
#include <crl_unitree_simulator/SimNode.h>
#include <memory>

#include "unitree_go1_interface/Node.hpp"

crl_fsm_states(States, SQUAT, STAND, WALK, ESTOP);
crl_fsm_machines(Machines, ONBOARD);

namespace crl::unitree_go1_interface {
void run(bool useSimulator) {
  crl::fsm::Transition<States::ESTOP, States::SQUAT> t1;
  crl::fsm::Transition<States::SQUAT, States::ESTOP> t2;
  crl::fsm::Transition<States::SQUAT, States::STAND> t3;
  crl::fsm::Transition<States::STAND, States::SQUAT> t4;
  crl::fsm::Transition<States::STAND, States::ESTOP> t5;
  crl::fsm::Transition<States::STAND, States::WALK> t6;
  crl::fsm::Transition<States::WALK, States::ESTOP> t7;
  crl::fsm::Transition<States::WALK, States::STAND> t8;
  // data
  const auto &model = crl::unitree::commons::robotModels.at("GO1");
  const auto robot = std::make_shared<crl::loco::LeggedRobot>(
      model.rbsFile.c_str(), model.rsFile.c_str());
  for (const auto &limb : model.limbNames) {
    robot->addLimb(limb.first, limb.second);
  }
  const auto data =
      std::make_shared<crl::unitree::commons::UnitreeLeggedRobotData>(robot);
  // machines
  auto m1 =
      crl::fsm::make_non_persistent_ps<Machines::ONBOARD, States::ESTOP>([&]() {
        return std::make_shared<crl::unitree::commons::EstopNode>(model, data);
      });
  auto m2 =
      crl::fsm::make_non_persistent_ps<Machines::ONBOARD, States::SQUAT>([&]() {
        return std::make_shared<crl::unitree::commons::StarterNode>(
            crl::unitree::commons::StarterNode::TargetMode::SQUAT, model, data,
            "squat");
      });
  auto m3 =
      crl::fsm::make_non_persistent_ps<Machines::ONBOARD, States::STAND>([&]() {
        return std::make_shared<crl::unitree::commons::StarterNode>(
            crl::unitree::commons::StarterNode::TargetMode::STAND, model, data,
            "stand");
      });
  auto m4 =
      crl::fsm::make_non_persistent_ps<Machines::ONBOARD, States::WALK>([&]() {
        return std::make_shared<crl::unitree_go1_interface::Node>(model, data);
      });
  auto s_cols =
      crl::fsm::make_states_collection_for_machine<Machines::ONBOARD, States>(
          m1, m2, m3, m4);
  constexpr auto t_cols = crl::fsm::make_transitions_collection<States>(
      t1, t2, t3, t4, t5, t6, t7, t8);
  // init ros process
  auto machine = crl::fsm::make_fsm<Machines, Machines::ONBOARD>(
      "robot", States::ESTOP, s_cols, t_cols);
  std::array<Machines, 1> monitoring = {Machines::ONBOARD};
  const auto contactStateEstimatorNode = std::make_shared<
      crl::unitree::commons::GaitPlanContactStateEstimatorNode>(model, data);
  const auto baseStateEstimatorNode =
      std::make_shared<crl::unitree::commons::TwoStageBaseStateEstimatorNode>(
          model, data);
  std::shared_ptr<crl::unitree::commons::RobotNode<States, Machines, 1>>
      robotNode;
  if (useSimulator) {
    robotNode =
        std::make_shared<crl::unitree::simulator::SimNode<States, Machines, 1>>(
            model, data, monitoring, machine.is_transitioning());
  } else {
    robotNode =
        std::make_shared<crl::unitree::go1::Go1Node<States, Machines, 1>>(
            model, data, monitoring, machine.is_transitioning());
  }
  const auto commNode =
      std::make_shared<crl::unitree::commons::CommNode>(model, data);
  auto &executor = machine.get_executor();
  executor.add_node(contactStateEstimatorNode);
  executor.add_node(baseStateEstimatorNode);
  executor.add_node(robotNode);
  executor.add_node(commNode);
  // main loop
  machine.spin();
  // clean up
}
} // namespace crl::unitree_go1_interface

namespace {
bool parseCommandLine(int argc, char **argv) {
  bool useSimulator = false;

  boost::program_options::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")(
      "simulator,s", boost::program_options::bool_switch(&useSimulator),
      "use simulator mode");
  boost::program_options::variables_map vm;
  try {
    boost::program_options::parsed_options parsed =
        boost::program_options::command_line_parser(argc, argv)
            .options(desc)
            .allow_unregistered() // Ignore unknown args
            .run();
    boost::program_options::notify(vm);
  } catch (const boost::program_options::error &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    std::cerr << desc << "\n";
    exit(1);
  }
  if (vm.count("help")) {
    std::cout << desc << "\n";
    exit(0);
  }
  return useSimulator;
}
} // namespace

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  const bool useSimulator = parseCommandLine(argc, argv);
  crl::unitree_go1_interface::run(useSimulator);
  rclcpp::shutdown();
  return 0;
}