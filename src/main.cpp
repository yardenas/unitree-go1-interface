#include <crl_fsm/fsm.h>
#include <crl_unitree_commons/RobotData.h>
#include <crl_unitree_commons/nodes/CommNode.h>
#include <crl_unitree_commons/nodes/ContactStateEstimatorNode.h>
#include <crl_unitree_commons/nodes/EstopNode.h>
#include <crl_unitree_commons/nodes/GaitNode.h>
#include <crl_unitree_commons/nodes/StarterNode.h>
#include <crl_unitree_go1/Go1Node.h>

#include "unitree_go1_interface/Node.hpp"

crl_fsm_states(States, SQUAT, STAND, WALK, ESTOP);
crl_fsm_machines(Machines, ONBOARD);

int main(int argc, char **argv) {
  //   // transitions
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
  rclcpp::init(argc, argv);
  auto machine = crl::fsm::make_fsm<Machines, Machines::ONBOARD>(
      "robot", States::ESTOP, s_cols, t_cols);
  std::array<Machines, 1> monitoring = {Machines::ONBOARD};
  const auto robotNode =
      std::make_shared<crl::unitree::go1::Go1Node<States, Machines, 1>>(
          model, data, monitoring, machine.is_transitioning());
  auto &executor = machine.get_executor();
  executor.add_node(robotNode);
  // main loop
  machine.spin();
  // clean up
  rclcpp::shutdown();
  return 0;
}