

#include "unitree_go1_interface/NNPolicy.hpp"
#include <array>
#include <crl-basic/utils/mathDefs.h>
#include <filesystem>
#include <string>

namespace crl::unitree_go1_interface {
NNPolicy::NNPolicy(const std::shared_ptr<crl::loco::LeggedRobot> &robot,
                   const std::shared_ptr<crl::loco::LeggedLocomotionData> &data)
    : crl::loco::LocomotionController(robot, data) {
  // update cache
  allocateMemory();
}

void NNPolicy::allocateMemory() {
  crl::resize(action_, robot->getJointCount());
  action_.setZero();
  // TODO (yarden): these can be set at compile time. Technically this could
  // just be an array, but I don't want to break old code that is known to work.
  constexpr int inputDim = 48;
  constexpr int outputDim = 12;
  inputData_.resize(inputDim);
  outputData_.resize(outputDim);
  inputShape_ = {1, inputDim};
  outputShape_ = {1, outputDim};
}

void NNPolicy::setup(const std::filesystem::path &policyPath,
                     const std::vector<double> &defaultPose, double actScale) {
  loadModelFromFile(policyPath);
  defaultPose_.resize(defaultPose.size());
  for (int i = 0; i < (int)defaultPose.size(); i++) {
    defaultPose_[i] = defaultPose[i];
  }
  actScale_ = actScale;
}

void NNPolicy::loadModelFromFile(const std::filesystem::path &policyPath) {
  // load configuration
  std::ifstream file(policyPath.string());
  if (file.fail()) {
    const std::string errorMsg =
        "Failed to load RL policy: " + policyPath.string();
    crl::Logger::print(errorMsg.c_str());
    throw std::runtime_error(errorMsg);
  }
  // load policy
  auto dirPath = policyPath.parent_path();
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  session_ =
      Ort::Session(env_, policyPath.c_str(), Ort::SessionOptions{nullptr});
  inputTensor_ = Ort::Value::CreateTensor<float>(
      memory_info, inputData_.data(), inputData_.size(), inputShape_.data(),
      inputShape_.size());
  outputTensor_ = Ort::Value::CreateTensor<float>(
      memory_info, outputData_.data(), outputData_.size(), outputShape_.data(),
      outputShape_.size());
}

void NNPolicy::drawDebugInfo(const crl::gui::Shader &, float) {}

crl::dVector NNPolicy::getGyro() const {
  crl::dVector gyro(3);
  const auto &state = data->getLeggedRobotState();
  crl::V3D baseAngVel =
      state.baseOrientation.inverse() * state.baseAngularVelocity;
  for (int i = 0; i < 3; i++) {
    gyro[VELOCITY_INDEX_MAP[i]] = baseAngVel[i];
  }
  return gyro;
}

crl::dVector NNPolicy::getGravity() const {
  const auto &state = data->getLeggedRobotState();
  return state.baseOrientation.inverse() * crl::Vector3d(0, 0, -1.);
}

crl::dVector NNPolicy::getPose() const {
  // TODO (yarden): make sure that the orders of indexed/joints match the ones
  // in mujoco playground
  // Is it radians or degrees---quick check says it's radians, but need to
  // double check. If radians -- is it -pi to pi or 0 to 2pi?
  // also make sure the directions correspond to each other (should we add minus
  // sign somewhere?)
  const auto &state = data->getLeggedRobotState();
  crl::dVector pose;
  crl::resize(pose, state.jointStates.size());
  for (size_t i = 0; i < state.jointStates.size(); i++) {
    // These poses need to be remapped to the correct order
    pose[JOINT_INDEX_MAP[i]] = state.jointStates[i].jointPos;
  }
  for (int i = 0; i < pose.size(); i++) {
    pose[i] -= defaultPose_[i];
  }
  return pose;
}

crl::dVector NNPolicy::getJointVelocities() const {
  const auto &state = data->getLeggedRobotState();
  crl::dVector jointVelocities;
  crl::resize(jointVelocities, state.jointStates.size());
  for (size_t i = 0; i < state.jointStates.size(); i++) {
    jointVelocities[JOINT_INDEX_MAP[i]] = state.jointStates[i].jointVel;
  }
  return jointVelocities;
}

crl::dVector NNPolicy::getLinearVelocity() const {
  const auto &state = data->getLeggedRobotState();
  // TODO (yarden): double-check this. The velocity should be in the local
  // frame so it might be that the orientation is not relevant
  // should also map to the correct order.
  crl::V3D baseLinVel = state.baseOrientation.inverse() * state.baseVelocity;
  crl::dVector linVel(3);
  for (int i = 0; i < 3; i++) {
    linVel[VELOCITY_INDEX_MAP[i]] = baseLinVel[i];
  }
  return linVel;
}

crl::dVector NNPolicy::getCommand() const {
  const auto &command = data->getCommand();
  crl::dVector cmd(2);
  cmd[0] = command.targetForwardSpeed;
  cmd[1] = command.targetTurngingSpeed;
  return cmd;
}

crl::dVector NNPolicy::getObservation() const {
  const crl::dVector obsAction = action_;
  const auto gyro = getGyro();
  const auto gravity = getGravity();
  const auto jointAngles = getPose();
  const auto jointVelocities = getJointVelocities();
  const auto linearVelocity = getLinearVelocity();
  const auto command = getCommand();
  crl::dVector observation;
  int obsProprioDim = gyro.size() + gravity.size() + jointAngles.size() +
                      jointVelocities.size() + linearVelocity.size() +
                      obsAction.size() + command.size();
  crl::resize(observation, obsProprioDim);
  observation << linearVelocity, gyro, gravity, jointAngles, jointVelocities,
      obsAction, command;
  return observation;
}

void NNPolicy::computeControlSignals(double) {
  auto observation = getObservation();
  // query the policy network
  crl::dVector output;
  crl::resize(output, 12);
  output = queryNetwork(observation);
  action_ = output;
}

void NNPolicy::applyControlSignals(double) {
  crl::dVector jointTargets = getJointTargets();
  for (int i = 0; i < robot->getJointCount(); i++) {
    // use POSITION_MODE for simulation
    // FIXME (yarden): this should be a parameter or just fixed for this task?
    // robot->getJoint(i)->controlMode =
    // crl::loco::RBJointControlMode::FORCE_MODE;
    robot->getJoint(i)->controlMode =
        crl::loco::RBJointControlMode::POSITION_MODE;
    robot->getJoint(i)->desiredControlPosition =
        jointTargets[JOINT_INDEX_MAP[i]];
    // TODO (yarden): double check those zeros
    robot->getJoint(i)->desiredControlSpeed = 0;
    robot->getJoint(i)->desiredControlTorque = 0;
  }
}

void NNPolicy::populateData() {
  crl::loco::LeggedRobotControlSignal control(robot);
  control.jointControl.resize(robot->getJointCount());
  for (int i = 0; i < robot->getJointCount(); i++) {
    control.jointControl[i].mode = robot->getJoint(i)->controlMode;
    control.jointControl[i].desiredPos =
        robot->getJoint(i)->desiredControlPosition;
    control.jointControl[i].desiredSpeed =
        robot->getJoint(i)->desiredControlSpeed;
    control.jointControl[i].desiredTorque =
        robot->getJoint(i)->desiredControlTorque;
  }
  data->setControlSignal(control);
}

crl::dVector NNPolicy::getJointTargets() const {
  return action_ * actScale_ + defaultPose_;
}

crl::dVector NNPolicy::queryNetwork(const crl::dVector &obs) {
  const char *inputNames[] = {"obs"};
  const char *outputNames[] = {"continuous_actions"};
  // populate input
  for (int i = 0; i < obs.size(); i++) {
    inputData_[i] = obs[i];
  }
  // run
  Ort::RunOptions runOptions;
  session_.Run(runOptions, inputNames, &inputTensor_, 1, outputNames,
               &outputTensor_, 1);
  // populate output
  crl::dVector out;
  crl::resize(out, outputData_.size());
  for (int i = 0; i < out.size(); i++) {
    out[i] = outputData_[i];
  }
  return out;
}

} // namespace crl::unitree_go1_interface