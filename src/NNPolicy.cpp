

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
                     const std::vector<double> &defaultPose, double actScale,
                     bool useSimulator) {
  useSimulator_ = useSimulator;
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
  const auto gravityLocal =
      state.baseOrientation.inverse() * crl::Vector3d(0, -1., 0);
  crl::dVector gravity(3);
  for (int i = 0; i < 3; i++) {
    gravity[VELOCITY_INDEX_MAP[i]] = gravityLocal[i];
  }
  return gravity;
}

crl::dVector NNPolicy::getPose() const {
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
  crl::V3D baseLinVel = state.baseOrientation.inverse() * state.baseVelocity;
  crl::dVector linVel(3);
  for (int i = 0; i < 3; i++) {
    linVel[VELOCITY_INDEX_MAP[i]] = baseLinVel[i];
  }
  return linVel;
}

crl::dVector NNPolicy::getCommand() const {
  const auto &command = data->getCommand();
  crl::dVector cmd(3);
  cmd[0] = command.targetForwardSpeed;
  cmd[1] = command.targetSidewaysSpeed;
  cmd[2] = command.targetTurngingSpeed;
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
    if (useSimulator_) {
      robot->getJoint(i)->controlMode =
          crl::loco::RBJointControlMode::POSITION_MODE;
    } else {
      robot->getJoint(i)->controlMode =
          crl::loco::RBJointControlMode::FORCE_MODE;
    }
    robot->getJoint(i)->desiredControlPosition =
        jointTargets[JOINT_INDEX_MAP[i]];
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