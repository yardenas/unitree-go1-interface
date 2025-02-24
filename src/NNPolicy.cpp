

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
  // TODO (yarden): these can be set at compile time. Technically this could
  // just be an array, but I don't want to break old code that is known to work.
  constexpr int inputDim = 48;
  constexpr int outputDim = 12;
  inputData_.resize(inputDim);
  outputData_.resize(outputDim);
  inputShape_ = {1, inputDim};
  outputShape_ = {1, outputDim};
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
  return state.baseOrientation.inverse() * crl::Vector3d(0, 0, -9.81);
}

crl::dVector NNPolicy::getJointAngles() const {
  // TODO (yarden): make sure that the orders of indexed/joints match the ones
  // in mujoco playground
  const auto &state = data->getLeggedRobotState();
  crl::dVector jointAngles;
  crl::resize(jointAngles, state.jointStates.size());
  for (int i = 0; i < (int)state.jointStates.size(); i++) {
    jointAngles[i] = state.jointStates[i].jointPos;
  }
  return jointAngles;
}

crl::dVector NNPolicy::getJointVelocities() const {
  const auto &state = data->getLeggedRobotState();
  crl::dVector jointVelocities;
  crl::resize(jointVelocities, state.jointStates.size());
  for (int i = 0; i < (int)state.jointStates.size(); i++) {
    jointVelocities[i] = state.jointStates[i].jointVel;
  }
  return jointVelocities;
}

crl::dVector NNPolicy::getLinearVelocity() const {
  const auto &state = data->getLeggedRobotState();
  // TODO (yarden): double-check this. The velocity should be in the local
  // frame so it might be that the orientation is not relevant
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
  const auto jointAngles = getJointAngles();
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
    robot->getJoint(i)->controlMode = crl::loco::RBJointControlMode::FORCE_MODE;
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
  crl::dVector scaledAction = action_;
  crl::dVector jointTarget = action_;
  // jointTarget: (FR_hip, FR_thigh, FR_calf), (FL_hip, FL_thigh, FL_calf),
  // ...

  constexpr double go2_Hip_max = 1.0472;         // unit:radian ( = 48   degree)
  constexpr double go2_Hip_min = -1.0472;        // unit:radian ( = -48  degree)
  constexpr double go2_Front_Thigh_max = 3.4907; // unit:radian ( = 200  degree)
  constexpr double go2_Front_Thigh_min = -1.5708; // unit:radian ( = -90 degree)
  constexpr double go2_Rear_Thigh_max = 4.5379;   // unit:radian ( = 260 degree)
  constexpr double go2_Rear_Thigh_min = -0.5236; // unit:radian ( = -30  degree)
  constexpr double go2_Calf_max = -0.83776;      // unit:radian ( = -48  degree)
  constexpr double go2_Calf_min = -2.7227;       // unit:radian ( = -156 degree)

  crl::dVector MIN_JOINT_LIMIT = crl::dVector::Zero(12);
  MIN_JOINT_LIMIT << go2_Hip_min, go2_Front_Thigh_min, go2_Calf_min,
      go2_Hip_min, go2_Front_Thigh_min, go2_Calf_min, go2_Hip_min,
      go2_Rear_Thigh_min, go2_Calf_min, go2_Hip_min, go2_Rear_Thigh_min,
      go2_Calf_min;

  crl::dVector MAX_JOINT_LIMIT = crl::dVector::Zero(12);
  MAX_JOINT_LIMIT << go2_Hip_max, go2_Front_Thigh_max, go2_Calf_max,
      go2_Hip_max, go2_Front_Thigh_max, go2_Calf_max, go2_Hip_max,
      go2_Rear_Thigh_max, go2_Calf_max, go2_Hip_max, go2_Rear_Thigh_max,
      go2_Calf_max;

  crl::dVector SOFT_MIN_JOINT_LIMIT = crl::dVector::Zero(12);
  crl::dVector SOFT_MAX_JOINT_LIMIT = crl::dVector::Zero(12);
  double softDofPosLimit = 0.8;

  for (int i = 0; i < 12; i++) {
    double m = (MIN_JOINT_LIMIT[i] + MAX_JOINT_LIMIT[i]) / 2;
    double r = MAX_JOINT_LIMIT[i] - MIN_JOINT_LIMIT[i];
    SOFT_MIN_JOINT_LIMIT[i] = m - 0.5 * r * softDofPosLimit;
    SOFT_MAX_JOINT_LIMIT[i] = m + 0.5 * r * softDofPosLimit;
  }
  // clip joint target
  for (int i = 0; i < 12; i++) {
    if (jointTarget[i] < SOFT_MIN_JOINT_LIMIT[i]) {
      jointTarget[i] = SOFT_MIN_JOINT_LIMIT[i];
    } else if (jointTarget[i] > SOFT_MAX_JOINT_LIMIT[i]) {
      jointTarget[i] = SOFT_MAX_JOINT_LIMIT[i];
    }
  }
  // jointTarget
  return jointTarget;
}

crl::dVector NNPolicy::queryNetwork(const crl::dVector &obs) {
  const char *inputNames[] = {"input"};
  const char *outputNames[] = {"action"};
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