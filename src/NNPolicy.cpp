

#include "unitree_go1_interface/NNPolicy.hpp"
#include <array>
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
  crl::resize(actScaleFactor_, robot->getJointCount());
  crl::resize(obsJointAngleZeroOffset_, robot->getJointCount());
  crl::resize(obsJointAngleScaleFactor_, robot->getJointCount());
  crl::resize(obsJointSpeedScaleFactor_, robot->getJointCount());
  crl::resize(actJointAngleZeroOffset_, robot->getJointCount());
  crl::resize(obsAngVelScaleFactor_, 3);
  // TODO (yarden): these can be set at compile time. Technically this could
  // just be an array, but I don't want to break old code that is known to work.
  constexpr int inputDim = 48;
  constexpr int outputDim = 12;
  inputData_.resize(inputDim);
  outputData_.resize(outputDim);
  inputShape_ = {1, inputDim};
  outputShape_ = {1, outputDim};
}

void NNPolicy::setActionNormalizer(const crl::dVector &obsJointAngleZeroOffset,
                                   const crl::dVector &obsJointAngleScaleFactor,
                                   const crl::dVector &obsJointSpeedScaleFactor,
                                   const crl::dVector &obsAngVelScaleFactor,
                                   const crl::dVector &actJointAngleZeroOffset,
                                   const crl::dVector &actScaleFactor) {
  // Set observation normalizer
  for (int i = 0; i < robot->getJointCount(); i++) {
    obsJointAngleScaleFactor_[i] = obsJointAngleScaleFactor[i];
    obsJointSpeedScaleFactor_[i] = obsJointSpeedScaleFactor[i];
    obsJointAngleZeroOffset_[i] = obsJointAngleZeroOffset[i];
  }
  for (int i = 0; i < 3; i++) {
    obsAngVelScaleFactor_[i] = obsAngVelScaleFactor[i];
  }

  // Set action normalizer
  for (int i = 0; i < robot->getJointCount(); i++) {
    actScaleFactor_[i] = actScaleFactor[i];
    actJointAngleZeroOffset_[i] = actJointAngleZeroOffset[i];
  }
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

crl::dVector NNPolicy::getProprioObservation() {
  // priprioceptive
  crl::dVector base_ang_vel(3);
  double roll;
  double pitch;
  double yaw;
  crl::dVector placeHolder1 = crl::dVector::Zero(1);
  crl::dVector deltaYaw(1);
  crl::dVector deltaNextYaw(1);
  crl::dVector placeHolder2 = crl::dVector::Zero(1);
  crl::dVector standBoolean(1);
  crl::dVector command(1);
  crl::dVector walkBoolean(2);
  crl::dVector dof_pos(12);
  crl::dVector dof_vel(12);
  crl::dVector contactBoolean(4);
  // Hardcode scale factor
  {
    const auto &state = data->getLeggedRobotState();
    const auto &sensor = data->getSensor();
    crl::V3D baseAngVel =
        state.baseOrientation.inverse() * state.baseAngularVelocity;
    for (int i = 0; i < 3; i++) {
      base_ang_vel[VELOCITY_INDEX_MAP[i]] =
          baseAngVel[i] * obsAngVelScaleFactor_[i];
    }
    crl::utils::computeEulerAnglesFromQuaternion(
        state.baseOrientation, crl::V3D(0, 0, 1), crl::V3D(1, 0, 0),
        crl::V3D(0, 1, 0), roll, pitch, yaw);
    crl::dVector jointAngles;
    crl::dVector jointSpeed;
    crl::resize(jointAngles, state.jointStates.size());
    crl::resize(jointSpeed, state.jointStates.size());
    for (int i = 0; i < (int)state.jointStates.size(); i++) {
      jointAngles[i] = state.jointStates[i].jointPos;
      jointSpeed[i] = state.jointStates[i].jointVel;
    }
    for (int i = 0; i < 12; i++) {
      dof_pos[JOINT_INDEX_MAP[i]] =
          (jointAngles[i] - obsJointAngleZeroOffset_[i]) *
          obsJointAngleScaleFactor_[i];
      dof_vel[JOINT_INDEX_MAP[i]] =
          jointSpeed[i] * obsJointSpeedScaleFactor_[i];
    }
    walkBoolean[0] = 1.0;
    walkBoolean[1] = 0.0;
  }

  // command
  crl::dVector cmd(2);
  {
    const auto &command = data->getCommand();
    cmd[0] = command.targetForwardSpeed;
    cmd[1] = command.targetTurngingSpeed;

    double commandThreshold = 0.1;
    // calculate absolute value of command.targetForwardSpeed
    if (command.targetForwardSpeed < commandThreshold &&
        command.targetForwardSpeed > -commandThreshold) {
      cmd[0] = 0;
      standBoolean[0] = 1.0;
    } else {
      standBoolean[0] = 0.0;
    }
  }
  if (cmd[0] > 1.0) {
    cmd[0] = 1.0;
  }
  if (cmd[0] < -1.0) {
    cmd[0] = -1.0;
  }
  deltaYaw << cmd[1];
  deltaNextYaw << cmd[1];
  command << cmd[0];
  double delta_yaw_thres = PI / 180. * 30.;
  if (deltaYaw[0] > delta_yaw_thres) {
    deltaYaw[0] = delta_yaw_thres;
  }
  if (deltaYaw[0] < -delta_yaw_thres) {
    deltaYaw[0] = -delta_yaw_thres;
  }
  if (deltaNextYaw[0] > delta_yaw_thres) {
    deltaNextYaw[0] = delta_yaw_thres;
  }
  if (deltaNextYaw[0] < -delta_yaw_thres) {
    deltaNextYaw[0] = -delta_yaw_thres;
  }
  crl::dVector obsAction(12);
  for (int i = 0; i < 12; i++) {
    obsAction[i] = action_[i];
  }
  crl::dVector obsProprio;
  int obsProprioDim = base_ang_vel.size() + 2 + placeHolder1.size() +
                      deltaYaw.size() + deltaNextYaw.size() +
                      standBoolean.size() + placeHolder2.size() +
                      command.size() + walkBoolean.size() + dof_pos.size() +
                      dof_vel.size() + obsAction.size() + contactBoolean.size();
  crl::resize(obsProprio, obsProprioDim);
  obsProprio << base_ang_vel, roll, pitch, placeHolder1, deltaYaw, deltaNextYaw,
      placeHolder2, standBoolean, command, walkBoolean, dof_pos, dof_vel,
      obsAction, contactBoolean;
  return obsProprio;
}

crl::dVector NNPolicy::getPrivObservation() {
  crl::dVector base_lin_vel(3);
  crl::dVector placeholder = crl::dVector::Zero(6);

  // hardcode
  double obsLinVelScaleFactor_ = 2;

  {
    const auto &state = data->getLeggedRobotState();
    crl::V3D baseLinVel = state.baseOrientation.inverse() * state.baseVelocity;
    for (int i = 0; i < 3; i++) {
      base_lin_vel[VELOCITY_INDEX_MAP[i]] =
          baseLinVel[i] * obsLinVelScaleFactor_;
    }
  }

  crl::dVector obsPriv(9);
  obsPriv << base_lin_vel, placeholder;
  return obsPriv;
}

void NNPolicy::computeControlSignals(double) {
  // proprioceptive
  auto obsProprio = getProprioObservation();

  // exteroceptive
  const auto &state = data->getLeggedRobotState();
  double roll, pitch, yaw;
  crl::utils::computeEulerAnglesFromQuaternion(
      state.baseOrientation, crl::V3D(0, 0, 1), crl::V3D(1, 0, 0),
      crl::V3D(0, 1, 0), roll, pitch, yaw);
  //   TODO (yarden): check if this is the correct way to calculate yaw
  //   crl::Quaternion yawQuat =
  //       crl::utils::getRotationQuaternion(yaw, crl::V3D(0, 1, 0));
  // privLatent is filled internally by the policy
  crl::dVector privLatent = crl::dVector::Zero(29);

  // combine observations
  auto obsPriv = getPrivObservation();
  crl::dVector obs;
  int obsDim = obsProprio.size() + obsPriv.size() + privLatent.size();
  crl::resize(obs, obsDim);
  obs << obsProprio, obsPriv, privLatent;
  // query the policy network
  crl::dVector output;
  crl::resize(output, 12);
  output = queryNetwork(obs);
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
  scaledAction = scaledAction.cwiseProduct(actScaleFactor_);
  // clip scaledAction to [-1.2, 1.2]
  for (int i = 0; i < 12; i++) {
    if (scaledAction[i] > 1.2) {
      scaledAction[i] = 1.2;
    } else if (scaledAction[i] < -1.2) {
      scaledAction[i] = -1.2;
    }
  }
  // scaledAction = scaledAction*0;
  for (int i = 0; i < 12; i++) {
    jointTarget[JOINT_INDEX_MAP[i]] =
        scaledAction[JOINT_INDEX_MAP[i]] + actJointAngleZeroOffset_[i];
  }
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