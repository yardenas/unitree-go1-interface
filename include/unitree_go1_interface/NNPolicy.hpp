//
// Created by Dongho Kang on 25.02.23.
//

#ifndef UNITREE_GO1_INTERFACE_CONTROLLER_H
#define UNITREE_GO1_INTERFACE_CONTROLLER_H

#include "crl-loco/control/LocomotionController.h"
#include <filesystem>
#include <onnxruntime/onnxruntime_cxx_api.h>

namespace crl::unitree_go1_interface {

class NNPolicy final : public crl::loco::LocomotionController {
  // FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh,
  // RR_calf, RL_hip, RL_thigh, RL_calf JOINT_INDEX_MAP[CRL_INDEX] =
  // POLICY_INDEX
  // mujoco playground observation order: {'FR_hip_joint': 0, 'FR_thigh_joint':
  // 1, 'FR_calf_joint': 2, 'FL_hip_joint': 3, 'FL_thigh_joint': 4,
  // 'FL_calf_joint': 5, 'RR_hip_joint': 6, 'RR_thigh_joint': 7,
  // 'RR_calf_joint': 8, 'RL_hip_joint': 9, 'RL_thigh_joint': 10,
  // 'RL_calf_joint': 11}
  // mujoco playground actuators oder: {'FR_hip': 0, 'FR_thigh': 1, 'FR_calf':
  // 2, 'FL_hip': 3, 'FL_thigh': 4, 'FL_calf': 5, 'RR_hip': 6, 'RR_thigh': 7,
  // 'RR_calf': 8, 'RL_hip': 9, 'RL_thigh': 10, 'RL_calf': 11}
  const int JOINT_INDEX_MAP[12] = {
      /* hip */
      3, // fl
      0, // fr
      9, // rl
      6, // rr
      /* thigh */
      4,  // fl
      1,  // fr
      10, // rl
      7,  // rr
      /* calf */
      5,  // fl
      2,  // fr
      11, // rl
      8   // rr
  };

  const int VELOCITY_INDEX_MAP[3] = {
      1, // y
      2, // z
      0  // x
  };

public:
  explicit NNPolicy(
      const std::shared_ptr<crl::loco::LeggedRobot> &robot,
      const std::shared_ptr<crl::loco::LeggedLocomotionData> &data);

  void allocateMemory();

  void drawDebugInfo(const crl::gui::Shader &shader,
                     float alpha = 1.0f) override;
  void setup(const std::filesystem::path &policyPath,
             const std::vector<double> &defaultPose, double actScale);

  virtual ~NNPolicy() = default;

private:
  void loadModelFromFile(const std::filesystem::path &policyPath);
  void computeControlSignals(double dt) override;

  void applyControlSignals(double dt) override;

  void populateData() override;

  crl::dVector getObservation() const;

  crl::dVector getGyro() const;

  crl::dVector getGravity() const;

  crl::dVector getPose() const;

  crl::dVector getJointVelocities() const;

  crl::dVector getLinearVelocity() const;

  crl::dVector getCommand() const;

  crl::dVector getJointTargets() const;

  crl::dVector queryNetwork(const crl::dVector &obs);

  crl::dVector action_;
  // TODO (yarden): these should be consts and be initialized in the
  // constructor.
  crl::dVector defaultPose_;
  double actScale_;
  // onnx
  Ort::Env env_;
  Ort::Session session_{nullptr};
  Ort::Value inputTensor_{nullptr};
  Ort::Value outputTensor_{nullptr};
  // TODO (yarden): these could technically be initialized in the constructor
  // and be constants.
  std::vector<float> inputData_;
  std::vector<float> outputData_;
  std::array<int64_t, 2> inputShape_;
  std::array<int64_t, 2> outputShape_;
};
} // namespace crl::unitree_go1_interface

#endif // UNITREE_GO1_INTERFACE_CONTROLLER_H
