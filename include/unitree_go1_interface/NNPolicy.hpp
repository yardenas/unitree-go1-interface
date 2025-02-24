//
// Created by Dongho Kang on 25.02.23.
//

#ifndef UNITREE_GO1_INTERFACE_CONTROLLER_H
#define UNITREE_GO1_INTERFACE_CONTROLLER_H

#include "crl-loco/control/LocomotionController.h"
#include <onnxruntime/onnxruntime_cxx_api.h>

namespace crl::unitree_go1_interface {

class NNPolicy final : public crl::loco::LocomotionController {
  const int FL_TO_FR_INDEX_MAP[12] = {
      /* FR */
      3, // hip
      4, // thigh
      5, // calf
      /* FL */
      0, // hip
      1, // thigh
      2, // calf
      /* RR */
      9,  // hip
      10, // thigh
      11, // calf
      /* RL */
      6, // hip
      7, // thigh
      8  // calf
  };
  // FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh,
  // RR_calf, RL_hip, RL_thigh, RL_calf JOINT_INDEX_MAP[CRL_INDEX] =
  // POLICY_INDEX
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

  void setConstants();

  bool loadModelFromFile(const std::string &fileName);

  bool loadMotionFromFile(const std::string &fileName);

  void setActionNormalizer(const crl::dVector &obsJointAngleZeroOffset,
                           const crl::dVector &obsJointAngleScaleFactor,
                           const crl::dVector &obsJointSpeedScaleFactor,
                           const crl::dVector &obsAngVelScaleFactor,
                           const crl::dVector &actJointAngleZeroOffset,
                           const crl::dVector &actScaleFactor);

  void setJointObservationNormalizer(const crl::dVector &angleScaleFactor,
                                     const crl::dVector &angleZeroOffset,
                                     const crl::dVector &speedScaleFactor);

  void drawDebugInfo(const crl::gui::Shader &shader,
                     float alpha = 1.0f) override;

  crl::dVector getProprioObservation();
  crl::dVector getPrivObservation();

private:
  void computeControlSignals(double dt) override;

  void applyControlSignals(double dt) override;

  void populateData() override;

  crl::dVector getJointTargets() const;

  crl::dVector queryNetwork(const crl::dVector &obs);

private:
  // cache
  crl::dVector action_;
  crl::dVector obsHistory_;
  int inputDim_;
  int outputDim_;
  crl::dVector privLatent = crl::dVector::Zero(29);
  // normalizer
  // action
  crl::dVector actScaleFactor_;
  // joint state normalizer
  crl::dVector obsJointAngleZeroOffset_;
  crl::dVector obsJointAngleScaleFactor_;
  crl::dVector obsJointSpeedScaleFactor_;
  crl::dVector obsAngVelScaleFactor_;
  crl::dVector actJointAngleZeroOffset_;
  // onnx
  Ort::Env env_;
  Ort::Session session_{nullptr};
  Ort::Value inputTensor_{nullptr};
  Ort::Value outputTensor_{nullptr};
  std::vector<float> inputData_;
  std::vector<float> outputData_;
  std::array<int64_t, 2> inputShape_;
  std::array<int64_t, 2> outputShape_;
  // first run flags
  bool firstQuery = true;
};
} // namespace crl::unitree_go1_interface

#endif // UNITREE_GO1_INTERFACE_CONTROLLER_H
