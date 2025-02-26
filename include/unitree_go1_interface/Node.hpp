
#ifndef UNITREE_GO1_INTERFACE_NODE_H
#define UNITREE_GO1_INTERFACE_NODE_H

#include <crl_unitree_commons/nodes/BaseNode.h>
#include <crl_unitree_commons/nodes/ControllerNode.h>

#include "unitree_go1_interface/NNPolicy.hpp"

namespace crl::unitree_go1_interface {

class Node : public crl::unitree::commons::ControllerNode<NNPolicy> {
public:
  Node(const crl::unitree::commons::UnitreeRobotModel &model, //
       const std::shared_ptr<crl::unitree::commons::UnitreeLeggedRobotData>
           &data, //
       const std::string &nodeName = "controller", bool useSimulator = false);
  virtual ~Node() = default;
};
} // namespace crl::unitree_go1_interface

#endif // UNITREE_GO1_INTERFACE_NODE_H
