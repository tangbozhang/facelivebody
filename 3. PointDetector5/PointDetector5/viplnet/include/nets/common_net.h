/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Identification module, containing codes implementing the
 * face identification method described in the following paper:
 *
 *   
 *   VIPLFaceNet: An Open Source Deep Face Recognition SDK,
 *   Xin Liu, Meina Kan, Wanglong Wu, Shiguang Shan, Xilin Chen.
 *   In Frontiers of Computer Science.
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Wanglong Wu(a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems. 
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#ifndef COMMON_NET_H_
#define COMMON_NET_H_

#include "blob.h"
#include "log.h"
#include "net_factory.h"
#include "net.h"

#include <vector>

#include <orz/io/i.h>

// for each layer
#include "bias_adder_net.h"
#include "bn_net.h"
#include "conv_net.h"
#include "eltwise_net.h"
#include "eltwise_op_net.h"
#include "inner_product_net.h"
#include "max_pooling_net.h"
#include "pad_net.h"
#include "relu_net.h"
#include "scale_net.h"
#include "shape_index_patch_net.h"
#include "spatial_transform_net.h"
#include "tform_maker_net.h"

class CommonNet : public Net {
 public:
  CommonNet();
  ~CommonNet();
  // load model
  static std::shared_ptr<Net> Load(orz::FILE* file);
  // initialize the networks from a binary file
  virtual void SetUp();
  // execute the networks
  virtual void Execute();
};

#endif // COMMON_NET_H_
