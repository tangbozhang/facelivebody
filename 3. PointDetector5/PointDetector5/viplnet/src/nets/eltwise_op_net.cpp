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

#include "eltwise_op_net.h"

void EltwiseOPNet::SetUp() {
  op_ = *(std::string*)(this->hyper_param()->param("eltwise_op"));
  if (op_ == "SUM") {
    this->input_blobs().resize(2);
    this->output_blobs().resize(1);
    this->input_plugs().resize(2);
    this->output_plugs().resize(1);
  }
  else if (op_ == "PROD") {
  }
  else if (op_ == "MAX") {
  }
}

void EltwiseOPNet::Execute() {
  CheckInput();
  if (op_ == "SUM") {
    const Blob &input0 = input_blobs_[0];
    const Blob &input1 = input_blobs_[1];

    CHECK_EQ(input0.num(), input1.num());
    CHECK_EQ(input0.channels(), input1.channels());
    CHECK_EQ(input0.height(), input1.height());
    CHECK_EQ(input0.width(), input1.width());

    int count = input0.count();
    float* const buff = new float[count];
    for (int i = 0; i < count; i++)
      buff[i] = input0[i] + input1[i];

    output_blobs_[0].CopyData(input0.num(), input0.channels(), input0.height(), input0.width(), buff);
    delete[] buff;
  }
  else if (op_ == "PROD") {
  }
  else if (op_ == "MAX") {
  }
  CheckOutput();
}

REGISTER_NET_CLASS(EltwiseOP);
