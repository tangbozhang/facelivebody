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
* The codes are mainly developed by Mengru Zhang(a M.S. supervised by Prof. Shiguang Shan)
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

#include "scale_net.h"

void ScaleNet::SetUp() {
  //check input and output blob size
  this->input_blobs().resize(1);
  this->output_blobs().resize(1);
  this->input_plugs().resize(1);
  this->output_plugs().resize(1);
  this->params().resize(2);
}

void ScaleNet::Execute() {
  CheckInput();

  const Blob &input = input_blobs_[0];
  const Blob &alpha = params_[0];
  const Blob &beta = params_[1];

  int channels = input.channels();
  CHECK_EQ(channels, alpha.channels());
  CHECK_EQ(channels, beta.channels());

  int height = input.height();
  int width = input.width();
  int num = input.num();

  float* const dst_head = new float[num * channels * height * width];
  for (int n = 0, offset = 0; n < num; ++n) {
    for (int ichannel = 0; ichannel < channels; ++ichannel) {
      for (int i = 0; i < height * width; ++i, ++offset) {
        dst_head[offset] = input[offset] * alpha[ichannel] + beta[ichannel];
      }
    }
  }
  output_blobs_[0].CopyData(num, channels, height, width, dst_head);
  delete[] dst_head;
  CheckOutput();
}

REGISTER_NET_CLASS(Scale);
