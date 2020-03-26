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
 * The codes are mainly developed by Zining Xu(a M.S. supervised by Prof. Shiguang Shan)
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

#include "shape_index_patch_net.h"

void ShapeIndexPatchNet::SetUp() {
  // params
  origin_patch_h_ = *(int *)(hyper_params_.param("origin_patch_h"));
  origin_patch_w_ = *(int *)(hyper_params_.param("origin_patch_w"));
  origin_h_ = *(int *)(hyper_params_.param("origin_h"));
  origin_w_ = *(int *)(hyper_params_.param("origin_w"));

  // check input and output blob size
  input_blobs_.resize(2);
  output_blobs_.resize(1);
  input_plugs_.resize(2);
  output_plugs_.resize(1);
}

void ShapeIndexPatchNet::Execute() {
  CheckInput();
  const Blob &feat_blob = input_blobs_[0];
  const Blob &pos_blob = input_blobs_[1];

  int feat_h = feat_blob.height();
  int feat_w = feat_blob.width();
  int feat_patch_h = int(origin_patch_h_ * feat_h / float(origin_h_) + 0.5f);
  int feat_patch_w = int(origin_patch_w_ * feat_w / float(origin_w_) + 0.5f);

  const int num = feat_blob.num();
  const int channels = feat_blob.channels();
  const float r_h = (feat_patch_h - 1) / 2.0f;
  const float r_w = (feat_patch_w - 1) / 2.0f;
  const int landmark_num = pos_blob.channels() / 2;

  // offset
  float *const buff = new float[num * channels * feat_patch_h * feat_patch_w * landmark_num];
  output_blobs_[0].reshape(num, channels, feat_patch_h, feat_patch_w * landmark_num);
  for (int i = 0; i < landmark_num; i++) {
    for (int n = 0; n < num; n++) { // x1, y1, ..., xn, yn
      // coordinate of the first patch pixel, scale to the feature map coordinate
      const int y = int(pos_blob[pos_blob.offset(n, 2 * i + 1)] * (feat_h - 1) - r_h + 0.5f);
      const int x = int(pos_blob[pos_blob.offset(n, 2 * i)] * (feat_w - 1) - r_w + 0.5f);

      for (int c = 0; c < channels; c++) {
        for (int ph = 0; ph < feat_patch_h; ph++) {
          for (int pw = 0; pw < feat_patch_w; pw++) {
            const int y_p = y + ph;
            const int x_p = x + pw;
            // set zero if exceed the img bound
            if (y_p < 0 || y_p >= feat_h || x_p < 0 || x_p >= feat_w)
              buff[output_blobs_[0].offset(n, c, ph, pw + i * feat_patch_w)] = 0;
            else
              buff[output_blobs_[0].offset(n, c, ph, pw + i * feat_patch_w)] =
              feat_blob[feat_blob.offset(n, c, y_p, x_p)];
          }
        }
      }
    }
  }
  output_blobs_[0].CopyData(num, channels, feat_patch_h, feat_patch_w * landmark_num, buff);
  delete[] buff;
  CheckOutput();
}

REGISTER_NET_CLASS(ShapeIndexPatch);
