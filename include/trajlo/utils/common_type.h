/**
MIT License

Copyright (c) 2023 Xin Zheng <xinzheng@zju.edu.cn>.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef TRAJLO_COMMON_TYPE_H
#define TRAJLO_COMMON_TYPE_H

#include <trajlo/utils/pose_type.h>
#include <memory>
#include <trajlo/utils/eigen_utils.hpp>
#include "sophus/se3.hpp"

namespace traj {
struct PointXYZI {
  float x;
  float y;
  float z;
  float intensity;
};

struct PointXYZIT {
  float x;
  float y;
  float z;
  float intensity;
  double ts;
};

struct Scan {
  using Ptr = std::shared_ptr<Scan>;
  int64_t timestamp;  // first point time
  size_t size;
  double ros_tiemstamp;
//  Sophus::SE3d gt;

  std::vector<PointXYZIT> points;
};

using tStampPair = std::pair<int64_t, int64_t>;
using tStampPair_ros = std::pair<double, double>;

struct Measurement {
  using Ptr = std::shared_ptr<Measurement>;
  tStampPair tp;//测量对应的时间戳
  tStampPair_ros tp_ros;
  std::vector<PointXYZIT> points;  // for visualization
  Sophus::SE3d pseudoPrior;//SE3d，预测的位姿变化值
  
  int marg_times=0;

  Eigen::Matrix<double, 12, 12> delta_H;//H矩阵, 12维对应前后两个位姿
  Eigen::Matrix<double, 12, 1> delta_b;//b距阵
  double lastError = 0;
  double lastInliers = 0;

};

using posePair = std::pair<Sophus::SE3d, Sophus::SE3d>;
using posePairLin =
    std::pair<PoseStateWithLin<double>, PoseStateWithLin<double>>;

struct float3 {
  float x;
  float y;
  float z;
};

struct ScanVisData {
  using Ptr = std::shared_ptr<ScanVisData>;

  Sophus::SE3d T_w;
  std::vector<PointXYZI> data;
};

}  // namespace traj

#endif  // TRAJLO_COMMON_TYPE_H
