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

#include <trajlo/core/odometry.h>
#include <trajlo/utils/sophus_utils.hpp>

#include <fstream>
#include <iomanip>

uint64_t getCurrTime() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

namespace traj {
TrajLOdometry::TrajLOdometry(const TrajConfig& config)
    : config_(config), isFinish(false) {
  min_range_2_ = config.min_range * config.min_range;
  converge_thresh_ = config.converge_thresh;

  laser_data_queue.set_capacity(100);

  init_interval_ = config.init_interval;
  window_capability = config.seg_num;
  window_interval_ = config.seg_interval;
  max_frames_ = config.seg_num;
  ds_size_ = config.ds_size;
  mergSeg = false;
  ds_improve = config.ds_improve;

  store_path = config.store_path + "trajectory.txt";
  traj_stream.open(store_path, std::ios::out);

  if (!traj_stream) {
    std::cout << "ros error" << std::endl;
  }
  traj_stream.close();

  map_.reset(new MapManager(config.ds_size, config.voxel_size,
                            config.planer_thresh, config.max_voxel_num,
                            config.max_range));

  // setup marginalization
  marg_H.setZero(POSE_SIZE, POSE_SIZE);
  marg_b.setZero(POSE_SIZE);
  double init_pose_weight = config.init_pose_weight;
  marg_H.diagonal().setConstant(init_pose_weight);
}

void TrajLOdometry::Start() {
  auto lo_func = [&] {
    int frame_id = 0;
    Scan::Ptr curr_scan;
    bool first_scan = true;
    Measurement::Ptr measure;  // 点，H、B矩阵，pose
    Measurement::Ptr last_measure;

    while (true) {
      /*
       * this thread will block until the valid scan coming
       * */

      auto process_start = std::chrono::steady_clock::now();

      laser_data_queue.pop(curr_scan);
      if (!curr_scan.get()) break;

      if (first_scan) {
        last_begin_t_ns_ = curr_scan->timestamp;
        last_end_t_ns_ = last_begin_t_ns_ + init_interval_;

        last_begin_t_ns_ros_ = curr_scan->ros_tiemstamp;
        last_end_t_ns_ros_ = last_begin_t_ns_ros_ + init_interval_ * 1e-9;

        PCAAnalyse(curr_scan, first_scan);
        // std::cout<<"firstL:"<<last_end_t_ns_<<std::endl;
        first_scan = false;
      }
      PCAAnalyse(curr_scan);

      PointCloudSegment(curr_scan, measure);  // 分割当前帧，放入measure_cache

      while (!measure_cache.empty()) {

        measure = measure_cache.front();
        //measure = measure_cache.front();
        measure_cache.pop_front();
        if (mergSeg) {
          MergMeasurement(measure, last_measure);
          mergSeg = false;
        }
        // 1. range filter & compute the relative timestamps

        

        std::vector<Eigen::Vector4d> points;
        RangeFilter(measure, points);  // x,y,x,alpha

        // 当前帧的前后时间戳
        const auto& tp = measure->tp;
        const auto& tp_ros = measure->tp_ros;
        if (!map_->IsInit()) {
          T_wc_curr = Sophus::SE3d();  // 初始单位
          map_->MapInit(points);       // 加入当前点

          // standing start
          frame_poses_[tp.second] = PoseStateWithLin<double>(
              tp.second, tp_ros.second, T_wc_curr, true);
          trajectory_.emplace_back(tp.first, T_wc_curr);  // 放入当前帧的头
          trajectory_ros.emplace_back(tp_ros.first, T_wc_curr);
          map_->SetInit();
          // std::cout<<"map"<<map_->map_size()<<std::endl;
          T_prior = Sophus::SE3d();
        } else {
          // use falseSS
          if (measurements.size() == 0) {
            measure->pseudoPrior = T_prior;
          } else {
            tStampPair last_tp = measurements.rbegin()->first;
            // if(frame_poses_.find(last_tp.second)==frame_poses_.end()){
            //   std::cout<<"wrong"<<std::endl;
            // }
            T_prior = frame_poses_[last_tp.first].getPose().inverse() *
                      frame_poses_[last_tp.second].getPose();

            double pred_factor =
                double(tp.second - tp.first) / (last_tp.second - last_tp.first);
            if (pred_factor < 1.0) {
              // std::cout<<pred_factor<<"-----------------------------------------------"<<std::endl;
              measure->pseudoPrior =
                  Sophus::se3_expd(pred_factor * Sophus::se3_logd(T_prior));
            } else {
              measure->pseudoPrior = T_prior;
            }
          }
          // measure->pseudoPrior = T_prior;
          measurements[tp] = measure;
          // Sophus::SE3d T_w_pred =
          //  frame_poses_[tp.first].getPose() *
          //  T_prior;  // 当前帧的first与上一帧的end一样，预测当前帧的末尾值
          Sophus::SE3d T_w_pred =
              frame_poses_[tp.first].getPose() * measure->pseudoPrior;

          frame_poses_[tp.second] = PoseStateWithLin<double>(
              tp.second,
              T_w_pred);  // 设置当前帧的末尾位姿，加入frame_pose，
                          // 对于每一个measure，只加入末尾的位姿

          // std::cout<<measurements.size()<<std::endl;
          // ds_size_ = config_.ds_size;
          map_->setDownSample_size(ds_size_);
          kinematic_ = config_.kinematic_constrain;

        // 2. preprocess the point cloud.（下采样，放在std::map中）
        refresh_calculate:
          map_->PreProcess(points, tp);

          // 计算匹配的阈值用的，参考kiss-icp
          if (!isMove_) {
            isMove_ = (T_w_pred).translation().norm() > 0.5;
          } else {
            map_->ComputeThreshold();
          }

          // 3. find the optimal control poses based on the geometric and motion
          // constrains
          bool success = Optimize();
          if (!success) {
            if (mergSeg) {
              // delete last measure
              // last_measure = measurements[tp];
              last_measure = measure;
              measurements.erase(tp);
              frame_poses_.erase(tp.second);
              continue;
            }
            map_->setDownSample_size(ds_size_);
            goto refresh_calculate;
          }

          T_wc_curr =
              frame_poses_[tp.second].getPose();  // 当前帧末尾的位姿。优化后
          Sophus::SE3d model_deviation =
              T_w_pred.inverse() * T_wc_curr;  // 优化前后的差别
          map_->UpdateModelDeviation(
              model_deviation);  // 计算模型的阈值，点面匹配的阈值
          T_prior = frame_poses_[tp.first].getPose().inverse() *
                    frame_poses_[tp.second].getPose();  // 新的位姿变化

          // 4. marginalize the oldest segment and update the map using points
          // belond to the oldest segment. 滑动窗口边缘化最老的一帧
          Marginalize();

          // map & trajectory visualization
          if (vis_data_queue &&
              ((frame_id / config_.seg_num) % (config_.frame_num) == 0)) {
            ScanVisData::Ptr visData(new ScanVisData);

            posePair pp{frame_poses_[tp.first].getPose(),
                        frame_poses_[tp.second].getPose()};
            UndistortRawPoints(measure->points, visData->data, pp);

            visData->T_w = pp.first;
            vis_data_queue->push(visData);  // may block the thread
          }
        }
        frame_id++;
      }
      auto process_end = std::chrono::steady_clock::now();
      // std::cout << std::chrono::duration<double, std::milli>(process_end -
      //                                                        process_start).count()<<
      //                                                        std::endl;
    }

    if (vis_data_queue) vis_data_queue->push(nullptr);

    // save pose in window
    for (const auto& kv : frame_poses_) {
      trajectory_.emplace_back(kv.first, kv.second.getPose());
      trajectory_ros.emplace_back(kv.second.getT_s(), kv.second.getPose());
    }

    traj_stream.open(store_path, std::ios::out);

    if (!traj_stream) {
      std::cout << "store error" << std::endl;
    }

    for (int i = 0; i < trajectory_.size(); i++) {
      Eigen::Quaterniond q = trajectory_[i].second.unit_quaternion();
      Eigen::Vector3d t = trajectory_[i].second.translation();
      std::string time_head = std::to_string(trajectory_[i].first);
      std::string time_s = time_head.substr(0, 10);
      std::string time_ns = time_head.substr(10, 9);
      traj_stream << time_s << "." << time_ns << " " << t[0] << " " << t[1]
                  << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z()
                  << " " << q.w() << std::endl;
      // traj_stream << trajectory_[i].first << "," << i << ","
      //             << trajectory_[i].first << "," << t[0] << "," << t[1] <<
      //             ","
      //             << t[2] << "," << q.x() << "," << q.y() << "," << q.z() <<
      //             ","
      //             << q.w() << "," << 0 << "," << 0 << "," << 0 << "," << 0
      //             << "," << 0 << "," << 0 << std::endl;
    }
    // Here, you can save the trajectory for comparision

    isFinish = true;
    std::cout << "Finisher LiDAR Odometry " << std::endl;
  };

  processing_thread_.reset(new std::thread(lo_func));
}

/*
 * The analytic Jacobians in the paper are derived in SE(3) form. For the
 * efficiency in our implementation, we instead update poses in SO(3)+R3
 * form. The connection between them has been discussed in
 * https://gitlab.com/VladyslavUsenko/basalt/-/issues/37
 * */
bool TrajLOdometry::Optimize() {
  AbsOrderMap aom;
  for (const auto& kv :
       frame_poses_) {  // 滑动窗口内，每个测量的末尾位姿（时间戳， poseLIN）
    aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_SIZE);
    aom.total_size += POSE_SIZE;
    aom.items++;
  }

  Eigen::MatrixXd abs_H;
  Eigen::VectorXd abs_b;

  // for observe
  Eigen::MatrixXd abs_H_observe;
  Eigen::VectorXd abs_b_observe;
  bool isDegenerate = false;
  Eigen::MatrixXd update_P;
  // for kinematic
  // Eigen::MatrixXd abs_H_kinematic;
  // Eigen::VectorXd abs_b_kinematic;
  // // for marg
  // Eigen::MatrixXd abs_H_marg;
  // Eigen::VectorXd abs_b_marg;

  for (int iter = 0; iter < config_.max_iterations; iter++) {
    abs_H.setZero(
        aom.total_size,
        aom.total_size);  // keyframe_pose个数 * 6，（如果只有一个，eg：6*6）
    abs_b.setZero(aom.total_size);  // 只有一个 ： 6

    abs_H_observe.setZero(aom.total_size, aom.total_size);
    abs_b_observe.setZero(aom.total_size);
    update_P.setZero(aom.total_size, aom.total_size);

    // abs_H_kinematic.setZero(aom.total_size, aom.total_size);
    // abs_b_kinematic.setZero(aom.total_size);

    // abs_H_marg.setZero(aom.total_size, aom.total_size);
    // abs_b_marg.setZero(aom.total_size);

    for (auto& m : measurements) {  // 当前测量
      // 当前测量的前后时间戳
      int64_t idx_prev = m.first.first;
      int64_t idx_curr = m.first.second;
      // 对应的位姿
      const auto& prev = frame_poses_[idx_prev];
      const auto& curr = frame_poses_[idx_curr];

      posePair pp{prev.getPose(), curr.getPose()};
      const tStampPair& tp = m.second->tp;  //{idx_prev,idx_curr};一样的

      // 1. Geometric constrains from lidar point cloud.
      // 计算每一帧观测方程的H（m.second->delta_H） , b(m.second->delta_b)
      // ,error （m.second->lastError）
      map_->PointRegistrationNormal({prev, curr}, tp, m.second->delta_H,
                                    m.second->delta_b, m.second->lastError,
                                    m.second->lastInliers);

      Eigen::MatrixXd delta_H_observe = m.second->delta_H;
      Eigen::VectorXd delta_B_observe = m.second->delta_b;

      // Eigen::MatrixXd delta_H_kinematic(12, 12);
      // Eigen::VectorXd delta_B_kinematic(12);
      // 2. Motion constrains behind continuous movement.
      // Log(Tbe)-Log(prior) Equ.(6)
      {
        Sophus::SE3d T_be = pp.first.inverse() * pp.second;  // 当前帧的变化
        Sophus::Vector6d tau = Sophus::se3_logd(T_be);
        Sophus::Vector6d prev_tau = Sophus::se3_logd(m.second->pseudoPrior);
        Sophus::Vector6d res =
            tau - prev_tau;  // 当前帧的变化 减去 之前的预测值

        Sophus::Matrix6d J_T_w_b;
        Sophus::Matrix6d J_T_w_e;
        Sophus::Matrix6d rr_b;
        Sophus::Matrix6d rr_e;

        if (prev.isLinearized() || curr.isLinearized()) {
          pp = std::make_pair(prev.getPoseLin(), curr.getPoseLin());
          T_be = pp.first.inverse() * pp.second;
          tau = Sophus::se3_logd(T_be);
        }

        Sophus::rightJacobianInvSE3Decoupled(tau, J_T_w_e);
        J_T_w_b = -J_T_w_e * (T_be.inverse()).Adj();

        rr_b.setIdentity();
        rr_b.topLeftCorner<3, 3>() = pp.first.rotationMatrix().transpose();
        rr_e.setIdentity();
        rr_e.topLeftCorner<3, 3>() = pp.second.rotationMatrix().transpose();

        // Sophus::Matrix6d J_T_w_e_prev;

        // Sophus::rightJacobianInvSE3Decoupled(prev_tau, J_T_w_e_prev);

        Eigen::Matrix<double, 6, 12> J_be;
        // J_be.topLeftCorner<6, 6>() = J_T_w_b * rr_b - J_T_w_e_prev * rr_b;
        J_be.topLeftCorner<6, 6>() = J_T_w_b * rr_b;
        J_be.topRightCorner<6, 6>() = J_T_w_e * rr_e;

        // double alpha_e = config_.kinematic_constrain * m.second->lastInliers;
        double alpha_e = kinematic_ * m.second->lastInliers;
        // std::cout<<"inliers"<<m.second->lastInliers<<std::endl;
        m.second->delta_H += alpha_e * J_be.transpose() * J_be;
        m.second->delta_b -= alpha_e * J_be.transpose() * res;

        // delta_H_kinematic = alpha_e * J_be.transpose() * J_be;
        // delta_B_kinematic -= alpha_e * J_be.transpose() * res;
      }

      int abs_id = aom.abs_order_map.at(idx_prev).first;
      abs_H.block<POSE_SIZE * 2, POSE_SIZE * 2>(abs_id, abs_id) +=
          m.second->delta_H;
      abs_b.segment<POSE_SIZE * 2>(abs_id) +=
          m.second->delta_b;  // 把残差加入到整个大H矩阵

      // // for observe
      abs_H_observe.block<POSE_SIZE * 2, POSE_SIZE * 2>(abs_id, abs_id) +=
          delta_H_observe;
      abs_b_observe.segment<POSE_SIZE * 2>(abs_id) += delta_B_observe;

      // // for kinematic
      // abs_H_kinematic.block<POSE_SIZE * 2, POSE_SIZE * 2>(abs_id, abs_id) +=
      //     delta_H_kinematic;
      // abs_b_kinematic.segment<POSE_SIZE * 2>(abs_id) += delta_B_kinematic;
    }

    // Marginalization Error Term
    // reference: Square Root Marginalization for Sliding-Window Bundle
    // Adjustment (N Demmel, D Schubert, C Sommer, D Cremers and V Usenko)
    // https://arxiv.org/abs/2109.02182
    Eigen::VectorXd delta;
    for (const auto& p : frame_poses_) {
      if (p.second.isLinearized()) {
        delta = p.second.getDelta();  // 当前的第一个pose的更新量
      }
    }

    // abs_H_observe = abs_H;

    abs_H.block<POSE_SIZE, POSE_SIZE>(0, 0) += marg_H;
    abs_b.head<POSE_SIZE>() -= marg_b;
    abs_b.head<POSE_SIZE>() -= (marg_H * delta);

    Eigen::VectorXd update = abs_H.ldlt().solve(abs_b);  // 更新值

    if (iter == 0) {
      // Eigen::VectorXd abs_H_eigen_val(aom.total_size);
      // Eigen::MatrixXd abs_H_eigen_vec(aom.total_size, aom.total_size);
      // Eigen::MatrixXd abs_H_eigen_vec_2(aom.total_size, aom.total_size);
      // std::vector<double> eigen_thr(aom.total_size,
      //                               config_.degenerate_threshold);
      // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(abs_H_observe);

      Eigen::VectorXd abs_H_eigen_val(3*POSE_SIZE);
      Eigen::MatrixXd abs_H_eigen_vec(3*POSE_SIZE, 3*POSE_SIZE);
      Eigen::MatrixXd abs_H_eigen_vec_2(3*POSE_SIZE, 3*POSE_SIZE);
      std::vector<double> eigen_thr(POSE_SIZE,
                                    config_.degenerate_threshold);
      //应该对最后一组测试，否则中间有的话一直会合并
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(abs_H_observe.bottomRightCorner<3*POSE_SIZE,3*POSE_SIZE>());
      

      abs_H_eigen_vec = es.eigenvectors();
      abs_H_eigen_vec_2 = abs_H_eigen_vec.transpose();
      abs_H_eigen_val = es.eigenvalues();

      double factor_ = std::log10(abs_H_eigen_val[abs_H_eigen_val.size() - 1] /
                                  abs_H_eigen_val[0]);
      // std::cout<<"degenerate factor log:"<< factor_<<std::endl;

      // std::cout << "min_value: " << abs_H_eigen_val[0] << std::endl;
      for (int i = 0; i < abs_H_eigen_val.size() - 1; i++) {
        if (abs_H_eigen_val[i] < eigen_thr[i]) {
          for (int j = 0; j < abs_H_eigen_val.size(); j++) {
            abs_H_eigen_vec_2(i, j) = 0;
          }
          isDegenerate = true;
        }
      }

      update_P = abs_H_eigen_vec.transpose().inverse() * abs_H_eigen_vec_2;

      if (isDegenerate) {
        // std::cout << "degenerate: " << abs_H_eigen_val[0] << std::endl;
        if (config_.ds_improve) {
          // if (ds_size_ == config_.ds_size) {
          //   min_eigen_val = abs_H_eigen_val[0];
          // } else {
          //   double improve = abs_H_eigen_val[0] / min_eigen_val;
          //   std::cout << "befor:" << min_eigen_val
          //             << "after:" << abs_H_eigen_val[0] << std::endl;
          //   std::cout << "improve:" << improve << std::endl;
          //   min_eigen_val = abs_H_eigen_val[0];
          //   if (improve < 2) {
          //     goto MERG;
          //   }
          // }

          if (ds_size_ > 0.2) {
            ds_size_ = ds_size_ / (4.0);
            //std::cout << "high resolution:" << ds_size_ << std::endl;
            return false;
          } else {
            goto MERG;
          }
        }

      MERG:
        if (config_.merg) {
          // int64_t time_reserve;
          // if (measure_cache.empty()) {
          //   std::cout<<"have empty"<<std::endl;
          //   time_reserve = config_.seg_interval;
          // }else{
          //   time_reserve = window_interval_;
          // }
          if ((measurements.rbegin()->first.second -
               measurements.rbegin()->first.first) +
                  window_interval_ <
              1e8) {
            std::cout << "merg" << std::endl;
            mergSeg = true;
            return false;
          } else {
            // if(ds_size_ == config_.ds_size){
            //   ds_size_ = ds_size_ / 4.0;
            //   std::cout<<"need downsample:"<<ds_size_<<std::endl;
            //   return false;
            // }
          }
        }

        // merg

        // for(int i=0;i<aom.total_size;i++){
        //   std::cout<<abs_H_eigen_val[i]<<" ";
        // }
        // std::cout<<"----------------"<<std::endl;

        // for(int i=0;i<aom.total_size;i++){
        //   std::cout<<update[i]<<" ";
        // }
        // std::cout<<"----------------"<<std::endl;

        // update = update_P * update;
      }
    }

  GOHEAD:
  // if(Seg){
  //    std::cout << " -----------------------" << std::endl;
  // }
   
    double max_inc = update.array().abs().maxCoeff();

    if (max_inc < converge_thresh_) {
      break;
    }

    for (auto& kv : frame_poses_) {
      int idx = aom.abs_order_map.at(kv.first).first;
      kv.second.applyInc(update.segment<POSE_SIZE>(idx));  // frame更新
    }
  }

  // update pseudo motion prior after each optimization
  int64_t begin_t = measurements.begin()->first.first;
  int64_t end_t = measurements.begin()->first.second;
  auto begin = frame_poses_[begin_t];
  auto end = frame_poses_[end_t];

  const int64_t m0_t = begin_t;
  tStampPair last_tp = measurements.begin()->first;
  for (auto m : measurements) {  // 匀速递推
    if (m.first.first == m0_t) continue;
    // m.second->pseudoPrior = begin.getPose().inverse() * end.getPose();
    // begin = frame_poses_[m.first.first];
    // end = frame_poses_[m.first.second];
    double factor = (double)(m.first.second - m.first.first) /
                    (last_tp.second - last_tp.first);
    m.second->pseudoPrior = begin.getPose().inverse() * end.getPose();
    if (factor < 1.0) {
      m.second->pseudoPrior =
          Sophus::se3_expd(factor * Sophus::se3_logd(m.second->pseudoPrior));
    }
    begin = frame_poses_[m.first.first];
    end = frame_poses_[m.first.second];
    last_tp = m.first;
  }
  return true;
}

void TrajLOdometry::Marginalize() {
  // remove pose with minimal timestamp
  while (measurements.size() >=
         measurements.begin()->second->marg_times) {  //>=3,
    // if(measurements.size()>window_capability){
    const auto& tp = measurements.begin()->first;
    const auto& tp_ros = measurements.begin()->second->tp_ros;
    const posePair pp{frame_poses_[tp.first].getPose(),
                      frame_poses_[tp.second].getPose()};
    map_->Update(pp, tp);  // 将最前面的一帧加入到地图中

    Eigen::VectorXd delta = frame_poses_[tp.first].getDelta();  // 更新量

    Eigen::Matrix<double, 12, 12> marg_H_new =  // 最开始的一段对应的H矩阵
        measurements.begin()->second->delta_H;
    Eigen::Matrix<double, 12, 1> marg_b_new =
        measurements.begin()->second->delta_b;  // 最开始一段对应的B矩阵

    // 加入以前的边缘化的值....
    marg_H_new.topLeftCorner<POSE_SIZE, POSE_SIZE>() +=
        marg_H;  // 为什么要加上以前的margH？

    marg_b_new.head<POSE_SIZE>() -= marg_b;
    marg_b_new.head<POSE_SIZE>() -= (marg_H * delta);

    // 舒尔补的公共矩阵
    Eigen::MatrixXd H_mm_inv =
        marg_H_new.topLeftCorner<6, 6>().fullPivLu().solve(
            Eigen::MatrixXd::Identity(6, 6));
    marg_H_new.bottomLeftCorner<6, 6>() *= H_mm_inv;

    // 舒尔补
    marg_H = marg_H_new.bottomRightCorner<6, 6>();
    marg_b = marg_b_new.tail<6>();
    marg_H -=
        marg_H_new.bottomLeftCorner<6, 6>() * marg_H_new.topRightCorner<6, 6>();
    marg_b -= marg_H_new.bottomLeftCorner<6, 6>() * marg_b_new.head<6>();

    // erase，
    frame_poses_.erase(tp.first);
    measurements.erase(tp);

    trajectory_.emplace_back(tp.first, pp.first);
    trajectory_ros.emplace_back(tp_ros.first, pp.first);
    frame_poses_[tp.second].setLinTrue();
  }
}

void TrajLOdometry::PointCloudSegment(Scan::Ptr scan,
                                      Measurement::Ptr measure) {
  int count = 0;
  for (size_t i = 0; i < scan->size; i++) {
    const auto& p = scan->points[i];
    if (static_cast<int64_t>(p.ts * 1e9) < last_end_t_ns_) {
      points_cache.emplace_back(p);
    } else {
      // pub one measurement
      measure.reset(new Measurement);
      measure->tp = {last_begin_t_ns_, last_end_t_ns_};
      measure->tp_ros = {last_begin_t_ns_ros_, last_end_t_ns_ros_};
      measure->points = points_cache;
      measure->marg_times = window_capability;
      measure_cache.push_back(measure);

      count++;

      // reset cache and time
      points_cache.clear();
      last_begin_t_ns_ = last_end_t_ns_;
      last_end_t_ns_ = last_begin_t_ns_ + window_interval_;

      last_begin_t_ns_ros_ = last_end_t_ns_ros_;
      last_end_t_ns_ros_ = last_begin_t_ns_ros_ + window_interval_ * 1e-9;
      // std::cout<<last_end_t_ns_<<std::endl;
      if (static_cast<int64_t>(p.ts * 1e9) < last_end_t_ns_) {
        points_cache.emplace_back(p);
      }
    }
  }
  // std::cout<<"the scan seg "<<count<<std::endl;
}

void TrajLOdometry::RangeFilter(Measurement::Ptr measure,
                                std::vector<Eigen::Vector4d>& points) {
  points.reserve(measure->points.size());
  const auto& tp = measure->tp;
  double interv = (tp.second - tp.first) * 1e-9;
  double begin_ts = tp.first * 1e-9;
  for (const auto& p : measure->points) {
    if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)) continue;
    double len = (p.x * p.x + p.y * p.y + p.z * p.z);
    if (len < min_range_2_) continue;

    double alpha = (p.ts - begin_ts) / interv;
    points.emplace_back(Eigen::Vector4d(p.x, p.y, p.z, alpha));
  }
}

void TrajLOdometry::MergMeasurement(Measurement::Ptr curr_meas,
                                    Measurement::Ptr last_meas) {
  // std::cout << "*******Merging********" << std::endl;
  if (last_meas == nullptr) {
    std::cout << "last measure is nullptr!" << std::endl;
  }
  curr_meas->tp.first = last_meas->tp.first;

  std::vector<PointXYZIT>& c_ps = curr_meas->points;
  std::vector<PointXYZIT>& l_ps = last_meas->points;
  // std::cout << "before merg curr points: " << curr_meas->points.size()
  //           << std::endl;
  // std::cout << "before merg last points: " << last_meas->points.size()
  //           << std::endl;

  l_ps.insert(l_ps.begin(), c_ps.begin(), c_ps.end());

  c_ps.swap(l_ps);
  // std::cout << "after merg points: " << curr_meas->points.size() <<
  // std::endl;
  std::cout << "after merg tp: " << curr_meas->tp.second - curr_meas->tp.first
            << std::endl;
  last_meas = nullptr;
}

void TrajLOdometry::UndistortRawPoints(std::vector<PointXYZIT>& pc_in,
                                       std::vector<PointXYZI>& pc_out,
                                       const posePair& pp) {
  Sophus::Vector6f tau =
      Sophus::se3_logd(pp.first.inverse() * pp.second).cast<float>();

  std::sort(pc_in.begin(), pc_in.end(),
            [&](PointXYZIT& pl, PointXYZIT& pr) { return pl.ts < pr.ts; });
  double interv = (pc_in.back().ts - pc_in.front().ts);
  double begin_ts = pc_in.front().ts;

  pc_out.reserve(pc_in.size());
  int i = 0;
  for (const auto& p : pc_in) {
    if (i % config_.point_num == 0) {
      //        float alpha = i * 1.0f / num;
      float alpha = (p.ts - begin_ts) / interv;
      Eigen::Vector3f point(p.x, p.y, p.z);
      if (point.hasNaN() || point.squaredNorm() < 4) continue;

      Sophus::SE3f T_b_i = Sophus::se3_expd(alpha * tau);
      point = T_b_i * point;
      PointXYZI po{point(0), point(1), point(2), p.intensity};
      pc_out.emplace_back(po);
    }
    i++;
  }
}

void TrajLOdometry::PCAAnalyse(Scan::Ptr scan_, bool first_scan) {
  // ds_improve = 0;
  Seg = false;
  window_interval_ = config_.seg_interval;
  window_capability = config_.seg_num;
  ds_size_ = config_.ds_size;
  if (!config_.PCA_ANALYSE) {
    return;
  }
  Eigen::Vector3f center_point;
  center_point.setZero();
  int n = scan_->points.size();
  for (const auto& p : scan_->points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.y)) {
      continue;
    }
    center_point[0] += p.x;
    center_point[1] += p.y;
    center_point[2] += p.z;
  }

  center_point /= (double)n;
  Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
  for (const auto& p : scan_->points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.y)) {
      continue;
    }
    Eigen::Vector3f p_vec = Eigen::Vector3f(p.x, p.y, p.z);
    for (int i = 0; i < 3; i++) {
      for (int j = i; j < 3; j++) {
        covariance(i, j) +=
            (p_vec[i] - center_point[i]) * (p_vec[j] - center_point[j]);
      }
    }
  }

  covariance(1, 0) = covariance(0, 1);
  covariance(2, 0) = covariance(0, 2);
  covariance(2, 1) = covariance(1, 2);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(covariance);

  Eigen::Vector3f PCA_CURR_MAIN_DIR = es.eigenvectors().col(2).normalized();
  Eigen::Vector3f PCA_CURR_VALUE = es.eigenvalues();

  if (!first_scan) {
    double angle_1 =
        acos(PCA_CURR_MAIN_DIR.dot(PCA_LAST_MAIN_DIR) /
             (PCA_CURR_MAIN_DIR.norm() * PCA_LAST_MAIN_DIR.norm()));
    angle_1 = angle_1 / M_PI * 180;
    if (angle_1 > 90.0) {
      angle_1 = 180.0 - angle_1;
    }
    double value_1 = abs((PCA_CURR_VALUE[2] / PCA_LAST_VALUE[2]) - 1);
    double value_2 = abs((PCA_CURR_VALUE[1] / PCA_LAST_VALUE[1]) - 1);
    double value_3 = abs((PCA_CURR_VALUE[0] / PCA_LAST_VALUE[0]) - 1);
    double value = std::max(std::max(value_1, value_3), value_2);

    if (angle_1 > config_.PCA_ANGLE_THRAS && value < config_.PCA_VALUE_THRES) {
      std::cout << "angle:" << angle_1 << "value:" << value << std::endl;
      std::cout << "seg more!" << std::endl;
      window_interval_ = window_interval_ / 4.0;  // 4.0
      window_capability = config_.seg_num + 1;
      // ds_size_ = ds_size_ / 4.0;
      // ds_improve = 1;
      Seg = true;
    }
    // std::cout<<"angle:"<<angle_1<<"value:"<<value<<std::endl;
  }
  PCA_LAST_MAIN_DIR = PCA_CURR_MAIN_DIR;
  PCA_LAST_VALUE = PCA_CURR_VALUE;
}

}  // namespace traj
