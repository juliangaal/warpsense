#pragma once

#include <Eigen/Dense>

struct PoseRefinement
{
    PoseRefinement() 
    : q_last(Eigen::Quaterniond::Identity())
    , t_last(Eigen::Vector3d::Zero())    
    , q_diff(Eigen::Quaterniond::Identity())
    , t_diff(Eigen::Vector3d::Zero())
    , q(Eigen::Quaterniond::Identity())
    , t(Eigen::Vector3d::Zero())
    {}

    ~PoseRefinement() = default;

    void update_estimate(const Eigen::Quaterniond& q_abs, const Eigen::Vector3d& t_abs)
    {
        q_diff = q_abs * q_last.inverse();
        ROS_INFO_STREAM("LAST POSE TRANSLATION: " << t_last.x() << "; " << t_last.y() << "; " << t_last.z());
        t_diff = t_abs - t_last;
    }

    Eigen::Quaterniond initial_q()
    {
        return q * q_diff;
    }

    Eigen::Vector3d initial_t()
    {
        return t + t_diff;
    }

    void reset(const Eigen::Quaterniond& qq, const Eigen::Vector3d& tt)
    {
        // reset q_last, t_last
        auto last_pose = Eigen::Isometry3d::Identity();
        last_pose.rotate(qq);
        last_pose.pretranslate(tt);
        q_last = Eigen::Quaterniond(last_pose.rotation());
        t_last = last_pose.translation();

        // q is the pose that will be updated with refinement
        q = q_last;
        t = t_last;
        
        // reset differences since q_last, t_last
        q_diff = Eigen::Quaterniond::Identity();
        t_diff = Eigen::Vector3d::Zero();
    }

    void reset(const Eigen::Isometry3d& pose)
    {
        reset(Eigen::Quaterniond(pose.rotation()), pose.translation());
    }
 
    void refine(const Eigen::Quaterniond& q_update, const Eigen::Vector3d& t_update)
    {
        q *= q_update;
        t = q_update * t + t_update;
    }

    void refine(const Eigen::Matrix4f& mat)
    {
        Eigen::Quaterniond q_update(mat.block<3, 3>(0, 0).cast<double>());
        Eigen::Vector3d t_update(mat.block<3, 1>(0, 3).cast<double>());
        refine(q_update, t_update);
    }

    Eigen::Quaterniond q_last;
    Eigen::Vector3d t_last;
    Eigen::Quaterniond q_diff;
    Eigen::Vector3d t_diff;
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
};