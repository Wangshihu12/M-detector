#include <ros/ros.h>
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <iostream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <types.h>
#include <m-detector/DynObjFilter.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <pcl/filters/random_sample.h>
#include <Eigen/Eigen>
#include <eigen_conversions/eigen_msg.h>

#include <deque>

// #include "preprocess.h"

using namespace std;

shared_ptr<DynObjFilter> DynObjFilt(new DynObjFilter());
M3D cur_rot = Eigen::Matrix3d::Identity();
V3D cur_pos = Eigen::Vector3d::Zero();

// 四叉树最大层数
int QUAD_LAYER_MAX = 1;
// 遮挡检测窗口大小
int occlude_windows = 3;
// 点云索引
int point_index = 0;
// 垂直方向分辨率上限
float VER_RESOLUTION_MAX = 0.01;
// 水平方向分辨率上限
float HOR_RESOLUTION_MAX = 0.01;
// 角度噪声阈值
float angle_noise = 0.001;
// 遮挡角度阈值
float angle_occlude = 0.02;
// 动态检测时间窗口
float dyn_windows_dur = 0.5;
// 动态检测开关和调试开关
bool dyn_filter_en = true, dyn_filter_dbg_en = true;
// 点云和里程计话题名
string points_topic, odom_topic;
// 输出文件夹路径
string out_folder, out_folder_origin;
// 激光雷达结束时间
double lidar_end_time = 0;
// 数据集类型
int dataset = 0;
// 当前帧数
int cur_frame = 0;

// 旋转矩阵、位置、时间戳和点云的缓存队列
deque<M3D> buffer_rots;
deque<V3D> buffer_poss;
deque<double> buffer_times;
deque<boost::shared_ptr<PointCloudXYZI>> buffer_pcs;

ros::Publisher pub_pcl_dyn, pub_pcl_dyn_extend, pub_pcl_std;

// 这是里程计回调函数,用于处理接收到的里程计消息
void OdomCallback(const nav_msgs::Odometry &cur_odom)
{
    // 创建四元数对象用于旋转表示
    Eigen::Quaterniond cur_q;
    // 创建临时四元数对象存储消息中的方向数据
    geometry_msgs::Quaternion tmp_q;
    tmp_q = cur_odom.pose.pose.orientation;
    // 将ROS消息中的四元数转换为Eigen四元数
    tf::quaternionMsgToEigen(tmp_q, cur_q);
    // 将四元数转换为旋转矩阵
    cur_rot = cur_q.matrix();
    // 提取位置信息到向量中
    cur_pos << cur_odom.pose.pose.position.x, cur_odom.pose.pose.position.y, cur_odom.pose.pose.position.z;
    // 将旋转矩阵和位置向量存入缓存队列
    buffer_rots.push_back(cur_rot);
    buffer_poss.push_back(cur_pos);
    // 获取时间戳并转换为秒
    lidar_end_time = cur_odom.header.stamp.toSec();
    // 将时间戳存入缓存队列
    buffer_times.push_back(lidar_end_time);
}

// 这是点云回调函数,用于处理接收到的点云消息
void PointsCallback(const sensor_msgs::PointCloud2ConstPtr &msg_in)
{
    // 创建一个智能指针来存储未去畸变的点云特征
    boost::shared_ptr<PointCloudXYZI> feats_undistort(new PointCloudXYZI());
    // 将ROS点云消息转换为PCL点云格式
    pcl::fromROSMsg(*msg_in, *feats_undistort);
    // 将转换后的点云存入缓存队列
    buffer_pcs.push_back(feats_undistort);
}

// 这是定时器回调函数,用于定期处理缓存的数据
void TimerCallback(const ros::TimerEvent &e)
{
    // 检查所有缓存队列是否都有数据
    if (buffer_pcs.size() > 0 && buffer_poss.size() > 0 && buffer_rots.size() > 0 && buffer_times.size() > 0)
    {
        // 从各个缓存队列中取出最早的数据
        boost::shared_ptr<PointCloudXYZI> cur_pc = buffer_pcs.at(0); // 获取点云数据
        buffer_pcs.pop_front();                                      // 移除已处理的点云数据
        auto cur_rot = buffer_rots.at(0);                            // 获取旋转矩阵
        buffer_rots.pop_front();                                     // 移除已处理的旋转矩阵
        auto cur_pos = buffer_poss.at(0);                            // 获取位置向量
        buffer_poss.pop_front();                                     // 移除已处理的位置向量
        auto cur_time = buffer_times.at(0);                          // 获取时间戳
        buffer_times.pop_front();                                    // 移除已处理的时间戳

        // 构建输出文件名
        string file_name = out_folder;
        stringstream ss;
        ss << setw(6) << setfill('0') << cur_frame; // 将帧号格式化为6位数字
        file_name += ss.str();
        file_name.append(".label"); // 添加文件扩展名

        // 构建原始文件名
        string file_name_origin = out_folder_origin;
        stringstream sss;
        sss << setw(6) << setfill('0') << cur_frame; // 将帧号格式化为6位数字
        file_name_origin += sss.str();
        file_name_origin.append(".label"); // 添加文件扩展名

        // 如果文件名长度超过15,则设置输出路径
        if (file_name.length() > 15 || file_name_origin.length() > 15)
            DynObjFilt->set_path(file_name, file_name_origin);

        // 对点云数据进行过滤处理
        DynObjFilt->filter(cur_pc, cur_rot, cur_pos, cur_time);
        // 发布处理后的动态点云数据
        DynObjFilt->publish_dyn(pub_pcl_dyn, pub_pcl_dyn_extend, pub_pcl_std, cur_time);
        cur_frame++; // 帧计数器递增
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "dynfilter_odom");
    ros::NodeHandle nh;
    nh.param<string>("dyn_obj/points_topic", points_topic, "");
    nh.param<string>("dyn_obj/odom_topic", odom_topic, "");
    nh.param<string>("dyn_obj/out_file", out_folder, "");
    nh.param<string>("dyn_obj/out_file_origin", out_folder_origin, "");

    DynObjFilt->init(nh);
    /*** ROS subscribe and publisher initialization ***/
    pub_pcl_dyn_extend = nh.advertise<sensor_msgs::PointCloud2>("/m_detector/frame_out", 10000);
    pub_pcl_dyn = nh.advertise<sensor_msgs::PointCloud2>("/m_detector/point_out", 100000);
    pub_pcl_std = nh.advertise<sensor_msgs::PointCloud2>("/m_detector/std_points", 100000);
    ros::Subscriber sub_pcl = nh.subscribe(points_topic, 200000, PointsCallback);
    ros::Subscriber sub_odom = nh.subscribe(odom_topic, 200000, OdomCallback);
    ros::Timer timer = nh.createTimer(ros::Duration(0.01), TimerCallback);

    ros::spin();
    return 0;
}
