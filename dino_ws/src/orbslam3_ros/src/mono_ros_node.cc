// #include <ros/ros.h>
// #include <cv_bridge/cv_bridge.h>
// #include <sensor_msgs/Image.h>
// #include <System.h>

// class MonoSLAMNode {
// public:
//     MonoSLAMNode(const std::string& voc_path, const std::string& config_path)
//     {
//         // Set last argument to 'false' to disable viewer
//         slam_ = new ORB_SLAM3::System(voc_path, config_path, ORB_SLAM3::System::MONOCULAR, false);

//         ros::NodeHandle nh;
//         sub_ = nh.subscribe("/camera/color/image_raw", 1, &MonoSLAMNode::imageCallback, this);
//     }

//     void imageCallback(const sensor_msgs::ImageConstPtr& msg)
// {
//     try {
//         cv_bridge::CvShareConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");

//         if (!cv_ptr || cv_ptr->image.empty()) {
//             ROS_WARN("Empty image received!");
//             return;
//         }

//         slam_->TrackMonocular(cv_ptr->image, msg->header.stamp.toSec());

//     } catch (cv_bridge::Exception& e) {
//         ROS_ERROR("cv_bridge error: %s", e.what());
//     } catch (std::exception& e) {
//         ROS_ERROR("Exception during tracking: %s", e.what());
//     }
// }

// void imageCallback(const sensor_msgs::ImageConstPtr& msg)
// {
//     try {
//         std::cout << "[ORB-SLAM3 ROS] Image received" << std::endl;
//         cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
//         std::cout << "[ORB-SLAM3 ROS] Running TrackMonocular..." << std::endl;
//         if (image.empty()) {
//             std::cerr << "[ERROR] Empty image received! Skipping frame." << std::endl;
//             return;
//         }
//         std::cout << "Image type: " << image.type() << std::endl;
//         std::cout << "[INFO] Image OK: " << image.cols << "x" << image.rows
//                   << " channels=" << image.channels() << std::endl;

//         slam_->TrackMonocular(image, msg->header.stamp.toSec());
//         std::cout << "[ORB-SLAM3 ROS] Tracking done." << std::endl;

//     } catch (cv_bridge::Exception& e) {
//         ROS_ERROR("cv_bridge error: %s", e.what());
//     } catch (std::exception& e) {
//         ROS_ERROR("std::exception: %s", e.what());
//     } catch (...) {
//         ROS_ERROR("Unknown error occurred during image callback");
//     }
// }

// void imageCallback(const sensor_msgs::ImageConstPtr& msg)
// {
//     try {
//         std::cout << "[ORB-SLAM3 ROS] Image received" << std::endl;
//         cv_bridge::CvShareConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");

//         if (!cv_ptr) {
//             std::cerr << "[ERROR] cv_ptr is null" << std::endl;
//             return;
//         }

//         const cv::Mat& image = cv_ptr->image;

//         if (image.empty()) {
//             std::cerr << "[ERROR] Image is empty" << std::endl;
//             return;
//         }

//         if (image.channels() != 3 || image.depth() != CV_8U) {
//             std::cerr << "[ERROR] Unexpected image format: channels=" << image.channels()
//                       << " depth=" << image.depth() << " type=" << image.type() << std::endl;
//             return;
//         }

//         std::cout << "[INFO] Image OK: " << image.cols << "x" << image.rows
//                   << " channels=" << image.channels() << std::endl;

//         slam_->TrackMonocular(image, msg->header.stamp.toSec());
//         std::cout << "[ORB-SLAM3 ROS] Tracking done." << std::endl;

//     } catch (cv_bridge::Exception& e) {
//         ROS_ERROR("cv_bridge error: %s", e.what());
//     } catch (std::exception& e) {
//         ROS_ERROR("std::exception: %s", e.what());
//     } catch (...) {
//         ROS_ERROR("Unknown error occurred during image callback");
//     }
// }


// private:
//     ORB_SLAM3::System* slam_;
//     ros::Subscriber sub_;
// };

// int main(int argc, char** argv)
// {
//     if (argc != 3) {
//         std::cerr << "Usage: rosrun orbslam3_ros mono_node <path_to_vocabulary> <path_to_config>" << std::endl;
//         return -1;
//     }

//     ros::init(argc, argv, "mono_node");
//     MonoSLAMNode node(argv[1], argv[2]);
//     ros::spin();
//     return 0;
// }

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <System.h>
#include <opencv2/imgproc/imgproc.hpp>

class MonoSLAMNode {
public:
    MonoSLAMNode(const std::string& voc_path, const std::string& config_path)
    {
        slam_ = new ORB_SLAM3::System(voc_path, config_path, ORB_SLAM3::System::MONOCULAR, true);
        ros::NodeHandle nh;
        sub_ = nh.subscribe("/camera/color/image_raw", 1, &MonoSLAMNode::imageCallback, this);
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        try {
            cv::Mat bgr = cv_bridge::toCvShare(msg, "bgr8")->image;
            cv::Mat rgb;
            cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);  // Critical conversion
            slam_->TrackMonocular(rgb, msg->header.stamp.toSec());
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge error: %s", e.what());
        }
    }

private:
    ORB_SLAM3::System* slam_;
    ros::Subscriber sub_;
};

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: rosrun orbslam3_ros mono_node <path_to_vocabulary> <path_to_config>" << std::endl;
        return -1;
    }

    ros::init(argc, argv, "mono_node");
    MonoSLAMNode node(argv[1], argv[2]);
    ros::spin();
    return 0;
}
