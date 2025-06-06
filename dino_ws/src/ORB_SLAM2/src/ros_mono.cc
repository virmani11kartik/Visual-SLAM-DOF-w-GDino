
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include <geometry_msgs/PoseStamped.h> 
// #include <tf/tf.h> 
#include <tf/transform_datatypes.h> 
// #include"../../../include/Converter.h"
#include "Converter.h"

#include<ros/ros.h>
// #include"std_msgs/String.h"
#include <sensor_msgs/Range.h>
#include <cv_bridge/cv_bridge.h>

#include<opencv2/core/core.hpp>

// #include"../../../include/System.h"
#include "System.h"

//#include<python2.7/Python.h>
string trackingState;
int ST;
// float *pointerTS = &trackingState;

// tf::Transform new_transform;

using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM2::System* pSLAM):mpSLAM(pSLAM){}

    void GrabImage(const sensor_msgs::ImageConstPtr& msg);

    ORB_SLAM2::System* mpSLAM;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Mono");
    ros::start();

    if(argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM2 Mono path_to_vocabulary path_to_settings" << endl;        
        ros::shutdown();
        return 1;
    }    

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);
    // SLAM.ActivateLocalizationMode();

    ImageGrabber igb(&SLAM);

    ros::NodeHandle nodeHandler;

    ros::Subscriber sub = nodeHandler.subscribe("/camera/image_raw", 1, &ImageGrabber::GrabImage,&igb);    
    ros::Publisher trackin_pub = nodeHandler.advertise<sensor_msgs::Range>("SLAM_TS", 1000);    
    // ros::Publisher camera_pose_pub = nodeHandler.advertise<geometry_msgs::PoseStamped>("SLAMCameraPose", 1000); 

    ros::Rate loop_rate(100); 

    while (ros::ok()) {        

        // //----Pose
        // geometry_msgs::PoseStamped pose;
        // pose.header.stamp = ros::Time::now();
        // pose.header.frame_id ="map";        
        // tf::poseTFToMsg(new_transform, pose.pose);
        
        // camera_pose_pub.publish(pose);

        //----TrackingS
        sensor_msgs::Range SLMAs;
        
        // std::stringstream ss;
        // ss << *pointerTS;  
        // SLMAs.range = ss.str();
        // msg.data = ss.str();
        // std::stringstream geek(trackingState);
        // geek >> ST;

        // SLMAs.header.stamp = ros::Time::now();
        // SLMAs.header.frame_id ="trackingState";  
        SLMAs.range = ST;
        trackin_pub.publish(SLMAs); 

        ros::spinOnce();           
        loop_rate.sleep();
    }  
    
    ros::spin();    

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    // Save the pointcloud       
    SLAM.SaveCloudMap("CloudMapROS.xyz");    

    ros::shutdown();

    return 0;
}

// void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg)
// {
//     // Copy the ros image message to cv::Mat.
//     cv_bridge::CvImageConstPtr cv_ptr;
//     try
//     {
//         cv_ptr = cv_bridge::toCvShare(msg);
//     }
//     catch (cv_bridge::Exception& e)
//     {
//         ROS_ERROR("cv_bridge exception: %s", e.what());
//         return;
//     }

//     cv::Mat Tcw = mpSLAM->TrackMonocular(cv_ptr->image,cv_ptr->header.stamp.toSec());
        
//     //------- Get TrackingState for publish
//     ST = mpSLAM->GetTrackingState();
    


//     // //------- Pose calculation
//     // if( countNonZero(Tcw) ){
//     //     cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t(); // Rotation information
//     //     cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3); // translation information        
//     //     vector<float> q = ORB_SLAM2::Converter::toQuaternion(Rwc);

//     //     // std:: cout << "Rwc" << twc.at<float>(0, 0) << endl;
//     //     // std:: cout << "twc" << twc << endl;
//     //     // std:: cout << "q" << q[0] << ", " << q[1] << ", " << q[2] << ", " << q[3] << endl;
                
//     //     new_transform.setOrigin(tf::Vector3(twc.at<float>(0, 0), twc.at<float>(0, 1), twc.at<float>(0, 2)));
//     //     tf::Quaternion quaternion(q[0], q[1], q[2], q[3]);
//     //     new_transform.setRotation(quaternion);    
//     // } else {
//     //     new_transform.setOrigin(tf::Vector3(0.0,0.0,0.0));
//     //     tf::Quaternion quaternion(0.0, 0.0, 0.0, 1.0);
//     //     new_transform.setRotation(quaternion);  
//     // }
        
// }

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Debug info
    std::cout << "Received image. Type: " << cv_ptr->image.type()
              << ", Channels: " << cv_ptr->image.channels()
              << ", Size: " << cv_ptr->image.cols << "x" << cv_ptr->image.rows << std::endl;

    if (cv_ptr->image.empty() || cv_ptr->image.cols == 0 || cv_ptr->image.rows == 0) {
        std::cerr << "Empty or invalid image received. Skipping." << std::endl;
        return;
    }

    // Force grayscale conversion
    cv::Mat gray;
    if (cv_ptr->image.channels() == 3) {
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = cv_ptr->image.clone();
    }

    try {
        cv::Mat Tcw = mpSLAM->TrackMonocular(gray, cv_ptr->header.stamp.toSec());
        ST = mpSLAM->GetTrackingState();
    } catch (const std::exception& e) {
        std::cerr << "Exception in TrackMonocular: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception occurred in TrackMonocular" << std::endl;
    }
}




