#include "sonar_segmentation/sonar_segmentation_node.hpp"
#include <random>

namespace vortex::sonar_segmentation {

SonarSegmentationNode::SonarSegmentationNode()
: Node("sonar_segmentation_node")
{
    declare_parameters();
    setup_publishers_and_subscribers();
    setup_service_clients();
    setup_timer();
}

void SonarSegmentationNode::declare_parameters() {
    this->declare_parameter<std::string>("topic.sonar_sub_topic");
    this->declare_parameter<std::string>("topic.segmentation_image_sub_topic");
    this->declare_parameter<std::string>("topic.depth_image_sub_topic");
    this->declare_parameter<std::string>("topic.segmentation_camera_info_topic");

    this->declare_parameter<std::string>("service.respawn_service");

    this->declare_parameter<double>("data_format.fov");
    this->declare_parameter<double>("data_format.sonar_range");
    this->declare_parameter<std::string>("data_format.output_frame_id");

    fov_ =
        this->get_parameter("data_format.fov").as_double();

    sonar_range_ =
        this->get_parameter("data_format.sonar_range").as_double();

    output_frame_id_ =
        this->get_parameter("data_format.output_frame_id").as_string();
}

void SonarSegmentationNode::setup_publishers_and_subscribers() {
    auto sonar_topic =
        this->get_parameter("topic.sonar_sub_topic").as_string();

    auto segmentation_topic =
        this->get_parameter("topic.segmentation_image_sub_topic").as_string();

    auto depth_topic =
        this->get_parameter("topic.depth_image_sub_topic").as_string();

    auto camera_info_topic =
        this->get_parameter("topic.segmentation_camera_info_topic").as_string();

    segmentation_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        segmentation_topic, 1,
        std::bind(&SonarSegmentationNode::segmentationCallback, this, std::placeholders::_1));

    depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        depth_topic, 1,
        std::bind(&SonarSegmentationNode::depthCallback, this, std::placeholders::_1));

    sonar_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        sonar_topic, 1,
        std::bind(&SonarSegmentationNode::sonarCallback, this, std::placeholders::_1));

    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic, 1,
        std::bind(&SonarSegmentationNode::cameraInfoCallback, this, std::placeholders::_1));

    output_overlay_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/sonar_segmentation/overlay_image", 10);

    output_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/sonar_segmentation/image", 10);
}

void SonarSegmentationNode::setup_service_clients() {

    auto respawn_service =
        this->get_parameter("service.respawn_service").as_string();

    respawn_client_ = this->create_client<stonefish_ros2::srv::Respawn>(respawn_service);

    while (!respawn_client_->wait_for_service(std::chrono::seconds(1))) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Service not available, waiting again...");
    }
}

void SonarSegmentationNode::setup_timer() {
    respawn_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(700),
        [this]() {
            callRespawnService();
        }
    );
}

void SonarSegmentationNode::callRespawnService() {
    auto request = std::make_shared<stonefish_ros2::srv::Respawn::Request>();

    camera_info_ready_ = false;
    depth_ready_ = false;
    segmentation_ready_ = false;
    sonar_ready_ = false;
    timer_ready_ = false;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distx(-2, 2);
    std::uniform_real_distribution<> disty(-13, 2);
    std::uniform_real_distribution<> distyaw(0, 360);

    request->name = "Orca";

    request->origin.position.x = distx(gen);
    request->origin.position.y = disty(gen);
    request->origin.position.z = -request->origin.position.y/9.0 + 5.5;

    rotando += 90.0;

    Eigen::Quaterniond q = vortex::utils::math::euler_to_quat(-95.0 * M_PI/180.0, 0.0, distyaw(gen)*M_PI/180.0);
    request->origin.orientation.x = q.x();
    request->origin.orientation.y = q.y();
    request->origin.orientation.z = q.z();
    request->origin.orientation.w = q.w();

    auto future = respawn_client_->async_send_request(request,
            [this](rclcpp::Client<stonefish_ros2::srv::Respawn>::SharedFuture response) {
                if (response.get()->success) {
                    RCLCPP_INFO(this->get_logger(), "Robot respawned successfully: %s",
                                response.get()->message.c_str());
                } else {
                    RCLCPP_WARN(this->get_logger(), "Failed to respawn robot: %s",
                                response.get()->message.c_str());
                }
            });
    camera_info_ready_ = false;
    depth_ready_ = false;
    segmentation_ready_ = false;
    sonar_ready_ = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    timer_ready_ = true;

}

void SonarSegmentationNode::cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    camera_k_ = (cv::Mat_<double>(3,3) <<
        msg->k[0], msg->k[1], msg->k[2],
        msg->k[3], msg->k[4], msg->k[5],
        msg->k[6], msg->k[7], msg->k[8]
    );
    camera_info_ready_ = true;
}

void SonarSegmentationNode::segmentationCallback(
    const sensor_msgs::msg::Image::SharedPtr msg)
{
    segmentation_img_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
    segmentation_ready_ = true;
}

void SonarSegmentationNode::depthCallback(
    const sensor_msgs::msg::Image::SharedPtr msg)
{
    depth_img_ = cv_bridge::toCvCopy(msg)->image;
    depth_ready_ = true;
}

void SonarSegmentationNode::sonarCallback(
    const sensor_msgs::msg::Image::SharedPtr msg)
{
    sonar_img_ = cv_bridge::toCvCopy(msg, "mono8")->image;
    sonar_ready_ = true;
    process();
}

void SonarSegmentationNode::process()
{
    if (!segmentation_ready_ || !depth_ready_ || !sonar_ready_ || !camera_info_ready_ || !timer_ready_) {
        return;
    }

    cv::Vec3b pipe_color = cv::Vec3b(91, 22, 229);

    timer_ready_ = false;

    cv::Mat output = cv::Mat::zeros(sonar_img_.size(), CV_8UC1);
    cv::Mat output_overlay = sonar_img_.clone();

    int width = segmentation_img_.cols;
    int height = segmentation_img_.rows;

    int sonar_center_x = output.cols / 2;
    int sonar_center_y = output.rows;

    double fx = camera_k_.at<double>(0,0);
    double fy = camera_k_.at<double>(1,1);
    double cx = camera_k_.at<double>(0,2);
    double cy = camera_k_.at<double>(1,2);
    int pmy = fy*tan(15*M_PI/180);
    int ys = cy-pmy;
    int ye = cy+pmy;

    for (int row = ys; row < ye; row++)
    {
        for (int col = 0; col < width; col++)
        {
            float z = depth_img_.at<float>(row, col);
            if (z <= 0.0 || z > sonar_range_)
                continue;

            double x_cam = (col - cx) * z / fx;
            double y_cam = (row - cy) * z / fy;

            double radius = std::sqrt(x_cam*x_cam + y_cam*y_cam + z*z);
            double angle = std::atan2(x_cam, z);

            double r_scaled = radius / sonar_range_ * output.rows;

            int x_sonar = sonar_center_x + r_scaled * sin(angle);
            int y_sonar = sonar_center_y - r_scaled * cos(angle);

            if (x_sonar >= 0 && x_sonar < output.cols &&
                y_sonar >= 0 && y_sonar < output.rows)
            {
                cv::Vec3b color = segmentation_img_.at<cv::Vec3b>(row, col);
                if (color == pipe_color) {
                    output.at<uint8_t>(y_sonar, x_sonar) = 255;
                }
            }
        }
    }

    cv::Mat filled = output.clone();

    std::vector<cv::Point> points;
    for (int y = 0; y < output.rows; y++) {
        for (int x = 0; x < output.cols; x++) {
            if (output.at<uint8_t>(y, x) == 255) {
                points.emplace_back(x, y);
            }
        }
    }

    double maxDist = 10.0;

    for (size_t i = 0; i < points.size(); i++) {
        for (size_t j = i + 1; j < points.size(); j++) {

            double dist = cv::norm(points[i] - points[j]);

            if (dist < maxDist) {
                // Draw thin line between nearby points
                cv::line(filled, points[i], points[j], 255, 1);
            }
        }
    }

    output = filled;

    auto sonar_msg = cv_bridge::CvImage(
        std_msgs::msg::Header(), "mono8", sonar_img_).toImageMsg();

    output_overlay_pub_->publish(*sonar_msg);

    auto msg = cv_bridge::CvImage(
        std_msgs::msg::Header(), "mono8", output).toImageMsg();

    output_pub_->publish(*msg);

    static int frame_id = 0;

    std::string home_dir = std::getenv("HOME");
    std::string folder = home_dir + "/dataset/testing";
    std::stringstream img_name;
    img_name << folder << "/images" << "/frame_" << std::setw(6) << std::setfill('0') << frame_id << ".png";
    cv::imwrite(img_name.str(), sonar_img_);

    std::stringstream mask_name;
    mask_name << folder << "/masks" << "/frame_" << std::setw(6) << std::setfill('0') << frame_id << "_mask.tiff";
    cv::imwrite(mask_name.str(), output);

    frame_id++;



}

} // namespace vortex::sonar_segmentation
