#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <unordered_map>
#include <opencv2/opencv.hpp>

// DepthAI-Bibliotheken
#include "depthai/depthai.hpp"

// Eigene Module
#include "HistogramProcessor.hpp"
#include "MotionDetector.hpp"

static std::atomic<bool> downscaleColor{true};
static constexpr int fps = 30;
static constexpr auto monoRes = dai::MonoCameraProperties::SensorResolution::THE_720_P;

static float rgbWeight = 0.4f;
static float depthWeight = 0.6f;

int main(int argc, char **argv)
{
    using namespace std;

    cv::VideoWriter videoBGR, videoDPT;
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <mode>" << endl;
        return 1;
    }
    int mode = atoi(argv[1]);

    // Pipeline und Gerät initialisieren
    dai::Pipeline pipeline;
    dai::Device device;
    vector<string> queueNames;

    // Erstelle Nodes
    auto camRgb = pipeline.create<dai::node::ColorCamera>();
    auto left = pipeline.create<dai::node::MonoCamera>();
    auto right = pipeline.create<dai::node::MonoCamera>();
    auto stereo = pipeline.create<dai::node::StereoDepth>();

    auto rgbOut = pipeline.create<dai::node::XLinkOut>();
    auto depthOut = pipeline.create<dai::node::XLinkOut>();

    rgbOut->setStreamName("rgb");
    queueNames.push_back("rgb");
    depthOut->setStreamName("depth");
    queueNames.push_back("depth");

    // Konfiguration der Kameras
    camRgb->setBoardSocket(dai::CameraBoardSocket::CAM_A);
    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRgb->setFps(fps);
    if (downscaleColor)
        camRgb->setIspScale(2, 3);

    try
    {
        auto calibData = device.readCalibration2();
        auto lensPosition = calibData.getLensPosition(dai::CameraBoardSocket::CAM_A);
        if (lensPosition)
        {
            camRgb->initialControl.setManualFocus(lensPosition);
        }
    }
    catch (const std::exception &ex)
    {
        cerr << ex.what() << endl;
        return 1;
    }

    stereo->setExtendedDisparity(true);
    stereo->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_7x7);
    stereo->initialConfig.setConfidenceThreshold(200);

    left->setResolution(monoRes);
    left->setCamera("left");
    left->setFps(fps);
    right->setResolution(monoRes);
    right->setCamera("right");
    right->setFps(fps);

    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    stereo->setLeftRightCheck(true);
    stereo->setDepthAlign(dai::CameraBoardSocket::CAM_A);

    // Verknüpfe die Nodes
    camRgb->isp.link(rgbOut->input);
    left->out.link(stereo->left);
    right->out.link(stereo->right);
    stereo->disparity.link(depthOut->input);

    device.startPipeline(pipeline);

    // Verschiedene Modi (live, Aufnahme, Auswertung)
    if (mode == 1)
    {
        cv::VideoCapture cap(0);
        // Hier könnte die Liveverarbeitung implementiert werden.
    }
    else if (mode == 2)
    {
        videoBGR.open("bgr.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(1280, 720));
        videoDPT.open("dpt.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(1280, 720), false);
    }
    else if (mode == 3)
    {
        cout << "Auswertung des aufgezeichneten Videos" << endl;
        cv::VideoCapture depthVid("dpt.avi");
        cv::VideoCapture rgbVid("bgr.avi");

        while (true)
        {
            cv::Mat depthFrame, rgbFrame;
            depthVid >> depthFrame;
            rgbVid >> rgbFrame;
            if (depthFrame.empty() || rgbFrame.empty())
                break;

            cv::cvtColor(depthFrame, depthFrame, cv::COLOR_BGR2GRAY);
            cv::Mat binaryImage = HistogramProcessor::computeThreshold(depthFrame);
            MotionDetector::processMotion(rgbFrame, depthFrame, binaryImage);
            cv::imshow("Farbbild", rgbFrame);
            char key = (char)cv::waitKey(31);
            if (key == 25)
                break;
        }
        depthVid.release();
        rgbVid.release();
        cv::destroyAllWindows();
        return 0;
    }
    else
    {
        return 0;
    }

    // Setze die Ausgabewarteschlangen
    for (const auto &name : queueNames)
    {
        device.getOutputQueue(name, 4, false);
    }

    std::unordered_map<std::string, cv::Mat> frames;
    const std::string rgbWindowName = "rgb";
    const std::string depthWindowName = "depth";

    cv::namedWindow(rgbWindowName);
    cv::namedWindow(depthWindowName);

    while (true)
    {
        std::unordered_map<std::string, std::shared_ptr<dai::ImgFrame>> latestPacket;
        auto queueEvents = device.getQueueEvents(queueNames);
        for (const auto &name : queueEvents)
        {
            auto packets = device.getOutputQueue(name)->tryGetAll<dai::ImgFrame>();
            if (!packets.empty())
            {
                latestPacket[name] = packets.back();
            }
        }

        for (const auto &name : queueNames)
        {
            if (latestPacket.find(name) != latestPacket.end())
            {
                if (name == "depth")
                {
                    frames[name] = latestPacket[name]->getFrame();
                    auto maxDisparity = stereo->initialConfig.getMaxDisparity();
                    frames[name].convertTo(frames[name], CV_8UC1, 255.0 / maxDisparity);
                }
                else
                { // rgb
                    frames[name] = latestPacket[name]->getCvFrame();
                }

                if (name == "rgb")
                {
                    if (frames[name].empty())
                        break;
                    videoBGR.write(frames[name]);
                    char key = (char)cv::waitKey(1);
                    if (key == 30)
                    {
                        videoBGR.release();
                        break;
                    }
                }
                if (name == "depth")
                {
                    if (frames[name].empty())
                        break;
                    videoDPT.write(frames[name]);
                    int key = cv::waitKey(1);
                    if (key == 'q' || key == 'Q')
                    {
                        return 0;
                    }
                    char keyChar = (char)cv::waitKey(1);
                    if (keyChar == 30)
                    {
                        videoDPT.release();
                        break;
                    }
                }

                if (!frames["rgb"].empty() && !frames["depth"].empty())
                {
                    cv::Mat binaryImg = HistogramProcessor::computeThreshold(frames["depth"]);
                    MotionDetector::processMotion(frames["rgb"], frames["depth"], binaryImg);
                }
                cv::imshow(name, frames[name]);
            }
        }
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q')
            break;
    }
    return 0;
}