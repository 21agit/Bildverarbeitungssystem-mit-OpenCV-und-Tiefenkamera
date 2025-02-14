#ifndef MOTION_DETECTOR_HPP
#define MOTION_DETECTOR_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

namespace MotionDetector
{

  /**
   * @brief Erkennt Bewegung in einem binarisierten Bild und annotiert das Farbbild.
   *
   * @param bgrFrame Farbbild (wird zur Anzeige der Bewegungsrichtung annotiert)
   * @param depthFrame Tiefenbild (Graustufen bzw. float-Matrix) zur Berechnung des mittleren Tiefenwerts
   * @param binaryPic Binäres Bild (Ergebnis der Schwellwertbildung)
   */
  inline void processMotion(cv::Mat &bgrFrame, cv::Mat &depthFrame, cv::Mat &binaryPic)
  {
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(binaryPic, labels, stats, centroids);

    int largestLabel = 0;
    int largestArea = 0;
    for (int label = 1; label < numLabels; label++)
    {
      int area = stats.at<int>(label, cv::CC_STAT_AREA);
      if (area > largestArea)
      {
        largestArea = area;
        largestLabel = label;
      }
    }

    cv::Mat largestSegmentMask = (labels == largestLabel);
    int left = stats.at<int>(largestLabel, cv::CC_STAT_LEFT);
    int top = stats.at<int>(largestLabel, cv::CC_STAT_TOP);
    int width = stats.at<int>(largestLabel, cv::CC_STAT_WIDTH);
    int height = stats.at<int>(largestLabel, cv::CC_STAT_HEIGHT);

    // Zeichne das Rechteck um das größte Segment
    cv::cvtColor(binaryPic, binaryPic, cv::COLOR_GRAY2BGR);
    cv::rectangle(binaryPic, cv::Rect(left, top, width, height), cv::Scalar(0, 0, 255), 5);
    cv::imshow("Detected Region", binaryPic);

    // Berechne den mittleren Tiefenwert innerhalb des größten Segments
    double avgDepth = 0.0;
    int pixelCount = 0;
    for (int i = 0; i < depthFrame.rows; i++)
    {
      for (int j = 0; j < depthFrame.cols; j++)
      {
        if (largestSegmentMask.at<uchar>(i, j) > 0)
        {
          avgDepth += depthFrame.at<float>(i, j);
          pixelCount++;
        }
      }
    }

    // Statische Variablen, um Bewegungsänderungen zwischen Frames zu vergleichen
    static int prevLeft = 0;
    static int prevTop = 0;
    static double prevAvgDepth = 0.0;

    if (pixelCount > 0)
    {
      avgDepth /= pixelCount;
      int xDiff = std::abs(left - prevLeft);
      int yDiff = std::abs(top - prevTop);
      double depthDiff = std::abs(avgDepth - prevAvgDepth);

      // Schwellwerte für die Erkennung von Bewegungen
      int thresholdX = 10;
      int thresholdY = 10;
      double thresholdDepth = 3.0;

      if (xDiff > yDiff && xDiff > thresholdX)
      {
        cv::putText(bgrFrame, "links-rechts", cv::Point(10, 42),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
      }
      if (yDiff > xDiff && yDiff > thresholdY)
      {
        cv::putText(bgrFrame, "oben-unten", cv::Point(10, 92),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
      }
      if (depthDiff > thresholdDepth)
      {
        cv::putText(bgrFrame, "vorne-hinten", cv::Point(10, 142),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
      }
      else if (xDiff <= thresholdX && yDiff <= thresholdY && depthDiff <= thresholdDepth)
      {
        cv::putText(bgrFrame, "keine Bewegung", cv::Point(10, 42),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
      }

      prevLeft = left;
      prevTop = top;
      prevAvgDepth = avgDepth;
    }
  }

} // namespace MotionDetector

#endif // MOTION_DETECTOR_HPP