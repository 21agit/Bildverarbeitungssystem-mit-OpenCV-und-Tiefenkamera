#ifndef HISTOGRAM_PROCESSOR_HPP
#define HISTOGRAM_PROCESSOR_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace HistogramProcessor
{

  /**
   * @brief Berechnet anhand eines Histogramms einen Schwellenwert und liefert ein binarisiertes Bild.
   *
   * @param pic Eingangsbild (Graustufen)
   * @return cv::Mat Binäres Bild nach Schwellenwertbildung
   */
  inline cv::Mat computeThreshold(cv::Mat &pic)
  {
    cv::Mat normalized;
    cv::normalize(pic, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // Histogramm als 1x256 float-Matrix initialisieren
    cv::Mat histogram = cv::Mat::zeros(1, 256, CV_32FC1);
    for (int i = 0; i < normalized.rows; i++)
    {
      for (int j = 0; j < normalized.cols; j++)
      {
        histogram.at<float>(0, static_cast<int>(normalized.at<uchar>(i, j)))++;
      }
    }

    // Histogramm glätten (Gauß-Filter)
    cv::Mat smoothedHist;
    cv::GaussianBlur(histogram, smoothedHist, cv::Size(19, 19), 2.0);

    // Bestimme den maximalen Wert für die Skalierung
    double maxHistValue;
    cv::minMaxLoc(smoothedHist, nullptr, &maxHistValue);

    // Suche im Histogramm einen Hochpunkt und einen lokalen Tiefpunkt
    double imageMax;
    cv::minMaxLoc(normalized, nullptr, &imageMax);
    int index = static_cast<int>(imageMax);
    while (index >= 0 && smoothedHist.at<float>(index) <= 200)
    {
      index--;
    }
    int rearPeak = index;
    while (index >= 0 && smoothedHist.at<float>(index) >= smoothedHist.at<float>(rearPeak))
    {
      rearPeak = index;
      index--;
    }
    int localMin = rearPeak;
    while (index >= 0 && smoothedHist.at<float>(index) <= smoothedHist.at<float>(localMin))
    {
      localMin = index;
      index--;
    }

    // (Optional) Histogramm visualisieren
    cv::Mat histImage(400, 256, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < 256; i++)
    {
      line(histImage,
           cv::Point(i, 400 - smoothedHist.at<float>(i) / maxHistValue * 400),
           cv::Point(i + 1, 400 - smoothedHist.at<float>(std::min(i + 1, 255)) / maxHistValue * 400),
           cv::Scalar(255));
    }
    cv::line(histImage, cv::Point(rearPeak, 400), cv::Point(rearPeak, 0), cv::Scalar(255));
    cv::line(histImage, cv::Point(localMin, 400), cv::Point(localMin, 0), cv::Scalar(255));
    // cv::imshow("Histogram", histImage);

    // Binarisiere das Bild anhand des lokalen Tiefpunkts
    cv::threshold(normalized, normalized, localMin, 255, cv::THRESH_BINARY);
    return normalized;
  }

} // namespace HistogramProcessor

#endif // HISTOGRAM_PROCESSOR_HPP