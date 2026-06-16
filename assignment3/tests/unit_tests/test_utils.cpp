// tests/unit_tests/test_utils.cpp
// Unit tests for input_handler CPU path + HDR pipeline validation

#include <gtest/gtest.h>
#include "../../src/utils/input_handler.h"
#include "../../src/utils/filter_utils.h"
#include <opencv2/opencv.hpp>

// ============================================================
// CPU HDR pipeline — smoke test on tiny synthetic image
// ============================================================
TEST(HdrCpu, SmokeTest) {
    // 4×4 BGR image — all mid-grey
    cv::Mat frame(4, 4, CV_8UC3, cv::Scalar(128, 128, 128));
    HdrParams p;  // defaults
    cv::Mat result = applyHdrToneMappingCPU(frame, p);

    ASSERT_EQ(result.size(), frame.size());
    ASSERT_EQ(result.type(), CV_8UC3);
    // Output should not be all-black or all-white for mid-grey input
    for (int y = 0; y < result.rows; ++y)
        for (int x = 0; x < result.cols; ++x) {
            auto px = result.at<cv::Vec3b>(y, x);
            EXPECT_GT(px[0], 10);
            EXPECT_LT(px[0], 245);
        }
}

// ============================================================
// CPU HDR pipeline — high exposure boosts output brightness
// ============================================================
TEST(HdrCpu, ExposureIncreaseBrightness) {
    cv::Mat frame(4, 4, CV_8UC3, cv::Scalar(64, 64, 64));

    HdrParams pLow;  pLow.exposure  = 0.5f;
    HdrParams pHigh; pHigh.exposure = 4.0f;

    cv::Mat rLow  = applyHdrToneMappingCPU(frame, pLow);
    cv::Mat rHigh = applyHdrToneMappingCPU(frame, pHigh);

    // Mean of high-exposure output should be brighter
    cv::Scalar meanLow  = cv::mean(rLow);
    cv::Scalar meanHigh = cv::mean(rHigh);
    EXPECT_GT(meanHigh[0], meanLow[0]);
}

// ============================================================
// Saturation = 0 → greyscale output (all channels equal)
// ============================================================
TEST(HdrCpu, ZeroSaturationGivesGrey) {
    // Colourful input
    cv::Mat frame(4, 4, CV_8UC3, cv::Scalar(50, 128, 200));
    HdrParams p;
    p.saturation = 0.0f;
    cv::Mat result = applyHdrToneMappingCPU(frame, p);

    for (int y = 0; y < result.rows; ++y)
        for (int x = 0; x < result.cols; ++x) {
            auto px = result.at<cv::Vec3b>(y, x);
            // With zero chroma, R == G == B (allow ±1 rounding)
            EXPECT_NEAR(px[0], px[1], 2);
            EXPECT_NEAR(px[1], px[2], 2);
        }
}

// ============================================================
// Synthetic frame generator — dimensions correct
// ============================================================
TEST(SyntheticFrame, CheckerboardSize) {
    auto f = generateSyntheticFrame(SyntheticPattern::CHECKERBOARD, 640, 480);
    EXPECT_EQ(f.cols, 640);
    EXPECT_EQ(f.rows, 480);
    EXPECT_EQ(f.type(), CV_8UC3);
}

TEST(SyntheticFrame, GradientNotUniform) {
    auto f = generateSyntheticFrame(SyntheticPattern::GRADIENT, 64, 64);
    // Top-left and bottom-right pixels should differ
    auto tl = f.at<cv::Vec3b>(0,  0);
    auto br = f.at<cv::Vec3b>(63, 63);
    EXPECT_NE(tl, br);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
