// tests/unit_tests/test_convolution.cpp
// Unit tests for filter kernel creation + HDR parameter helpers

#include <gtest/gtest.h>
#include "../../src/utils/filter_utils.h"
#include <cmath>
#include <numeric>

// ============================================================
// FilterType string mapping
// ============================================================
TEST(FilterType, BlurMapping) {
    EXPECT_EQ(stringToFilterType("blur"), FilterType::BLUR);
}
TEST(FilterType, HdrMapping) {
    EXPECT_EQ(stringToFilterType("hdr"), FilterType::HDR_TONEMAPPING);
}
TEST(FilterType, UnknownThrows) {
    EXPECT_THROW(stringToFilterType("unknown"), std::invalid_argument);
}

// ============================================================
// ToneMappingAlgorithm string mapping
// ============================================================
TEST(ToneMapping, ReinhardMapping) {
    EXPECT_EQ(stringToToneMappingAlgorithm("reinhard"),
              ToneMappingAlgorithm::REINHARD);
}
TEST(ToneMapping, DragoMapping) {
    EXPECT_EQ(stringToToneMappingAlgorithm("drago"),
              ToneMappingAlgorithm::DRAGO);
}
TEST(ToneMapping, LocalMapping) {
    EXPECT_EQ(stringToToneMappingAlgorithm("local"),
              ToneMappingAlgorithm::LOCAL);
}
TEST(ToneMapping, UnknownThrows) {
    EXPECT_THROW(stringToToneMappingAlgorithm("bad"), std::invalid_argument);
}

// ============================================================
// Gaussian blur kernel — sum should be ~1.0
// ============================================================
TEST(FilterKernel, BlurSumsToOne) {
    FilterKernel fk = createFilterKernel(FilterType::BLUR, 5, 1.0f, 1.0f);
    ASSERT_EQ(fk.size, 5);
    float sum = std::accumulate(fk.data.begin(), fk.data.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// ============================================================
// HDR filter returns empty kernel (not convolution-based)
// ============================================================
TEST(FilterKernel, HdrReturnsEmptyKernel) {
    FilterKernel fk = createFilterKernel(FilterType::HDR_TONEMAPPING, 3);
    EXPECT_EQ(fk.size, 0);
    EXPECT_TRUE(fk.data.empty());
}

// ============================================================
// Reinhard operator: maps input luminance to (0,1)
// ============================================================
TEST(ReinhardMath, OutputInRange) {
    // Simulate: Ls = key/avgLum * L; Ld = Ls*(1+Ls/Lw²)/(1+Ls)
    float key = 0.18f, avgLum = 0.18f, whitePoint = 4.0f;
    float Lw2 = whitePoint * whitePoint;

    for (float L : {0.01f, 0.18f, 1.0f, 4.0f, 16.0f}) {
        float Ls = (key / avgLum) * L;
        float Ld = Ls * (1.0f + Ls / Lw2) / (1.0f + Ls);
        EXPECT_GE(Ld, 0.0f);
        EXPECT_LT(Ld, 1.5f);   // extended Reinhard stays bounded
    }
}

// ============================================================
// Drago: log function monotonically increases
// ============================================================
TEST(DragoMath, Monotone) {
    float bias = 0.85f, maxLum = 10.0f;
    float logBiasLog05 = std::log(bias) / std::log(0.5f);

    float prevLd = -1.0f;
    for (float L : {0.1f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f}) {
        float num   = std::log(L + 1.0f);
        float denom = std::log(2.0f + 8.0f * std::pow(L/maxLum, logBiasLog05));
        float Ld    = num / denom;
        EXPECT_GT(Ld, prevLd);
        prevLd = Ld;
    }
}

// ============================================================
// HdrParams defaults
// ============================================================
TEST(HdrParams, Defaults) {
    HdrParams p;
    EXPECT_FLOAT_EQ(p.exposure,   1.0f);
    EXPECT_FLOAT_EQ(p.gamma,      2.2f);
    EXPECT_FLOAT_EQ(p.saturation, 1.2f);
    EXPECT_EQ(p.algorithm, ToneMappingAlgorithm::REINHARD);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
