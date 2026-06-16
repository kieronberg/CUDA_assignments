#include "input_args_parser.h"
#include "../../external/cxxopts/include/cxxopts.hpp"
#include <iostream>

// ------------------------------------------------------------------
// Convert InputArgs → HdrParams
// ------------------------------------------------------------------
HdrParams InputArgs::toHdrParams() const {
    HdrParams hp;
    hp.exposure   = exposure;
    hp.gamma      = gamma;
    hp.saturation = saturation;
    hp.whitePoint = whitePoint;
    hp.bias       = dragoBias;
    hp.localRadius = localRadius;
    hp.localSigmaS = localSigmaS;
    hp.localSigmaR = localSigmaR;
    hp.algorithm  = stringToToneMappingAlgorithm(toneMappingAlg);
    return hp;
}

// ------------------------------------------------------------------
// CLI parser
// ------------------------------------------------------------------
bool parseArgs(int argc, char** argv, InputArgs& out) {
    try {
        cxxopts::Options opts("cuda-webcam-filter",
            "Real-time CUDA webcam filter with HDR tone-mapping");

        opts.add_options()
            // ---- input ----
            ("i,input",     "Input source: webcam, image, video, synthetic",
             cxxopts::value<std::string>()->default_value("webcam"))
            ("p,path",      "Path to image or video file",
             cxxopts::value<std::string>()->default_value("test_image.jpg"))
            ("s,synthetic", "Synthetic pattern: checkerboard, gradient, noise",
             cxxopts::value<std::string>()->default_value("checkerboard"))
            ("d,device",    "Camera device ID",
             cxxopts::value<int>()->default_value("0"))

            // ---- filter ----
            ("f,filter",       "Filter: blur, sharpen, edge, emboss, hdr",
             cxxopts::value<std::string>()->default_value("blur"))
            ("k,kernel-size",  "Kernel size (convolution filters)",
             cxxopts::value<int>()->default_value("3"))
            ("sigma",          "Sigma for Gaussian blur",
             cxxopts::value<float>()->default_value("1.0"))
            ("intensity",      "Filter intensity",
             cxxopts::value<float>()->default_value("1.0"))
            ("preview",        "Show original alongside filtered")

            // ---- HDR ----
            ("tone-mapping",   "HDR algorithm: reinhard, drago, local",
             cxxopts::value<std::string>()->default_value("reinhard"))
            ("exposure",       "Linear EV multiplier (default 1.0)",
             cxxopts::value<float>()->default_value("1.0"))
            ("gamma",          "Display gamma (default 2.2)",
             cxxopts::value<float>()->default_value("2.2"))
            ("saturation",     "Chroma saturation multiplier (default 1.2)",
             cxxopts::value<float>()->default_value("1.2"))
            ("white-point",    "Reinhard white point (default 4.0)",
             cxxopts::value<float>()->default_value("4.0"))
            ("drago-bias",     "Drago bias 0-1 (default 0.85)",
             cxxopts::value<float>()->default_value("0.85"))
            ("local-radius",   "Local TM bilateral radius (default 15)",
             cxxopts::value<int>()->default_value("15"))
            ("local-sigma-s",  "Bilateral spatial sigma (default 10.0)",
             cxxopts::value<float>()->default_value("10.0"))
            ("local-sigma-r",  "Bilateral range sigma (default 0.15)",
             cxxopts::value<float>()->default_value("0.15"))

            // ---- misc ----
            ("benchmark",      "Print GPU vs CPU timing each frame")
            ("h,help",         "Print usage")
            ("v,version",      "Print version");

        auto result = opts.parse(argc, argv);

        if (result.count("help")) {
            std::cout << opts.help() << "\n";
            return false;
        }
        if (result.count("version")) {
            std::cout << "cuda-webcam-filter v1.1.0 (HDR edition)\n";
            return false;
        }

        out.inputSource   = result["input"].as<std::string>();
        out.inputPath     = result["path"].as<std::string>();
        out.syntheticPat  = result["synthetic"].as<std::string>();
        out.deviceId      = result["device"].as<int>();
        out.filterType    = result["filter"].as<std::string>();
        out.kernelSize    = result["kernel-size"].as<int>();
        out.sigma         = result["sigma"].as<float>();
        out.intensity     = result["intensity"].as<float>();
        out.preview       = result.count("preview") > 0;

        out.toneMappingAlg = result["tone-mapping"].as<std::string>();
        out.exposure       = result["exposure"].as<float>();
        out.gamma          = result["gamma"].as<float>();
        out.saturation     = result["saturation"].as<float>();
        out.whitePoint     = result["white-point"].as<float>();
        out.dragoBias      = result["drago-bias"].as<float>();
        out.localRadius    = result["local-radius"].as<int>();
        out.localSigmaS    = result["local-sigma-s"].as<float>();
        out.localSigmaR    = result["local-sigma-r"].as<float>();
        out.benchmark      = result.count("benchmark") > 0;

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n";
        return false;
    }
    return true;
}
