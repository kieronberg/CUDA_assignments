#include "filter_utils.h"

// External kernel launch declarations
extern void launchBlur(const uchar3* in, uchar3* out, int w, int h, float sigma, cudaStream_t s);
extern void launchEdge(const uchar3* in, uchar3* out, int w, int h, cudaStream_t s);
extern void launchWipe(const uchar3* inA, const uchar3* inB, uchar3* out, int w, int h, float p, cudaStream_t s);

uchar3* FilterPipeline::execute(const uchar3* d_input) {
    uchar3* currentIn = const_cast<uchar3*>(d_input);
    uchar3* currentOut = d_bufferA;

    for (const auto& stage : stages) {
        switch (stage.type) {
            case FilterType::Blur: launchBlur(currentIn, currentOut, width, height, stage.param1, streamProcess); break;
            case FilterType::EdgeDetect: launchEdge(currentIn, currentOut, width, height, streamProcess); break;
            case FilterType::WipeTransition: 
                // Assumes d_bufferB holds the target frame
                launchWipe(currentIn, d_bufferB, currentOut, width, height, stage.param1, streamProcess); 
                break;
        }
        std::swap(currentIn, currentOut);
    }
    return currentIn; // Returns pointer to the final buffer containing result
}