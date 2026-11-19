// nvinfer custom bbox parser for Ultralytics YOLO (raw and end-to-end ONNX exports).
// Build inside the DeepStream container: `make -C deepstream/parser` (see Makefile).
// STATUS: decode logic is unit-tested (test_decode.cpp). This DeepStream glue has NOT been
// compiled against the real headers yet (no DeepStream in the authoring environment).
#include <cstdio>
#include <vector>

#include "nvdsinfer_custom_impl.h"
#include "yolo_decode.h"

extern "C" bool NvDsInferParseYoloUltralytics(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList) {
    if (outputLayersInfo.empty()) {
        std::fprintf(stderr, "[yolo parser] no output layers\n");
        return false;
    }
    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    std::vector<int> dims;
    for (unsigned i = 0; i < layer.inferDims.numDims; ++i) dims.push_back(layer.inferDims.d[i]);

    const int numClasses = static_cast<int>(detectionParams.numClassesConfigured);
    const auto layout = vi::detectLayout(dims, numClasses);
    const float *data = static_cast<const float *>(layer.buffer);
    std::vector<vi::Box> boxes;
    const float fallback = 0.25f;

    if (layout == vi::Layout::Raw) {
        int anchors = dims.back();
        vi::decodeRaw(data, anchors, numClasses, detectionParams.perClassPreclusterThreshold,
                      fallback, boxes);
    } else if (layout == vi::Layout::End2End) {
        int detections = dims[dims.size() - 2];
        vi::decodeEnd2End(data, detections, numClasses,
                          detectionParams.perClassPreclusterThreshold, fallback, boxes);
    } else {
        std::fprintf(stderr,
                     "[yolo parser] unsupported output dims (numDims=%u) for %d classes\n",
                     layer.inferDims.numDims, numClasses);
        return false;
    }

    vi::clampToNetwork(boxes, static_cast<float>(networkInfo.width),
                       static_cast<float>(networkInfo.height));
    objectList.reserve(boxes.size());
    for (const auto &b : boxes) {
        NvDsInferParseObjectInfo obj;
        obj.classId = static_cast<unsigned>(b.classId);
        obj.left = b.left;
        obj.top = b.top;
        obj.width = b.width;
        obj.height = b.height;
        obj.detectionConfidence = b.confidence;
        objectList.push_back(obj);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloUltralytics);
