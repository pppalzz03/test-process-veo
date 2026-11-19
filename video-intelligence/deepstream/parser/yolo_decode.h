// Ultralytics YOLO output decoding. Header-only and independent of DeepStream so it can be
// compiled and unit-tested anywhere (see test_decode.cpp).
//
// Two ONNX output layouts (batch dimension already stripped by nvinfer):
//   Raw        [4 + numClasses, numAnchors]  rows: cx, cy, w, h, class scores...   (YOLOv8/11)
//              needs NMS -> use cluster-mode=2 in the nvinfer config
//   End2End    [numDetections, 6]            rows: x1, y1, x2, y2, score, classId  (YOLOv10/26)
//              NMS already applied in the model -> use cluster-mode=4 (no clustering)
#pragma once

#include <algorithm>
#include <vector>

namespace vi {

struct Box {
    int classId;
    float left, top, width, height;
    float confidence;
};

enum class Layout { Raw, End2End, Unknown };

// Heuristic on tensor dims. A 2-class model has 4+2 == 6 channels, which equals the End2End
// column count; the anchor/detection count disambiguates (raw has thousands of anchors).
inline Layout detectLayout(const std::vector<int> &dims, int numClasses) {
    std::vector<int> d;
    for (size_t i = 0; i < dims.size(); ++i)
        if (!(i == 0 && dims.size() == 3 && dims[0] == 1)) d.push_back(dims[i]);  // drop batch
    if (d.size() != 2) return Layout::Unknown;
    if (d[0] == 4 + numClasses && d[0] < d[1] && d[1] > 1000) return Layout::Raw;
    if (d[1] == 6 && d[0] <= 1000) return Layout::End2End;
    if (d[0] == 4 + numClasses && d[0] < d[1]) return Layout::Raw;
    return Layout::Unknown;
}

inline float thresholdFor(const std::vector<float> &perClass, int classId, float fallback) {
    return (classId >= 0 && classId < static_cast<int>(perClass.size())) ? perClass[classId]
                                                                          : fallback;
}

// data is channels-first: value(channel c, anchor i) = data[c * numAnchors + i]
inline void decodeRaw(const float *data, int numAnchors, int numClasses,
                      const std::vector<float> &perClassThreshold, float fallbackThreshold,
                      std::vector<Box> &out) {
    for (int i = 0; i < numAnchors; ++i) {
        int best = 0;
        float bestScore = data[4 * numAnchors + i];
        for (int c = 1; c < numClasses; ++c) {
            float s = data[(4 + c) * numAnchors + i];
            if (s > bestScore) { bestScore = s; best = c; }
        }
        if (bestScore < thresholdFor(perClassThreshold, best, fallbackThreshold)) continue;
        float cx = data[0 * numAnchors + i], cy = data[1 * numAnchors + i];
        float w = data[2 * numAnchors + i], h = data[3 * numAnchors + i];
        out.push_back({best, cx - w / 2, cy - h / 2, w, h, bestScore});
    }
}

inline void decodeEnd2End(const float *data, int numDetections, int numClasses,
                          const std::vector<float> &perClassThreshold, float fallbackThreshold,
                          std::vector<Box> &out) {
    for (int i = 0; i < numDetections; ++i) {
        const float *row = data + i * 6;
        int classId = static_cast<int>(row[5]);
        if (classId < 0 || classId >= numClasses) continue;
        if (row[4] < thresholdFor(perClassThreshold, classId, fallbackThreshold)) continue;
        float w = row[2] - row[0], h = row[3] - row[1];
        if (w <= 0 || h <= 0) continue;
        out.push_back({classId, row[0], row[1], w, h, row[4]});
    }
}

// Keep boxes inside the network input so downstream scaling never sees negative sizes.
inline void clampToNetwork(std::vector<Box> &boxes, float netW, float netH) {
    for (auto &b : boxes) {
        float l = std::max(0.f, b.left), t = std::max(0.f, b.top);
        float r = std::min(netW, b.left + b.width), btm = std::min(netH, b.top + b.height);
        b.left = l; b.top = t;
        b.width = std::max(0.f, r - l); b.height = std::max(0.f, btm - t);
    }
    boxes.erase(std::remove_if(boxes.begin(), boxes.end(),
                               [](const Box &b) { return b.width < 1.f || b.height < 1.f; }),
                boxes.end());
}

}  // namespace vi
