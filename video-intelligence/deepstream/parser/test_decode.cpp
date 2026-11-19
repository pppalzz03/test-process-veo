// Host-side unit test for yolo_decode.h:  g++ -std=c++17 -o /tmp/t test_decode.cpp && /tmp/t
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "yolo_decode.h"

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

static bool near(float a, float b) { return std::fabs(a - b) < 1e-4f; }

int main() {
    using namespace vi;

    // --- layout detection --------------------------------------------------------------
    CHECK(detectLayout({84, 8400}, 80) == Layout::Raw);
    CHECK(detectLayout({1, 84, 8400}, 80) == Layout::Raw);
    CHECK(detectLayout({300, 6}, 80) == Layout::End2End);
    CHECK(detectLayout({1, 300, 6}, 80) == Layout::End2End);
    CHECK(detectLayout({6, 8400}, 2) == Layout::Raw);       // 2-class raw: 4+2 == 6 channels
    CHECK(detectLayout({300, 6}, 2) == Layout::End2End);    // ...must not be confused with e2e
    CHECK(detectLayout({10}, 80) == Layout::Unknown);

    // --- raw decode: 2 classes, 3 anchors, channels-first ------------------------------
    // anchor0: class0 score .9   anchor1: class1 score .6   anchor2: below threshold
    const float raw[6 * 3] = {
        /*cx*/ 100, 200, 300,
        /*cy*/ 100, 200, 300,
        /*w */ 40, 20, 10,
        /*h */ 60, 30, 10,
        /*c0*/ 0.9f, 0.1f, 0.05f,
        /*c1*/ 0.2f, 0.6f, 0.1f};
    std::vector<Box> out;
    decodeRaw(raw, 3, 2, {}, 0.25f, out);
    CHECK(out.size() == 2);
    CHECK(out[0].classId == 0 && near(out[0].left, 80) && near(out[0].top, 70) &&
          near(out[0].width, 40) && near(out[0].height, 60) && near(out[0].confidence, 0.9f));
    CHECK(out[1].classId == 1 && near(out[1].confidence, 0.6f));

    // per-class threshold overrides the fallback
    out.clear();
    decodeRaw(raw, 3, 2, {0.95f, 0.5f}, 0.25f, out);
    CHECK(out.size() == 1 && out[0].classId == 1);

    // --- end-to-end decode -------------------------------------------------------------
    const float e2e[3 * 6] = {10, 20, 110, 220, 0.8f, 3,    // valid
                              0,  0,  50,  50,  0.1f, 1,    // below threshold
                              5,  5,  5,   50,  0.9f, 2};   // zero width -> dropped
    out.clear();
    decodeEnd2End(e2e, 3, 80, {}, 0.25f, out);
    CHECK(out.size() == 1 && out[0].classId == 3 && near(out[0].width, 100) &&
          near(out[0].height, 200));

    // --- clamping ------------------------------------------------------------------------
    std::vector<Box> boxes = {{0, -10, -10, 50, 50, 0.9f}, {0, 630, 630, 50, 50, 0.9f},
                              {0, 700, 700, 20, 20, 0.9f}};
    clampToNetwork(boxes, 640, 640);
    CHECK(boxes.size() == 2);  // fully outside box removed
    CHECK(near(boxes[0].left, 0) && near(boxes[0].width, 40));
    CHECK(near(boxes[1].width, 10) && near(boxes[1].height, 10));

    std::puts("test_decode: all checks passed");
    return 0;
}
