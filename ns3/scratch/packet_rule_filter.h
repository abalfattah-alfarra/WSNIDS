#ifndef PACKET_RULE_FILTER_H
#define PACKET_RULE_FILTER_H
#include <stdint.h>
#include <cmath>

namespace ids {

static const uint8_t WIN = 50;
static uint8_t dstHist[WIN];
static uint8_t idxHist = 0;
static uint32_t lastTs = 0;
static float    meanIat = 1000.0;   // ms (updated online)

/*–– entropy ––*/
inline float Entropy ()
{
    uint16_t cnt[256] = {0};
    for (uint8_t i=0;i<WIN;i++) cnt[dstHist[i]]++;
    float H = 0.f, invWin = 1.f/WIN;
    for (uint16_t c : cnt)
        if (c) { float p = c*invWin; H -= p*log2f(p); }
    return H;
}

/*–– main rule ––*/
inline bool FilterPacket (uint8_t dst, uint32_t tsNow, uint16_t battMv)
{
    // battery bypass
    if (battMv < 3300) return false;

    // update hist
    dstHist[idxHist++ % WIN] = dst;

    // inter-arrival variance rule
    uint32_t iat = tsNow - lastTs;  lastTs = tsNow;
    meanIat = 0.9f*meanIat + 0.1f*static_cast<float>(iat);
    if (iat > 3*meanIat) return true;

    // entropy rule
    if (Entropy() < 0.20f) return true;

    return false;
}
}  // namespace ids
#endif
