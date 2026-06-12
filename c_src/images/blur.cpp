#include "blur.h"

#include <cmath>
#include <cstring>
#include <vector>

#define __my_countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
#if defined(_countof)
#undef _countof
#endif
// Just making sure
#define _countof __my_countof

namespace
{
struct BoxSizes
{
    // Three box widths that approximate the requested Gaussian sigma.
    int sizes[3];
};

static inline BoxSizes gaussian_to_box_sizes(double sigma)
{
    const double n = 3.0;
    double wIdeal = std::sqrt((12.0 * sigma * sigma / n) + 1.0);
    int wl = static_cast<int>(std::floor(wIdeal));
    if ((wl & 1) == 0)
    {
        wl -= 1;
    }
    int wu = wl + 2;

    double mIdeal =
        (12.0 * sigma * sigma - n * wl * wl - 4.0 * n * wl - 3.0 * n) / (-4.0 * wl - 4.0);
    int m = static_cast<int>(std::round(mIdeal));

    BoxSizes bs;
    for (int i = 0; i < 3; ++i)
    {
        bs.sizes[i] = (i < m) ? wl : wu;
    }
    return bs;
}

static inline int clamp_int(int v, int lo, int hi)
{
    if (v < lo)
    {
        return lo;
    }
    if (v > hi)
    {
        return hi;
    }
    return v;
}

struct ImgView
{
    // Base pointer to the first byte of image data.
    uint8_t* data;
    // Image width/height measured in pixels.
    int w;
    int h;
    // Distance in bytes between the start of two consecutive rows.
    int stride;
    // Number of channels per pixel (1 = A8, 4 = PRGB32).
    int channels;
};

//"Tightly packed" here means each scanline
// of the image is stored back-to-back with no padding bytes between rows.
struct BlurScratch
{
    // Reusable tight-packed working buffers to avoid allocations per call.
    std::vector<uint8_t> buf_a;
    std::vector<uint8_t> buf_b;
};

static thread_local BlurScratch blur_scratch;

static void box_blur_h(const ImgView& src, const ImgView& dst, int radius)
{
    const int dia = radius * 2 + 1;
    for (int y = 0; y < src.h; ++y)
    {
        // Pointer to the current source/destination row.
        const uint8_t* srow = src.data + static_cast<size_t>(y) * src.stride;
        uint8_t* drow = dst.data + static_cast<size_t>(y) * dst.stride;

        // Maintain a sliding window sum over the row for each channel.
        std::vector<int> sums(src.channels, 0);
        for (int i = -radius; i <= radius; ++i)
        {
            int ix = clamp_int(i, 0, src.w - 1);
            const uint8_t* p = srow + static_cast<size_t>(ix) * static_cast<size_t>(src.channels);
            for (int c = 0; c < src.channels; ++c)
            {
                sums[c] += p[c];
            }
        }

        for (int x = 0; x < src.w; ++x)
        {
            uint8_t* dst_px = drow + static_cast<size_t>(x) * static_cast<size_t>(src.channels);
            for (int c = 0; c < src.channels; ++c)
            {
                dst_px[c] = static_cast<uint8_t>((sums[c] + dia / 2) / dia);
            }

            // Slide the window: add the next pixel and remove the previous.
            int next = clamp_int(x + radius + 1, 0, src.w - 1);
            int prev = clamp_int(x - radius, 0, src.w - 1);

            const uint8_t* pNext = srow + static_cast<size_t>(next) * static_cast<size_t>(src.channels);
            const uint8_t* pPrev = srow + static_cast<size_t>(prev) * static_cast<size_t>(src.channels);

            for (int c = 0; c < src.channels; ++c)
            {
                sums[c] += pNext[c] - pPrev[c];
            }
        }
    }
}

static void box_blur_v(const ImgView& src, const ImgView& dst, int radius)
{
    const int dia = radius * 2 + 1;
    for (int x = 0; x < src.w; ++x)
    {
        // Column-wise sliding window sums per channel.
        std::vector<int> sums(src.channels, 0);
        for (int i = -radius; i <= radius; ++i)
        {
            int iy = clamp_int(i, 0, src.h - 1);
            const uint8_t* p = src.data + static_cast<size_t>(iy) * src.stride +
                               static_cast<size_t>(x) * static_cast<size_t>(src.channels);
            for (int c = 0; c < src.channels; ++c)
            {
                sums[c] += p[c];
            }
        }

        for (int y = 0; y < src.h; ++y)
        {
            uint8_t* dst_px = dst.data + static_cast<size_t>(y) * dst.stride +
                              static_cast<size_t>(x) * static_cast<size_t>(src.channels);
            for (int c = 0; c < src.channels; ++c)
            {
                dst_px[c] = static_cast<uint8_t>((sums[c] + dia / 2) / dia);
            }

            int next = clamp_int(y + radius + 1, 0, src.h - 1);
            int prev = clamp_int(y - radius, 0, src.h - 1);

            const uint8_t* pNext = src.data + static_cast<size_t>(next) * src.stride +
                                   static_cast<size_t>(x) * static_cast<size_t>(src.channels);
            const uint8_t* pPrev = src.data + static_cast<size_t>(prev) * src.stride +
                                   static_cast<size_t>(x) * static_cast<size_t>(src.channels);

            for (int c = 0; c < src.channels; ++c)
            {
                sums[c] += pNext[c] - pPrev[c];
            }
        }
    }
}
}  // namespace

BLResult blur_image_inplace(BLImage& img, const double sigma, const BLRectI& rect)
{
    // Approximates a Gaussian blur by applying three box blurs horizontally/vertically.
    if (sigma <= 0.0)
    {
        return BL_ERROR_INVALID_VALUE;
    }

    BLSizeI sz = img.size();
    if (sz.w == 0 || sz.h == 0)
    {
        return BL_SUCCESS;
    }

    BLFormat fmt = img.format();
    int channels = 0;
    switch (fmt)
    {
        case BL_FORMAT_PRGB32:
            channels = 4;
            break;
        case BL_FORMAT_A8:
            channels = 1;
            break;
        default:
            return BL_ERROR_INVALID_STATE;
    }

    BLImageData data{};
    BLResult r = img.get_data(&data);
    if (r != BL_SUCCESS)
    {
        return r;
    }

    // Clamp the requested rect to the image bounds.
    const int rx = clamp_int(rect.x, 0, sz.w);
    const int ry = clamp_int(rect.y, 0, sz.h);
    const int rx2 = clamp_int(rect.x + rect.w, 0, sz.w);
    const int ry2 = clamp_int(rect.y + rect.h, 0, sz.h);
    const int eff_w = rx2 - rx;
    const int eff_h = ry2 - ry;

    if (eff_w <= 0 || eff_h <= 0)
    {
        return BL_SUCCESS;
    }

    // Compute the bytes per tight row and total buffer size for the sub-region.
    const size_t row_bytes = static_cast<size_t>(eff_w) * static_cast<size_t>(channels);
    const size_t buf_bytes = row_bytes * static_cast<size_t>(eff_h);

    BlurScratch& scratch = blur_scratch;
    if (scratch.buf_a.size() < buf_bytes)
    {
        scratch.buf_a.resize(buf_bytes);
    }
    if (scratch.buf_b.size() < buf_bytes)
    {
        scratch.buf_b.resize(buf_bytes);
    }

    uint8_t* src_data = static_cast<uint8_t*>(data.pixel_data);

    // Pointer to the first pixel of the rect in the original image data.
    uint8_t* rect_origin = src_data + static_cast<size_t>(ry) * data.stride + static_cast<size_t>(rx) * static_cast<size_t>(channels);

    // Copy the rect rows into a tightly-packed working buffer.
    uint8_t* packed_src = scratch.buf_a.data();
    for (int y = 0; y < eff_h; ++y)
    {
        std::memcpy(packed_src + static_cast<size_t>(y) * row_bytes,
                    rect_origin + static_cast<size_t>(y) * data.stride,
                    row_bytes);
    }

    ImgView src{packed_src, eff_w, eff_h, static_cast<int>(row_bytes), channels};
    ImgView dst{scratch.buf_b.data(), eff_w, eff_h, static_cast<int>(row_bytes), channels};

    BoxSizes boxes = gaussian_to_box_sizes(sigma);
    for (size_t bi = 0; bi < _countof(boxes.sizes); ++bi)
    {
        int radius = boxes.sizes[bi] / 2;
        box_blur_h(src, dst, radius);
        box_blur_v(dst, src, radius);
    }

    // Write the blurred rect rows back into the image at the correct offset.
    for (int y = 0; y < eff_h; ++y)
    {
        std::memcpy(rect_origin + static_cast<size_t>(y) * data.stride,
                    packed_src + static_cast<size_t>(y) * row_bytes,
                    row_bytes);
    }

    return BL_SUCCESS;
}
