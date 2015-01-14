/*
 * Copyright (c) 2014-2015 Michael Niedermayer <michaelni@gmx.at>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with FFmpeg; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "libavutil/avassert.h"
// #include "libavutil/cpu.h"
// #include "libavutil/common.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/rational.h"
// #include "libavutil/imgutils.h"

#include "libavcodec/avcodec.h"
#include "libavcodec/internal.h"

#include "avfilter.h"
// #include "formats.h"
#include "internal.h"
// #include "video.h"

// TODO, clone last frame at EOF so no frames are lost
// TODO >8bit support
// TODO use SIMD for MC
// TODO use some linesize instead of w where SIMD is used

#define NB_INPUT_FRAMES 4
#define ALPHA_MAX 1024
#define LOG2_MB_SIZE 4

enum MCFPSMode {
    MCFPS_MODE_NN           = 0,
    MCFPS_MODE_LINEAR_IPOL  = 1,
    MCFPS_MODE_GMC          = 2,
    MCFPS_MODE_OBMC         = 3,
};

#define NB_MVS 20
typedef struct Pixel {
    int16_t mv[NB_MVS][2];
    uint32_t weight[NB_MVS];
    int8_t ref[NB_MVS];
    uint8_t layer[NB_MVS];
    int nb;
} Pixel;

typedef struct InputFrame {
    AVFrame *f;
    uint8_t *halfpel[4][4];
    int halfpel_linesize[4];
    int16_t (*mv[2])[2];
    int8_t *ref[2];
} InputFrame;

typedef struct MCFPSContext {
    const AVClass *class;

    AVRational       frame_rate;
    enum MCFPSMode   mode;

    InputFrame input[NB_INPUT_FRAMES];

    int64_t out_pts;

    int chroma_h_shift;
    int chroma_v_shift;
    int planes;

    AVCodecContext *avctx_enc[2];
    uint8_t *outbuf;
    int outbuf_size;
    int b_width, b_height;
    int log2_mv_precission;

    Pixel *pixel;
} MCFPSContext;

static const uint8_t obmc32[1024]={
  0,  0,  0,  0,  4,  4,  4,  4,  4,  4,  4,  4,  8,  8,  8,  8,  8,  8,  8,  8,  4,  4,  4,  4,  4,  4,  4,  4,  0,  0,  0,  0,
  0,  4,  4,  4,  8,  8,  8, 12, 12, 16, 16, 16, 20, 20, 20, 24, 24, 20, 20, 20, 16, 16, 16, 12, 12,  8,  8,  8,  4,  4,  4,  0,
  0,  4,  8,  8, 12, 12, 16, 20, 20, 24, 28, 28, 32, 32, 36, 40, 40, 36, 32, 32, 28, 28, 24, 20, 20, 16, 12, 12,  8,  8,  4,  0,
  0,  4,  8, 12, 16, 20, 24, 28, 28, 32, 36, 40, 44, 48, 52, 56, 56, 52, 48, 44, 40, 36, 32, 28, 28, 24, 20, 16, 12,  8,  4,  0,
  4,  8, 12, 16, 20, 24, 28, 32, 40, 44, 48, 52, 56, 60, 64, 68, 68, 64, 60, 56, 52, 48, 44, 40, 32, 28, 24, 20, 16, 12,  8,  4,
  4,  8, 12, 20, 24, 32, 36, 40, 48, 52, 56, 64, 68, 76, 80, 84, 84, 80, 76, 68, 64, 56, 52, 48, 40, 36, 32, 24, 20, 12,  8,  4,
  4,  8, 16, 24, 28, 36, 44, 48, 56, 60, 68, 76, 80, 88, 96,100,100, 96, 88, 80, 76, 68, 60, 56, 48, 44, 36, 28, 24, 16,  8,  4,
  4, 12, 20, 28, 32, 40, 48, 56, 64, 72, 80, 88, 92,100,108,116,116,108,100, 92, 88, 80, 72, 64, 56, 48, 40, 32, 28, 20, 12,  4,
  4, 12, 20, 28, 40, 48, 56, 64, 72, 80, 88, 96,108,116,124,132,132,124,116,108, 96, 88, 80, 72, 64, 56, 48, 40, 28, 20, 12,  4,
  4, 16, 24, 32, 44, 52, 60, 72, 80, 92,100,108,120,128,136,148,148,136,128,120,108,100, 92, 80, 72, 60, 52, 44, 32, 24, 16,  4,
  4, 16, 28, 36, 48, 56, 68, 80, 88,100,112,120,132,140,152,164,164,152,140,132,120,112,100, 88, 80, 68, 56, 48, 36, 28, 16,  4,
  4, 16, 28, 40, 52, 64, 76, 88, 96,108,120,132,144,156,168,180,180,168,156,144,132,120,108, 96, 88, 76, 64, 52, 40, 28, 16,  4,
  8, 20, 32, 44, 56, 68, 80, 92,108,120,132,144,156,168,180,192,192,180,168,156,144,132,120,108, 92, 80, 68, 56, 44, 32, 20,  8,
  8, 20, 32, 48, 60, 76, 88,100,116,128,140,156,168,184,196,208,208,196,184,168,156,140,128,116,100, 88, 76, 60, 48, 32, 20,  8,
  8, 20, 36, 52, 64, 80, 96,108,124,136,152,168,180,196,212,224,224,212,196,180,168,152,136,124,108, 96, 80, 64, 52, 36, 20,  8,
  8, 24, 40, 56, 68, 84,100,116,132,148,164,180,192,208,224,240,240,224,208,192,180,164,148,132,116,100, 84, 68, 56, 40, 24,  8,
  8, 24, 40, 56, 68, 84,100,116,132,148,164,180,192,208,224,240,240,224,208,192,180,164,148,132,116,100, 84, 68, 56, 40, 24,  8,
  8, 20, 36, 52, 64, 80, 96,108,124,136,152,168,180,196,212,224,224,212,196,180,168,152,136,124,108, 96, 80, 64, 52, 36, 20,  8,
  8, 20, 32, 48, 60, 76, 88,100,116,128,140,156,168,184,196,208,208,196,184,168,156,140,128,116,100, 88, 76, 60, 48, 32, 20,  8,
  8, 20, 32, 44, 56, 68, 80, 92,108,120,132,144,156,168,180,192,192,180,168,156,144,132,120,108, 92, 80, 68, 56, 44, 32, 20,  8,
  4, 16, 28, 40, 52, 64, 76, 88, 96,108,120,132,144,156,168,180,180,168,156,144,132,120,108, 96, 88, 76, 64, 52, 40, 28, 16,  4,
  4, 16, 28, 36, 48, 56, 68, 80, 88,100,112,120,132,140,152,164,164,152,140,132,120,112,100, 88, 80, 68, 56, 48, 36, 28, 16,  4,
  4, 16, 24, 32, 44, 52, 60, 72, 80, 92,100,108,120,128,136,148,148,136,128,120,108,100, 92, 80, 72, 60, 52, 44, 32, 24, 16,  4,
  4, 12, 20, 28, 40, 48, 56, 64, 72, 80, 88, 96,108,116,124,132,132,124,116,108, 96, 88, 80, 72, 64, 56, 48, 40, 28, 20, 12,  4,
  4, 12, 20, 28, 32, 40, 48, 56, 64, 72, 80, 88, 92,100,108,116,116,108,100, 92, 88, 80, 72, 64, 56, 48, 40, 32, 28, 20, 12,  4,
  4,  8, 16, 24, 28, 36, 44, 48, 56, 60, 68, 76, 80, 88, 96,100,100, 96, 88, 80, 76, 68, 60, 56, 48, 44, 36, 28, 24, 16,  8,  4,
  4,  8, 12, 20, 24, 32, 36, 40, 48, 52, 56, 64, 68, 76, 80, 84, 84, 80, 76, 68, 64, 56, 52, 48, 40, 36, 32, 24, 20, 12,  8,  4,
  4,  8, 12, 16, 20, 24, 28, 32, 40, 44, 48, 52, 56, 60, 64, 68, 68, 64, 60, 56, 52, 48, 44, 40, 32, 28, 24, 20, 16, 12,  8,  4,
  0,  4,  8, 12, 16, 20, 24, 28, 28, 32, 36, 40, 44, 48, 52, 56, 56, 52, 48, 44, 40, 36, 32, 28, 28, 24, 20, 16, 12,  8,  4,  0,
  0,  4,  8,  8, 12, 12, 16, 20, 20, 24, 28, 28, 32, 32, 36, 40, 40, 36, 32, 32, 28, 28, 24, 20, 20, 16, 12, 12,  8,  8,  4,  0,
  0,  4,  4,  4,  8,  8,  8, 12, 12, 16, 16, 16, 20, 20, 20, 24, 24, 20, 20, 20, 16, 16, 16, 12, 12,  8,  8,  8,  4,  4,  4,  0,
  0,  0,  0,  0,  4,  4,  4,  4,  4,  4,  4,  4,  8,  8,  8,  8,  8,  8,  8,  8,  4,  4,  4,  4,  4,  4,  4,  4,  0,  0,  0,  0,
};

static const uint8_t obmc16[256]={
  0,  4,  4,  8,  8, 12, 12, 16, 16, 12, 12,  8,  8,  4,  4,  0,
  4,  8, 16, 20, 28, 32, 40, 44, 44, 40, 32, 28, 20, 16,  8,  4,
  4, 16, 24, 36, 44, 56, 64, 76, 76, 64, 56, 44, 36, 24, 16,  4,
  8, 20, 36, 48, 64, 76, 92,104,104, 92, 76, 64, 48, 36, 20,  8,
  8, 28, 44, 64, 80,100,116,136,136,116,100, 80, 64, 44, 28,  8,
 12, 32, 56, 76,100,120,144,164,164,144,120,100, 76, 56, 32, 12,
 12, 40, 64, 92,116,144,168,196,196,168,144,116, 92, 64, 40, 12,
 16, 44, 76,104,136,164,196,224,224,196,164,136,104, 76, 44, 16,
 16, 44, 76,104,136,164,196,224,224,196,164,136,104, 76, 44, 16,
 12, 40, 64, 92,116,144,168,196,196,168,144,116, 92, 64, 40, 12,
 12, 32, 56, 76,100,120,144,164,164,144,120,100, 76, 56, 32, 12,
  8, 28, 44, 64, 80,100,116,136,136,116,100, 80, 64, 44, 28,  8,
  8, 20, 36, 48, 64, 76, 92,104,104, 92, 76, 64, 48, 36, 20,  8,
  4, 16, 24, 36, 44, 56, 64, 76, 76, 64, 56, 44, 36, 24, 16,  4,
  4,  8, 16, 20, 28, 32, 40, 44, 44, 40, 32, 28, 20, 16,  8,  4,
  0,  4,  4,  8,  8, 12, 12, 16, 16, 12, 12,  8,  8,  4,  4,  0,
};

static const uint8_t obmc8[64]={
  4, 12, 20, 28, 28, 20, 12,  4,
 12, 36, 60, 84, 84, 60, 36, 12,
 20, 60,100,140,140,100, 60, 20,
 28, 84,140,196,196,140, 84, 28,
 28, 84,140,196,196,140, 84, 28,
 20, 60,100,140,140,100, 60, 20,
 12, 36, 60, 84, 84, 60, 36, 12,
  4, 12, 20, 28, 28, 20, 12,  4,
};

static const uint8_t obmc4[16]={
 16, 48, 48, 16,
 48,144,144, 48,
 48,144,144, 48,
 16, 48, 48, 16,
};

static const uint8_t * const obmc_tab[4]= {
    obmc32, obmc16, obmc8, obmc4
};

static int query_formats(AVFilterContext *ctx)
{
    static const enum PixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUV444P,  AV_PIX_FMT_YUV422P,
        AV_PIX_FMT_YUV420P,  AV_PIX_FMT_YUV411P,
        AV_PIX_FMT_YUV410P,  AV_PIX_FMT_YUV440P,
        AV_PIX_FMT_YUVJ444P, AV_PIX_FMT_YUVJ422P,
        AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUVJ440P,
/*        AV_PIX_FMT_GBRP,
        AV_PIX_FMT_GRAY8,    */AV_PIX_FMT_NONE
    };
    ff_set_common_formats(ctx, ff_make_format_list(pix_fmts));
    return 0;
}

av_cold static void uninit(AVFilterContext *ctx)
{
    MCFPSContext *mcfps = ctx->priv;
    int i, j, p;

    for (i=0; i<NB_INPUT_FRAMES; i++) {
        InputFrame *inf = &mcfps->input[i];
        av_freep(&inf->mv[0]);
        av_freep(&inf->mv[1]);
        av_freep(&inf->ref[0]);
        av_freep(&inf->ref[1]);
        av_frame_free(&inf->f);
        for (p = 0; p<mcfps->planes; p++)
            for (j=0; j<4; j++)
                av_freep(&inf->halfpel[p][j]);
    }

    for (i = 0; i<2; i++) {
        avcodec_close(mcfps->avctx_enc[i]);
        av_freep(&mcfps->avctx_enc[i]);
    }

    av_freep(&mcfps->pixel);

    av_freep(&mcfps->outbuf);
    mcfps->outbuf_size = 0;
}

av_cold static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx  = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];
    MCFPSContext *mcfps = ctx->priv;
    AVCodec *enc = avcodec_find_encoder(AV_CODEC_ID_SNOW);
    const int height = inlink->h;
    const int width  = inlink->w;
    int i;
    int ret;

    mcfps->log2_mv_precission = 2;

    outlink->flags |= FF_LINK_FLAG_REQUEST_LOOP;
    outlink->frame_rate = mcfps->frame_rate;
    outlink->time_base  = av_inv_q(mcfps->frame_rate);
av_log(0,0, "FPS %d/%d\n", mcfps->frame_rate.num, mcfps->frame_rate.den);
//     outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;
//     outlink->w = inlink->w;
//     outlink->h = inlink->h;

    mcfps->b_width  = FF_CEIL_RSHIFT(width,  LOG2_MB_SIZE);
    mcfps->b_height = FF_CEIL_RSHIFT(height, LOG2_MB_SIZE);

    if (!enc) {
        av_log(ctx, AV_LOG_ERROR, "SNOW encoder not found.\n");
        return AVERROR(EINVAL);
    }

    for (i = 0; i < 2; i++) {
        AVCodecContext *avctx_enc;
        AVDictionary *opts = NULL;

        if (!(mcfps->avctx_enc[i] = avcodec_alloc_context3(NULL))) {
            ret = AVERROR(ENOMEM);
            goto fail;
        }

        avctx_enc = mcfps->avctx_enc[i];
        avctx_enc->width = width;
        avctx_enc->height = height;
        avctx_enc->time_base = (AVRational){1,25};
        avctx_enc->gop_size = i ? 2 : INT_MAX;
        avctx_enc->max_b_frames = 0;
        avctx_enc->pix_fmt = inlink->format;
        avctx_enc->flags = CODEC_FLAG_QSCALE | CODEC_FLAG_LOW_DELAY;
        if (mcfps->log2_mv_precission > 1)
            avctx_enc->flags |= CODEC_FLAG_QPEL;
        avctx_enc->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;
        avctx_enc->global_quality = 12;
        avctx_enc->me_method = ME_ITER;
//         avctx_enc->dia_size = 16;
        avctx_enc->mb_decision = FF_MB_DECISION_RD;
        avctx_enc->scenechange_threshold = 2000000000;
        avctx_enc->me_sub_cmp =
        avctx_enc->me_cmp = FF_CMP_SATD;
        avctx_enc->mb_cmp = FF_CMP_SSE;

        av_dict_set(&opts, "no_bitstream", "1", 0);
        av_dict_set(&opts, "intra_penalty", "500", 0);
        ret = avcodec_open2(avctx_enc, enc, &opts);
        av_dict_free(&opts);
        if (ret < 0)
            goto fail;
        av_assert0(avctx_enc->codec);
    }

    mcfps->outbuf_size = (width + 16) * (height + 16) * 10;
    if (!(mcfps->outbuf = av_malloc(mcfps->outbuf_size))) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    if (!(mcfps->pixel = av_mallocz_array(width*height, sizeof(*mcfps->pixel)))) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    return 0;
fail:

    uninit(ctx);

    return ret;
}

static int get_temporal_mv_difference(MCFPSContext *mcfps, int dir, int mbx, int mby, int *nomv) {
    int x, y;
    int ref0 = mcfps->input[2-dir].ref[dir][mby*mcfps->b_width + mbx];
    int mvx0 = mcfps->input[2-dir].mv[dir][mby*mcfps->b_width + mbx][0];
    int mvy0 = mcfps->input[2-dir].mv[dir][mby*mcfps->b_width + mbx][1];
    int roughness = INT_MAX;
    int ref, mvx, mvy;
    int div = 1<<(mcfps->log2_mv_precission + LOG2_MB_SIZE);
    int dir1 = 1-dir;

    x = mbx + (mvx0 / div);
    y = mby + (mvy0 / div);
    x = av_clip(x, 0, mcfps->input[0].f->width - 1);
    y = av_clip(y, 0, mcfps->input[0].f->height - 1);

    ref = mcfps->input[2-dir1].ref[dir1][y*mcfps->b_width + x];
    if (ref < 0 || ref0 < 0) {
        (*nomv) ++;
        return 0;
    }

    av_assert1(ref0 == 0);

    mvx = -mcfps->input[2-dir1].mv[dir1][y*mcfps->b_width + x][0];
    mvy = -mcfps->input[2-dir1].mv[dir1][y*mcfps->b_width + x][1];

    return FFABS(mvx0 - mvx) + FFABS(mvy0 - mvy);
}

static int get_roughness(MCFPSContext *mcfps, int dir, int mbx, int mby) {
    int x, y;
    int ref0 = mcfps->input[2-dir].ref[dir][mby*mcfps->b_width + mbx];
    int mvx0 = mcfps->input[2-dir].mv[dir][mby*mcfps->b_width + mbx][0];
    int mvy0 = mcfps->input[2-dir].mv[dir][mby*mcfps->b_width + mbx][1];
    int roughness = INT_MAX;

    av_assert1(ref0 == 0);

    for (y = FFMAX(mby-1, 0); y < FFMIN(mby+2, mcfps->b_height); y++) {
        for (x = FFMAX(mbx-1, 0); x < FFMIN(mbx+2, mcfps->b_width); x++) {
            int d, ref, mvx, mvy;
            int dir1 = dir;
            if (x == mbx && y == mby) {
                dir1 = 1-dir;
            }
            ref = mcfps->input[2-dir].ref[dir1][y*mcfps->b_width + x];
            if (ref < 0)
                continue;
            mvx = mcfps->input[2-dir].mv[dir1][y*mcfps->b_width + x][0];
            mvy = mcfps->input[2-dir].mv[dir1][y*mcfps->b_width + x][1];
            if (dir != dir1) {
                mvx = -mvx;
                mvy = -mvy;
            }
            d = FFABS(mvx0 - mvx) + FFABS(mvy0 - mvy);
            roughness = FFMIN(roughness, d);
        }
    }
    return roughness;
}

static void fill_pixels(MCFPSContext *mcfps, int alpha)
{
    int x, y;
    int w = mcfps->input[0].f->width;
    int h = mcfps->input[0].f->height;
    int mby, mbx;
    int dir;
    int64_t temporal_diff = 0;
    int nomv = 0;

    for (y=0; y<h; y++){
        int ymax = (h-y-1)<<mcfps->log2_mv_precission;
        for (x=0; x<w; x++){
            int xmax = (w-x-1)<<mcfps->log2_mv_precission;
            Pixel *p = &mcfps->pixel[x + y*w];
            p->weight[0] = ALPHA_MAX-alpha;
            p->ref[0] = 1;
            p->mv[0][0] = 0;
            p->mv[0][1] = 0;
            p->weight[1] = alpha;
            p->ref[1] = 2;
            p->mv[1][0] = 0;
            p->mv[1][1] = 0;
            p->nb = 2;
//             p->nb = 0;
        }
    }

//FIXME remove outlier MVs ?
//FIXME fill areas which have no MVs
//FIXME change MVs t 1/16 earlier to improve precisiion

    for (dir = 0; dir<2; dir++) {
        for (mby=0; mby<mcfps->b_height; mby++) {
            for (mbx=0; mbx<mcfps->b_width; mbx++) {
                temporal_diff += get_temporal_mv_difference(mcfps, dir, mbx, mby, &nomv);
            }
        }
    }

    if (mcfps->b_height * mcfps->b_width * 5 / 8 > nomv)
    for (dir = 0; dir<2; dir++) {
        int a = dir ? alpha : (ALPHA_MAX-alpha);

        for (mby=0; mby<mcfps->b_height; mby++) {
            for (mbx=0; mbx<mcfps->b_width; mbx++) {
                int ref = mcfps->input[2-dir].ref[dir][mby*mcfps->b_width + mbx];
                int mvx = mcfps->input[2-dir].mv[dir][mby*mcfps->b_width + mbx][0];
                int mvy = mcfps->input[2-dir].mv[dir][mby*mcfps->b_width + mbx][1];
                int startx, starty, endx, endy;

                if(ref < 0)
                    continue;

                if (get_roughness(mcfps, dir, mbx, mby) > 32)
                    continue;

av_assert0(ref == 0);
                startx = (mbx<<LOG2_MB_SIZE) - (1<<LOG2_MB_SIZE)/2 + (mvx)*a/(ALPHA_MAX<<mcfps->log2_mv_precission);
                starty = (mby<<LOG2_MB_SIZE) - (1<<LOG2_MB_SIZE)/2 + (mvy)*a/(ALPHA_MAX<<mcfps->log2_mv_precission);
                endx = startx + (2<<LOG2_MB_SIZE);
                endy = starty + (2<<LOG2_MB_SIZE);

                startx = av_clip(startx, 0, w-1);
                starty = av_clip(starty, 0, h-1);
                endx = av_clip(endx, 0, w-1);
                endy = av_clip(endy, 0, h-1);

                if (dir) {
                    mvx = -mvx;
                    mvy = -mvy;
                }

                for (y = starty; y<endy; y++) {
                    int ymin =   -y   <<mcfps->log2_mv_precission;
                    int ymax = (h-y-1)<<mcfps->log2_mv_precission;
                    for (x = startx; x<endx; x++) {
                        int xmin =   -x   <<mcfps->log2_mv_precission;
                        int xmax = (w-x-1)<<mcfps->log2_mv_precission;
                        int obmc_weight = obmc_tab[4-LOG2_MB_SIZE][(x-startx) + ((y-starty)<<(1+LOG2_MB_SIZE))];
                        Pixel *p = &mcfps->pixel[x + y*w];
                        if (p->nb + 1 >= NB_MVS) //FIXME discrad the vector of lowest weight
                            continue;
                        p->ref[p->nb] = 1;
                        p->weight[p->nb] = obmc_weight * (ALPHA_MAX-alpha);
                        p->mv[p->nb][0] = av_clip((mvx * alpha) / ALPHA_MAX, xmin, xmax);
                        p->mv[p->nb][1] = av_clip((mvy * alpha) / ALPHA_MAX, ymin, ymax);
                        p->nb ++;

                        p->ref[p->nb] = 2;
                        p->weight[p->nb] = obmc_weight * alpha;
                        p->mv[p->nb][0] = av_clip(-(mvx * (ALPHA_MAX-alpha)) / ALPHA_MAX, xmin, xmax);
                        p->mv[p->nb][1] = av_clip(-(mvy * (ALPHA_MAX-alpha)) / ALPHA_MAX, ymin, ymax);
                        p->nb ++;
                    }
                }
            }
        }
    }
}

// this should be optimized but dont do premature optims, first find out what is best
static int mc_sample(MCFPSContext *mcfps, Pixel *pixel, int plane, int x, int y, int i)
{
    int ref = pixel->ref[i];
    int is_chroma = plane == 1 || plane == 2;
    int mvx = pixel->mv[i][0] << (4 - mcfps->log2_mv_precission - is_chroma); // 1/16pel precission
    int mvy = pixel->mv[i][1] << (4 - mcfps->log2_mv_precission - is_chroma); // 1/16pel precission
    av_assert0(ref >= 0 && ref <4);
    InputFrame *inpf = &mcfps->input[ref];
    int linesize;
    uint8_t *data, *p0, *p1, *p2, *p3;
    int mvxfull, mvyfull, mvxsub, mvysub;

    linesize = inpf->halfpel_linesize[plane];

    mvxfull = x + (mvx >> 4);
    mvyfull = y + (mvy >> 4);
    mvxsub  = mvx & 7;
    mvysub  = mvy & 7;

    p0 = inpf->halfpel[plane][0] + mvxfull + mvyfull*linesize;
    p1 = inpf->halfpel[plane][1] + mvxfull + mvyfull*linesize;
    p2 = inpf->halfpel[plane][2] + mvxfull + mvyfull*linesize;
    p3 = inpf->halfpel[plane][3] + mvxfull + mvyfull*linesize;

    if (mvx & 8) {
        p0 += 1;
        p2 += 1;
        mvxsub = 8-mvxsub;
    }
    if (mvy & 8) {
        p0 += linesize;
        p1 += linesize;
        mvysub = 8-mvysub;
    }

    return ( (8-mvysub)*((8-mvxsub)*p0[0] + (mvxsub)*p1[0])
            +  (mvysub)*((8-mvxsub)*p2[0] + (mvxsub)*p3[0]) + 32) >> 6;
}

static void interpolate_pixels(MCFPSContext *mcfps, AVFrame *out)
{
    int x, y, plane;

    for (plane=0; plane<mcfps->planes; plane++){
        int w = out->width;
        int h = out->height;

        if (plane == 1 || plane == 2) {
            w = FF_CEIL_RSHIFT(w, mcfps->chroma_h_shift);
            h = FF_CEIL_RSHIFT(h, mcfps->chroma_v_shift);
        }

        for (y=0; y<h; y++) {
            for (x=0; x<w; x++) {
                int i;
                int weight_sum = 0;
                int v = 0;
                Pixel *pixel;
                if (plane == 1 || plane == 2) //FIXME optimize
                    pixel = &mcfps->pixel[(x<<mcfps->chroma_h_shift) + (y<<mcfps->chroma_v_shift)*out->width];
                else
                    pixel = &mcfps->pixel[x + y*out->width];

                for(i=0; i<pixel->nb; i++) {
                    weight_sum += pixel->weight[i];
                }
if(!weight_sum)
    weight_sum = 1;
                for(i=0; i<pixel->nb; i++) {
                    int t = mc_sample(mcfps, pixel, plane, x, y, i);
                    v += t *  pixel->weight[i];
                }
                out->data[plane][ x + y*out->linesize[plane] ] =
                    ROUNDED_DIV(v, weight_sum);
            }
        }
    }
}

static void interpolate(AVFilterContext *ctx, AVFrame *out)
{
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    MCFPSContext *mcfps = ctx->priv;
    int64_t pts;
    AVFrame *frame;
    int plane, alpha;
    pts = av_rescale(out->pts,
                        outlink->time_base.num * (int64_t)ALPHA_MAX * inlink->time_base.den,
                        outlink->time_base.den * (int64_t)    inlink->time_base.num
                    );
    alpha = (pts - mcfps->input[1].f->pts*ALPHA_MAX)/ (mcfps->input[2].f->pts - mcfps->input[1].f->pts);
    alpha = av_clip(alpha, 0, ALPHA_MAX);

    switch(mcfps->mode) {
    case MCFPS_MODE_NN:
        pts = av_rescale_q(out->pts, outlink->time_base, inlink->time_base);
        if (FFABS(pts - mcfps->input[1].f->pts) < FFABS(pts - mcfps->input[2].f->pts)) {
            frame = mcfps->input[1].f;
        } else
            frame = mcfps->input[2].f;

        av_frame_copy(out, frame);

        break;
    case MCFPS_MODE_LINEAR_IPOL:
        for (plane=0; plane < mcfps->planes; plane++) {
            int x, y;
            int w = out->width;
            int h = out->height;

            if (plane == 1 || plane == 2) {
                w = FF_CEIL_RSHIFT(w, mcfps->chroma_h_shift);
                h = FF_CEIL_RSHIFT(h, mcfps->chroma_v_shift);
            }

            for (y=0; y<h; y++) {
                for (x=0; x<w; x++) {
                    out->data[plane][ x + y*out->linesize[plane] ] =
                        ((ALPHA_MAX - alpha)*mcfps->input[1].f->data[plane][ x + y*mcfps->input[1].f->linesize[plane] ] +
                         alpha              *mcfps->input[2].f->data[plane][ x + y*mcfps->input[2].f->linesize[plane] ] + 512) >> 10;
                }
            }
        }

        break;
    case MCFPS_MODE_OBMC:
        fill_pixels(mcfps, alpha);
        interpolate_pixels(mcfps, out);
        break;
    }

    //FIXME
}

static int fill_halfpel(MCFPSContext *mcfps, InputFrame *inpf)
{
    int x, y, p;
    int j;

    for (p=0; p<mcfps->planes; p++) {
        int w = inpf->f->width;
        int h = inpf->f->height;
        int linesize = inpf->f->linesize[p];
        uint8_t *data = inpf->f->data[p];
        int hlinesize;

        if (p == 1 || p == 2) {
            w = FF_CEIL_RSHIFT(w, mcfps->chroma_h_shift);
            h = FF_CEIL_RSHIFT(h, mcfps->chroma_v_shift);
        }

        if (!inpf->halfpel_linesize[p])
            inpf->halfpel_linesize[p] = FFALIGN(w+1, 16);
        hlinesize = inpf->halfpel_linesize[p];

        for (j=0; j<4; j++)
            if (!inpf->halfpel[p][j]) {
                if (!(inpf->halfpel[p][j] = av_malloc(hlinesize * (h+1)))) {
                    return AVERROR(ENOMEM);
                }
            }
// 1 -5 20
        for (x=0; x<w+1; x++) {
            int x1 = x < w ? x : (w-1);
            int a = data[x1 + 1*linesize];
            int b = data[x1 + 0*linesize];
            int c = data[x1 + 0*linesize];
            int d = data[x1 + 1*linesize];
            int e = data[x1 + 2*linesize];
            int f;
            for (y=0; y<h+1; y++) {
                int y1 = y + 3;
                if (y1 >= h)
                    y1 = 2*h - y1 - 1;
                f = data[y1*linesize + x1];
                a = (20*(c+d) - 5*(b+e) + (a+f) + 16) >> 5;
                if (a & ~255)
                    a = ~(a>>31);
                inpf->halfpel[p][2][x + y*hlinesize] = a;
                a=b; b=c; c=d; d=e; e=f;
            }
        }

        for (y=0; y<h+1; y++) {
            int y1 = y < h ? y : (h-1);
            int a = data[y1*linesize + 1];
            int b = data[y1*linesize + 0];
            int c = data[y1*linesize + 0];
            int d = data[y1*linesize + 1];
            int e = data[y1*linesize + 2];
            int f;
            memcpy(inpf->halfpel[p][0] + y*hlinesize, data + y1*linesize, w);
            for (x=0; x<w+1; x++) {
                int x1 = x + 3;
                if (x1 >= w)
                    x1 = 2*w - x1 - 1;
                f = data[y1*linesize + x1];
                a = (20*(c+d) - 5*(b+e) + (a+f) + 16) >> 5;
                if (a & ~255)
                    a = ~(a>>31);
                inpf->halfpel[p][1][x + y*hlinesize] = a;
                a=b; b=c; c=d; d=e; e=f;
            }
            a = inpf->halfpel[p][2][y*hlinesize + 1];
            b = inpf->halfpel[p][2][y*hlinesize + 0];
            c = inpf->halfpel[p][2][y*hlinesize + 0];
            d = inpf->halfpel[p][2][y*hlinesize + 1];
            e = inpf->halfpel[p][2][y*hlinesize + 2];
            for (x=0; x<w+1; x++) {
                int x1 = x + 3;
                if (x1 >= w)
                    x1 = 2*w - x1 - 1;
                f = inpf->halfpel[p][2][y*hlinesize + x1];
                a = (20*(c+d) - 5*(b+e) + (a+f) + 16) >> 5;
                if (a & ~255)
                    a = ~(a>>31);
                inpf->halfpel[p][3][x + y*hlinesize] = a;
                a=b; b=c; c=d; d=e; e=f;
            }
        }
    }

    return 0;
}

static int extract_mvs(MCFPSContext *mcfps, InputFrame *f, int dir)
{
    if (!f->mv[dir])
        f->mv[dir] = av_malloc(mcfps->b_width * mcfps->b_height * sizeof(*f->mv[0]));
    if (!f->ref[dir])
        f->ref[dir] = av_malloc(mcfps->b_width * mcfps->b_height * sizeof(*f->ref[0]));
    if (!f->mv[dir] || !f->ref[0])
        return AVERROR(ENOMEM);

    return avpriv_get_mvs(mcfps->avctx_enc[dir],
                        f->mv[dir],
                        f->ref[dir],
                        mcfps->b_width,
                        mcfps->b_height);
}

static int inject_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    MCFPSContext *mcfps = ctx->priv;
    int ret;
    InputFrame tmp, *f;
    AVPacket pkt = {0};
    int got_pkt_ptr;

    av_frame_free(&mcfps->input[0].f);
    tmp = mcfps->input[0];
    memmove(&mcfps->input[0], &mcfps->input[1], sizeof(mcfps->input[0]) * (NB_INPUT_FRAMES-1));
    mcfps->input[NB_INPUT_FRAMES-1] = tmp;
    mcfps->input[NB_INPUT_FRAMES-1].f = frame;

    if (mcfps->mode > MCFPS_MODE_LINEAR_IPOL) {
        ret = fill_halfpel(mcfps, &mcfps->input[NB_INPUT_FRAMES-1]);
        if (ret < 0)
            return ret;

        frame->quality = 2 * FF_QP2LAMBDA; //FIXME test/adjust
    //    init per MB qscale stuff FIXME

        av_init_packet(&pkt);
        pkt.data = mcfps->outbuf;
        pkt.size = mcfps->outbuf_size;

        avcodec_encode_video2(mcfps->avctx_enc[0], &pkt, frame, &got_pkt_ptr);
        av_free_packet(&pkt);

        if (mcfps->input[NB_INPUT_FRAMES-2].f) {
            avcodec_encode_video2(mcfps->avctx_enc[1], &pkt, frame, &got_pkt_ptr);
            av_assert0(pkt.flags & AV_PKT_FLAG_KEY);
            av_free_packet(&pkt);
            avcodec_encode_video2(mcfps->avctx_enc[1], &pkt, mcfps->input[NB_INPUT_FRAMES-2].f, &got_pkt_ptr);
            av_assert0(!(pkt.flags & AV_PKT_FLAG_KEY));
            av_free_packet(&pkt);

            ret = extract_mvs(mcfps, &mcfps->input[NB_INPUT_FRAMES-2], 1);
            av_assert0(ret >= 0);
        }

        ret = extract_mvs(mcfps, &mcfps->input[NB_INPUT_FRAMES-1], 0);
        av_assert0(ret >= 0);
    }

//FIXME    set mcfps->input[NB_INPUT_FRAMES-1] motion vectors and mb types

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    MCFPSContext *mcfps = ctx->priv;
    int ret;

    av_assert0(frame);
    mcfps->planes = av_pix_fmt_count_planes(frame->format);
    avcodec_get_chroma_sub_sample(frame->format, &mcfps->chroma_h_shift, &mcfps->chroma_v_shift);

av_assert0(frame->pts != AV_NOPTS_VALUE); //FIXME

    if (!mcfps->input[NB_INPUT_FRAMES-1].f ||
        frame->pts < mcfps->input[NB_INPUT_FRAMES-1].f->pts) {
        av_log(ctx, AV_LOG_VERBOSE, "Initializing outpts from input pts %"PRId64"\n",
               frame->pts);
        mcfps->out_pts = av_rescale_q(frame->pts, inlink->time_base, outlink->time_base);
    }

    if (!mcfps->input[NB_INPUT_FRAMES-1].f)
        inject_frame(inlink, av_frame_clone(frame));
    inject_frame(inlink, frame);

    if (!mcfps->input[0].f)
        return 0;

    for (;;) {
        AVFrame *out;

        if (av_compare_ts(mcfps->input[NB_INPUT_FRAMES/2].f->pts, inlink->time_base,
                          mcfps->out_pts, outlink->time_base) < 0)
            break;

        out = ff_get_video_buffer(ctx->outputs[0], inlink->w, inlink->h);
        if (!out)
            return AVERROR(ENOMEM);
        av_frame_copy_props(out, mcfps->input[NB_INPUT_FRAMES/2].f);
        out->pts = mcfps->out_pts;
        mcfps->out_pts ++;

        interpolate(ctx, out);

        ret = ff_filter_frame(ctx->outputs[0], out);
        if (ret < 0)
            return ret;
    }
    return 0;
}

#define OFFSET(x) offsetof(MCFPSContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

#define CONST(name, help, val, unit) { name, help, 0, AV_OPT_TYPE_CONST, {.i64=val}, INT_MIN, INT_MAX, FLAGS, unit }

static const AVOption mcfps_options[] = {
    { "mode",   "specify the interpolation mode", OFFSET(mode), AV_OPT_TYPE_INT, {.i64 = MCFPS_MODE_NN}, 0, 3, FLAGS, "mode"},
    CONST("nn",         "", MCFPS_MODE_NN,              "mode"),
    CONST("linear",     "", MCFPS_MODE_LINEAR_IPOL,     "mode"),
    CONST("gmc",        "", MCFPS_MODE_GMC,             "mode"),
    CONST("obmc",       "", MCFPS_MODE_OBMC,            "mode"),

    { "fps",   "specify the frame rate", OFFSET(frame_rate), AV_OPT_TYPE_RATIONAL, {.dbl = 25}, 0, INT_MAX, FLAGS},


    { NULL }
};

AVFILTER_DEFINE_CLASS(mcfps);

static const AVFilterPad avfilter_vf_mcfps_inputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .filter_frame  = filter_frame,
    },
    { NULL }
};

static const AVFilterPad avfilter_vf_mcfps_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
    { NULL }
};

AVFilter ff_vf_mcfps = {
    .name          = "mcfps",
    .description   = NULL_IF_CONFIG_SMALL("Frame rate changing with motion compensated interpolation"),
    .priv_size     = sizeof(MCFPSContext),
    .priv_class    = &mcfps_class,
    .uninit        = uninit,
    .query_formats = query_formats,
    .inputs        = avfilter_vf_mcfps_inputs,
    .outputs       = avfilter_vf_mcfps_outputs,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_INTERNAL,
};