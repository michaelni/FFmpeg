/*
 * Copyright (c) 2002 A'rpi
 * Copyright (C) 2012 Clément Bœsch
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

/**
 * @file
 * libpostproc filter, ported from MPlayer.
 */

#include "libavutil/avassert.h"
#include "libavutil/opt.h"
#include "internal.h"

#include "libpostproc/postprocess.h"

typedef struct {
    int mode_id;
    pp_mode *modes[PP_QUALITY_MAX + 1];
    void *pp_ctx;
} PPFilterContext;

static av_cold int pp_init(AVFilterContext *ctx, const char *args)
{
    int i;
    PPFilterContext *pp = ctx->priv;

    if (!args || !*args)
        args = "de";

    for (i = 0; i <= PP_QUALITY_MAX; i++) {
        pp->modes[i] = pp_get_mode_by_name_and_quality(args, i);
        if (!pp->modes[i])
            return AVERROR_EXTERNAL;
    }
    pp->mode_id = PP_QUALITY_MAX;
    return 0;
}

static int pp_process_command(AVFilterContext *ctx, const char *cmd, const char *args,
                              char *res, int res_len, int flags)
{
    PPFilterContext *pp = ctx->priv;

    if (!strcmp(cmd, "quality")) {
        pp->mode_id = av_clip(strtol(args, NULL, 10), 0, PP_QUALITY_MAX);
        return 0;
    }
    return AVERROR(ENOSYS);
}

static int pp_query_formats(AVFilterContext *ctx)
{
    static const enum PixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUVJ420P,
        AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUVJ422P,
        AV_PIX_FMT_YUV411P,
        AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUVJ444P,
        AV_PIX_FMT_NONE
    };
    ff_set_common_formats(ctx, ff_make_format_list(pix_fmts));
    return 0;
}

static int pp_config_props(AVFilterLink *inlink)
{
    int flags = PP_CPU_CAPS_AUTO;
    PPFilterContext *pp = inlink->dst->priv;

    switch (inlink->format) {
    case AV_PIX_FMT_YUVJ420P:
    case AV_PIX_FMT_YUV420P: flags |= PP_FORMAT_420; break;
    case AV_PIX_FMT_YUVJ422P:
    case AV_PIX_FMT_YUV422P: flags |= PP_FORMAT_422; break;
    case AV_PIX_FMT_YUV411P: flags |= PP_FORMAT_411; break;
    case AV_PIX_FMT_YUVJ444P:
    case AV_PIX_FMT_YUV444P: flags |= PP_FORMAT_444; break;
    default: av_assert0(0);
    }

    pp->pp_ctx = pp_get_context(inlink->w, inlink->h, flags);
    if (!pp->pp_ctx)
        return AVERROR(ENOMEM);
    return 0;
}

static int pp_filter_frame(AVFilterLink *inlink, AVFrame *inbuf)
{
    AVFilterContext *ctx = inlink->dst;
    PPFilterContext *pp = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    const int aligned_w = FFALIGN(outlink->w, 8);
    const int aligned_h = FFALIGN(outlink->h, 8);
    AVFrame *outbuf;

    outbuf = ff_get_video_buffer(outlink, aligned_w, aligned_h);
    if (!outbuf) {
        av_frame_free(&inbuf);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(outbuf, inbuf);

    pp_postprocess((const uint8_t **)inbuf->data, inbuf->linesize,
                   outbuf->data,                 outbuf->linesize,
                   aligned_w, outlink->h,
                   outbuf->qscale_table,
                   outbuf->qstride,
                   pp->modes[pp->mode_id],
                   pp->pp_ctx,
                   outbuf->pict_type);

    av_frame_free(&inbuf);
    return ff_filter_frame(outlink, outbuf);
}

static av_cold void pp_uninit(AVFilterContext *ctx)
{
    int i;
    PPFilterContext *pp = ctx->priv;

    for (i = 0; i <= PP_QUALITY_MAX; i++)
        pp_free_mode(pp->modes[i]);
    if (pp->pp_ctx)
        pp_free_context(pp->pp_ctx);
}

static const AVFilterPad pp_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = pp_config_props,
        .filter_frame = pp_filter_frame,
    },
    { NULL }
};

static const AVFilterPad pp_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter avfilter_vf_pp = {
    .name            = "pp",
    .description     = NULL_IF_CONFIG_SMALL("Filter video using libpostproc."),
    .priv_size       = sizeof(PPFilterContext),
    .init            = pp_init,
    .uninit          = pp_uninit,
    .query_formats   = pp_query_formats,
    .inputs          = pp_inputs,
    .outputs         = pp_outputs,
    .process_command = pp_process_command,
};
