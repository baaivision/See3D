# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import cv2
import numpy as np
from PIL import Image
from os.path import *
import re
def microsoft_sorted(input_list):
    return sorted(input_list, key=lambda s: [int(s) if s.isdigit() else s for s in sum(re.findall(r'(\D+)(\d+)', 'a'+s+'0'), ())])
    
UNKNOWN_FLOW_THRESH = 1e7


def flow_to_image(flow, global_max=None):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.0
    maxv = -999.0
    minu = 999.0
    minv = 999.0

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)

    if global_max == None:
        maxrad = max(-1, np.max(rad))
    else:
        maxrad = global_max

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col : col + YG, 0] = 255 - np.transpose(
        np.floor(255 * np.arange(0, YG) / YG)
    )
    colorwheel[col : col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col : col + CB, 1] = 255 - np.transpose(
        np.floor(255 * np.arange(0, CB) / CB)
    )
    colorwheel[col : col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += +BM

    # MR
    colorwheel[col : col + MR, 2] = 255 - np.transpose(
        np.floor(255 * np.arange(0, MR) / MR)
    )
    colorwheel[col : col + MR, 0] = 255

    return colorwheel


def resize_flow(flow, H_new, W_new):
    H_old, W_old = flow.shape[0:2]
    flow_resized = cv2.resize(flow, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
    flow_resized[:, :, 0] *= H_new / H_old
    flow_resized[:, :, 1] *= W_new / W_old
    return flow_resized


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )
    return res


def consistCheck(flowB, flowF):
    # |--------------------|  |--------------------|
    # |       y            |  |       v            |
    # |   x   *            |  |   u   *            |
    # |                    |  |                    |
    # |--------------------|  |--------------------|

    # sub: numPix * [y x t]

    imgH, imgW, _ = flowF.shape

    (fy, fx) = np.mgrid[0:imgH, 0:imgW].astype(np.float32)
    fxx = fx + flowB[:, :, 0]  # horizontal
    fyy = fy + flowB[:, :, 1]  # vertical

    u = fxx + cv2.remap(flowF[:, :, 0], fxx, fyy, cv2.INTER_LINEAR) - fx
    v = fyy + cv2.remap(flowF[:, :, 1], fxx, fyy, cv2.INTER_LINEAR) - fy
    BFdiff = (u**2 + v**2) ** 0.5

    return BFdiff, np.stack((u, v), axis=2)


def read_optical_flow(basedir, img_i_name, read_fwd):
    flow_dir = os.path.join(basedir, "flow")

    fwd_flow_path = os.path.join(flow_dir, "%s_fwd.npz" % img_i_name[:-4])
    bwd_flow_path = os.path.join(flow_dir, "%s_bwd.npz" % img_i_name[:-4])

    if read_fwd:
        fwd_data = np.load(fwd_flow_path)
        fwd_flow, fwd_mask = fwd_data["flow"], fwd_data["mask"]
        return fwd_flow, fwd_mask
    else:
        bwd_data = np.load(bwd_flow_path)
        bwd_flow, bwd_mask = bwd_data["flow"], bwd_data["mask"]
        return bwd_flow, bwd_mask


def compute_epipolar_distance(T_21, K, p_1, p_2):
    R_21 = T_21[:3, :3]
    t_21 = T_21[:3, 3]

    E_mat = np.dot(skew(t_21), R_21)
    # compute bearing vector
    inv_K = np.linalg.inv(K)

    F_mat = np.dot(np.dot(inv_K.T, E_mat), inv_K)

    l_2 = np.dot(F_mat, p_1)
    algebric_e_distance = np.sum(p_2 * l_2, axis=0)
    n_term = np.sqrt(l_2[0, :] ** 2 + l_2[1, :] ** 2) + 1e-8
    geometric_e_distance = algebric_e_distance / n_term
    geometric_e_distance = np.abs(geometric_e_distance)

    return geometric_e_distance


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
