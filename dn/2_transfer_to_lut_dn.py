import os
import sys

import numpy as np
from PIL import Image
np.set_printoptions(threshold=np.inf)
import torch
import torch.nn.functional as F
import time

sys.path.insert(0, "../")  # run under the current directory
from common.option_dnlut_sidd import TestOptions
import model_dnlut as model

def FourSimplexInterpFaster(weight, img_in, h, w, interval, rot, upscale=4, mode='s'):
    q = 2 ** interval
    L = 2 ** (8 - interval) + 1
    s1 = time.time()
    if mode == "s":
        # Extract MSBs
        img_a1 = img_in[:, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, 0:0 + h, 1:1 + w] // q
        img_c1 = img_in[:, 1:1 + h, 0:0 + w] // q
        img_d1 = img_in[:, 1:1 + h, 1:1 + w] // q

        # Extract LSBs
        fa = img_in[:, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, 0:0 + h, 1:1 + w] % q
        fc = img_in[:, 1:1 + h, 0:0 + w] % q
        fd = img_in[:, 1:1 + h, 1:1 + w] % q

    elif mode == 'd':
        img_a1 = img_in[:, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, 0:0 + h, 2:2 + w] // q
        img_c1 = img_in[:, 2:2 + h, 0:0 + w] // q
        img_d1 = img_in[:, 2:2 + h, 2:2 + w] // q

        fa = img_in[:, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, 0:0 + h, 2:2 + w] % q
        fc = img_in[:, 2:2 + h, 0:0 + w] % q
        fd = img_in[:, 2:2 + h, 2:2 + w] % q

    elif mode == 'y':
        img_a1 = img_in[:, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, 1:1 + h, 1:1 + w] // q
        img_c1 = img_in[:, 1:1 + h, 2:2 + w] // q
        img_d1 = img_in[:, 2:2 + h, 1:1 + w] // q

        fa = img_in[:, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, 1:1 + h, 1:1 + w] % q
        fc = img_in[:, 1:1 + h, 2:2 + w] % q
        fd = img_in[:, 2:2 + h, 1:1 + w] % q

    elif 'RG' in mode:
        img_a1 = img_in[1:2, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[1:2, 0:0 + h, 1:1 + w] // q
        img_c1 = img_in[0:1, 0:0 + h, 1:1 + w] // q
        img_d1 = img_in[0:1, 0:0 + h, 0:0 + w] // q

        fa = img_in[1:2, 0:0 + h, 0:0 + w] % q
        fb = img_in[1:2, 0:0 + h, 1:1 + w] % q
        fc = img_in[0:1, 0:0 + h, 1:1 + w] % q
        fd = img_in[0:1, 0:0 + h, 0:0 + w] % q

    elif 'GB' in mode:
        img_a1 = img_in[2:, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[2:, 0:0 + h, 1:1 + w] // q
        img_c1 = img_in[1:2, 0:0 + h, 1:1 + w] // q
        img_d1 = img_in[1:2, 0:0 + h, 0:0 + w] // q

        fa = img_in[2:, 0:0 + h, 0:0 + w] % q
        fb = img_in[2:, 0:0 + h, 1:1 + w] % q
        fc = img_in[1:2, 0:0 + h, 1:1 + w] % q
        fd = img_in[1:2, 0:0 + h, 0:0 + w] % q
        # print(img_a1.shape, img_b1.shape, img_c1.shape, img_d1.shape, img_in.shape)
    elif 'BR' in mode:
        img_a1 = img_in[0:1, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[0:1, 0:0 + h, 1:1 + w] // q
        img_c1 = img_in[2:, 0:0 + h, 1:1 + w] // q
        img_d1 = img_in[2:, 0:0 + h, 0:0 + w] // q

        fa = img_in[0:1, 0:0 + h, 0:0 + w] % q
        fb = img_in[0:1, 0:0 + h, 1:1 + w] % q
        fc = img_in[2:, 0:0 + h, 1:1 + w] % q
        fd = img_in[2:, 0:0 + h, 0:0 + w] % q

    else:
        # more sampling modes can be implemented similarly
        raise ValueError("Mode {} not implemented.".format(mode))

    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    img_d2 = img_d1 + 1
    
    # print(img_a1[0], img_b1[0], img_c1[0], img_d1[0])
    p0000 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0001 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0010 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0011 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0100 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0101 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0110 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0111 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p1000 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1001 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1010 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1011 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1100 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1101 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1110 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1111 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))


    # print(p0000[0], p0001[0], p0010[0], p0100[0], p1000[0], p0011[0], p0101[0], p0110[0], p1001[0], p1010[0], p1100[0], p1110[0], p1011[0], p1101[0], p1111[0])
    # Output image holder
    out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2]
    out = out.reshape(sz, -1)

    p0000 = p0000.reshape(sz, -1)
    
    p0100 = p0100.reshape(sz, -1)
    p1000 = p1000.reshape(sz, -1)
    p1100 = p1100.reshape(sz, -1)
    fa = fa.reshape(-1, 1)

    p0001 = p0001.reshape(sz, -1)
    p0101 = p0101.reshape(sz, -1)
    p1001 = p1001.reshape(sz, -1)
    p1101 = p1101.reshape(sz, -1)
    fb = fb.reshape(-1, 1)
    fc = fc.reshape(-1, 1)

    p0010 = p0010.reshape(sz, -1)
    p0110 = p0110.reshape(sz, -1)
    p1010 = p1010.reshape(sz, -1)
    p1110 = p1110.reshape(sz, -1)
    fd = fd.reshape(-1, 1)

    p0011 = p0011.reshape(sz, -1)
    p0111 = p0111.reshape(sz, -1)
    p1011 = p1011.reshape(sz, -1)
    p1111 = p1111.reshape(sz, -1)

    fab = fa > fb;
    fac = fa > fc;
    fad = fa > fd

    fbc = fb > fc;
    fbd = fb > fd;
    fcd = fc > fd

    
    i1 = i = np.logical_and.reduce((fab, fbc, fcd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i2 = i = np.logical_and.reduce((~i1[:, None], fab, fbc, fbd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i3 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], fab, fbc, fad)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i4 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc)).squeeze(1)

    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]

    i5 = i = np.logical_and.reduce((~(fbc), fab, fac, fbd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i6 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], fab, fac, fcd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i7 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i8 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]

    i9 = i = np.logical_and.reduce((~(fbc), ~(fac), fab, fbd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
    # i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], fab, fcd)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    # i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], fab, fad)).squeeze(1)  # c > a > d > b
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd)).squeeze(1)  # c > d > a > b
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i12 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]

    i13 = i = np.logical_and.reduce((~(fab), fac, fcd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i14 = i = np.logical_and.reduce((~(fab), ~i13[:, None], fac, fad)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i15 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], fac, fbd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i16 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]

    i17 = i = np.logical_and.reduce((~(fab), ~(fac), fbc, fad)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i18 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], fbc, fcd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i19 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i20 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]

    i21 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), fad)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i22 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i23 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i24 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None])).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    print('four:', time.time() - s1)
    out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    # print('out shape 4: ', out.shape)
    out = np.transpose(out, (0, 1, 3, 2, 4)).reshape(
        (img_a1.shape[0], img_a1.shape[1] * upscale, img_a1.shape[2] * upscale))
    out = np.rot90(out, rot, [1, 2])
    out = out / q
    return out


def Tetrahedral(weight, img_in, h, w, interval, rot, upscale=4, mode='s'):
    q = 2 ** interval
    L = 2 ** (8 - interval) + 1
    
    if mode == "v":
        # Extract MSBs
        img_a1 = img_in[:, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, 0:0 + h, 1:1 + w] // q
        img_c1 = img_in[:, 1:1 + h, 1:1 + w] // q

        # Extract LSBs
        fa = img_in[:, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, 0:0 + h, 1:1 + w] % q
        fc = img_in[:, 1:1 + h, 1:1 + w] % q

    elif mode == "q":
        # Extract MSBs
        img_a1 = img_in[0:1, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[1:2, 0:0 + h, 0:0 + w] // q
        img_c1 = img_in[2:, 0:0 + h, 0:0 + w] // q

        # Extract LSBs
        fa = img_in[0:1, 0:0 + h, 0:0 + w] % q
        fb = img_in[1:2, 0:0 + h, 0:0 + w] % q
        fc = img_in[2:, 0:0 + h, 0:0 + w] % q

    

    else:
        # more sampling modes can be implemented similarly
        raise ValueError("Mode {} not implemented.".format(mode))

    s1 = time.time()
    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    
    p000 = weight[img_a1.flatten().astype(np.int_) * L * L+ img_b1.flatten().astype(
        np.int_) * L + img_c1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p001 = weight[img_a1.flatten().astype(np.int_) * L * L + img_b1.flatten().astype(
        np.int_) * L + img_c2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p010 = weight[img_a1.flatten().astype(np.int_) * L * L + img_b2.flatten().astype(
        np.int_) * L + img_c1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p100 = weight[img_a2.flatten().astype(np.int_) * L * L + img_b1.flatten().astype(
        np.int_) * L + img_c1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p011 = weight[img_a1.flatten().astype(np.int_) * L * L + img_b2.flatten().astype(
        np.int_) * L + img_c2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p101 = weight[img_a2.flatten().astype(np.int_) * L * L + img_b1.flatten().astype(
        np.int_) * L + img_c2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p110 = weight[img_a2.flatten().astype(np.int_) * L * L + img_b2.flatten().astype(
        np.int_) * L + img_c1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p111 = weight[img_a2.flatten().astype(np.int_) * L * L + img_b2.flatten().astype(
        np.int_) * L + img_c2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2]
    out = out.reshape(sz, -1)

    p000 = p000.reshape(sz, -1)
    p001 = p001.reshape(sz, -1)
    p010 = p010.reshape(sz, -1)
    p100 = p100.reshape(sz, -1)
    p101 = p101.reshape(sz, -1)
    p110 = p110.reshape(sz, -1)
    p011 = p011.reshape(sz, -1)
    p111 = p111.reshape(sz, -1)
    fa = fa.reshape(-1, 1)
    fb = fb.reshape(-1, 1)
    fc = fc.reshape(-1, 1)


    fab = fa > fb;
    fac = fa > fc;
    fbc = fb > fc
    # fca = fc > fa;
    # fba = fb > fa;
    # fcb = fc > fb


    
    i1 = i = np.logical_and.reduce((fab, fbc)).squeeze(1)
    out[i] = (q - fa[i]) * p000[i] + (fa[i] - fb[i]) * p100[i] + (fb[i] - fc[i]) * p110[i] + (fc[i]) * p111[i]

    i2 = i = np.logical_and.reduce((~i1[:, None], fab, fac)).squeeze(1)
    # i2 = i = np.logical_and.reduce((fac, fcb)).squeeze(1)
    out[i] = (q - fa[i]) * p000[i] + (fa[i] - fc[i]) * p100[i] + (fc[i] - fb[i]) * p101[i] + (fb[i]) * p111[i]

    i3 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], fac, fbc)).squeeze(1)
    # i3 = i = np.logical_and.reduce((fba, fac)).squeeze(1)
    out[i] = (q - fb[i]) * p000[i] + (fb[i] - fa[i]) * p010[i] + (fa[i] - fc[i]) * p110[i] + (fc[i]) * p111[i]

    i4 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], ~i3[:, None], fbc)).squeeze(1)
    # i4 = i = np.logical_and.reduce((fbc, fca)).squeeze(1)
    out[i] = (q - fb[i]) * p000[i] + (fb[i] - fc[i]) * p010[i] + (fc[i] - fa[i]) * p011[i] + (fa[i]) * p111[i]

    i5 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], fab)).squeeze(1)
    # i5 = i = np.logical_and.reduce((fca, fab)).squeeze(1)
    out[i] = (q - fc[i]) * p000[i] + (fc[i] - fa[i]) * p001[i] + (fa[i] - fb[i]) * p101[i] + (fb[i]) * p111[i]

    i6 = i = np.logical_and.reduce((~(fbc), ~(fab), ~(fac))).squeeze(1)
    # i6 = i = np.logical_and.reduce((fcb, fba)).squeeze(1)
    out[i] = (q - fc[i]) * p000[i] + (fc[i] - fb[i]) * p001[i] + (fb[i] - fa[i]) * p011[i] + (fa[i]) * p111[i]
    
    
    out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    out = np.transpose(out, (0, 1, 3, 2, 4)).reshape(
        (img_a1.shape[0], img_a1.shape[1] * upscale, img_a1.shape[2] * upscale))
    out = np.rot90(out, rot, [1, 2])
    out = out / q
    print("three:", time.time()-s1)
    return out

def get_input_tensor(opt):
    # 1D input
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)

    # 2D input
    # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)
    # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    second = base.cuda().repeat(L)
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

    # 3D input
    # 256*256*256   0 x65536|1 x65536|...|255 x65536
    third = base.cuda().unsqueeze(1).repeat(1, L * L).reshape(-1)
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat(
        [third.unsqueeze(1), onebytwo], 1)  # [256*256*256, 3]

    # 4D input
    fourth = base.cuda().unsqueeze(1).repeat(1, L * L * L).reshape(
        -1)  # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebythree = onebythree.repeat(L, 1)
    # [256*256*256*256, 4]
    onebyfourth = torch.cat([fourth.unsqueeze(1), onebythree], 1)

    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_tensor = onebyfourth.unsqueeze(1).unsqueeze(
        1).reshape(-1, 1, 2, 2).float() / 255.0
    return input_tensor


def get_input_tensor_3d(opt):
    # 1D input
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)

    # 2D input
    # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)
    # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    second = base.cuda().repeat(L)
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

    # 3D input
    # 256*256*256   0 x65536|1 x65536|...|255 x65536
    third = base.cuda().unsqueeze(1).repeat(1, L * L).reshape(-1)
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat(
        [third.unsqueeze(1), onebytwo], 1)  # [256*256*256, 3]

    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_tensor = onebythree.unsqueeze(1).unsqueeze(
        1).reshape(-1, 1, 1, 3).float() / 255.0
    return input_tensor


def get_mode_input_tensor(input_tensor, mode):
    if mode == "d":
        input_tensor_dil = torch.zeros(
            (input_tensor.shape[0], input_tensor.shape[1], 3, 3), dtype=input_tensor.dtype).to(input_tensor.device)
        input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0, 0]
        input_tensor_dil[:, :, 0, 2] = input_tensor[:, :, 0, 1]
        input_tensor_dil[:, :, 2, 0] = input_tensor[:, :, 1, 0]
        input_tensor_dil[:, :, 2, 2] = input_tensor[:, :, 1, 1]
        input_tensor = input_tensor_dil
    elif mode == "y":
        input_tensor_dil = torch.zeros(
            (input_tensor.shape[0], input_tensor.shape[1], 3, 3), dtype=input_tensor.dtype).to(input_tensor.device)
        input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0, 0]
        input_tensor_dil[:, :, 1, 1] = input_tensor[:, :, 0, 1]
        input_tensor_dil[:, :, 1, 2] = input_tensor[:, :, 1, 0]
        input_tensor_dil[:, :, 2, 1] = input_tensor[:, :, 1, 1]
        input_tensor = input_tensor_dil

    elif mode == 'RG':
        input_tensor_dil = torch.zeros(
            (input_tensor.shape[0], 3, 1, 2), dtype=input_tensor.dtype).to(input_tensor.device)
        input_tensor_dil[:, 0, 0, 0] = input_tensor[:, 0, 1, 1]
        input_tensor_dil[:, 0, 0, 1] = input_tensor[:, 0, 1, 0]
        input_tensor_dil[:, 1, 0, 0] = input_tensor[:, 0, 0, 0]
        input_tensor_dil[:, 1, 0, 1] = input_tensor[:, 0, 0, 1]

        # input_tensor_dil = torch.zeros(
        #     (input_tensor.shape[0], 3, 1, 2), dtype=input_tensor.dtype).to(input_tensor.device)
        input_tensor = input_tensor_dil

    elif mode == 'GB':
        input_tensor_dil = torch.zeros(
            (input_tensor.shape[0], 3, 1, 2), dtype=input_tensor.dtype).to(input_tensor.device)
        input_tensor_dil[:, 1, 0, 0] = input_tensor[:, 0, 1, 1]
        input_tensor_dil[:, 1, 0, 1] = input_tensor[:, 0, 1, 0]
        input_tensor_dil[:, 2, 0, 0] = input_tensor[:, 0, 0, 0]
        input_tensor_dil[:, 2, 0, 1] = input_tensor[:, 0, 0, 1]
        input_tensor = input_tensor_dil
    elif mode == 'BR':
        input_tensor_dil = torch.zeros(
            (input_tensor.shape[0], 3, 1, 2), dtype=input_tensor.dtype).to(input_tensor.device)
        
        input_tensor_dil[:, 2, 0, 0] = input_tensor[:, 0, 1, 1]
        input_tensor_dil[:, 2, 0, 1] = input_tensor[:, 0, 1, 0]
        input_tensor_dil[:, 0, 0, 0] = input_tensor[:, 0, 0, 0]
        input_tensor_dil[:, 0, 0, 1] = input_tensor[:, 0, 0, 1]
        input_tensor = input_tensor_dil
    


    else:
        # more sampling modes can be implemented similarly
        raise ValueError("Mode {} not implemented.".format(mode))
    return input_tensor


def get_mode_input_tensor_3d(input_tensor, mode):
    # print(mode)
    if mode == "v":
        input_tensor_dil = torch.zeros(
            (input_tensor.shape[0], input_tensor.shape[1], 2, 2), dtype=input_tensor.dtype).to(input_tensor.device)
        input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0, 0]
        input_tensor_dil[:, :, 0, 1] = input_tensor[:, :, 0, 1]
        input_tensor_dil[:, :, 1, 1] = input_tensor[:, :, 0, 2]
        input_tensor = input_tensor_dil

    elif mode == "q":
        input_tensor_dil = torch.zeros(
            (input_tensor.shape[0], 3, 1, 1), dtype=input_tensor.dtype).to(input_tensor.device)
        input_tensor_dil[:, 0, 0, 0] = input_tensor[:, 0, 0, 0]
        input_tensor_dil[:, 1, 0, 0] = input_tensor[:, 0, 0, 1]
        input_tensor_dil[:, 2, 0, 0] = input_tensor[:, 0, 0, 2]
        input_tensor = input_tensor_dil
    


    else:
        # more sampling modes can be implemented similarly
        raise ValueError("Mode {} not implemented.".format(mode))
    return input_tensor


def build_m(model_G, input_tensor_4d, opt):
    B = input_tensor_4d.size(0) // 100
    for i in range(2):
        ## mixer RG
        input_tensor = get_mode_input_tensor(input_tensor_4d, 'RG')
        
        outputs = []

        with torch.no_grad():
            model_G.eval()
            for b in range(100):
                if b == 99:
                    batch_input = input_tensor[b * B:]
                else:
                    batch_input = input_tensor[b * B:(b + 1) * B]

                _, batch_output, _, _ = model_G(batch_input, stage=i+1, mode='m')
                
                results = torch.round(torch.clamp(batch_output, -1, 1)
                                        * 127).cpu().data.numpy().astype(np.int8)
                outputs += [results]

        results = np.concatenate(outputs, 0)

        lut_path = os.path.join(opt.expDir,
                                "dnlut_LUT_{}bit_int8_s{}_{}.npy".format(opt.interval, str(i+1), 'RG'))
        np.save(lut_path, results[:, 0, :, :])

        print("Resulting LUT size: ", results.shape, "Saved to", lut_path)

        # mixer GB
        input_tensor = get_mode_input_tensor(input_tensor_4d, 'GB')
        outputs = []

        with torch.no_grad():
            model_G.eval()
            for b in range(100):
                if b == 99:
                    batch_input = input_tensor[b * B:]
                else:
                    batch_input = input_tensor[b * B:(b + 1) * B]

                _, _, batch_output, _ = model_G(batch_input, stage=i+1, mode='m')

                results = torch.round(torch.clamp(batch_output, -1, 1)
                                        * 127).cpu().data.numpy().astype(np.int8)
                outputs += [results]
                

        results = np.concatenate(outputs, 0)

        lut_path = os.path.join(opt.expDir,
                                "dnlut_LUT_{}bit_int8_s{}_{}.npy".format(opt.interval, str(i+1), 'GB'))
        # print(results)
        np.save(lut_path, results)

        print("Resulting LUT size: ", results.shape, "Saved to", lut_path)

        ## mixer GB
        input_tensor = get_mode_input_tensor(input_tensor_4d, 'BR')
        
        outputs = []

        with torch.no_grad():
            model_G.eval()
            for b in range(100):
                if b == 99:
                    batch_input = input_tensor[b * B:]
                else:
                    batch_input = input_tensor[b * B:(b + 1) * B]

                _, _, _, batch_output = model_G(batch_input, stage=i+1, mode='m')

                results = torch.round(torch.clamp(batch_output, -1, 1)
                                        * 127).cpu().data.numpy().astype(np.int8)
                outputs += [results]
                

        results = np.concatenate(outputs, 0)

        lut_path = os.path.join(opt.expDir,
                                "dnlut_LUT_{}bit_int8_s{}_{}.npy".format(opt.interval, str(i+1), 'BR'))
        # print(results)
        np.save(lut_path, results)

        print("Resulting LUT size: ", results.shape, "Saved to", lut_path)

def build_v(model_G, input_tensor_3d, opt):
    input_tensor = get_mode_input_tensor_3d(input_tensor_3d, 'v')
    
    for s in range(stages):
        outputs = []
        with torch.no_grad():
            
            model_G.eval()
            batch_input = input_tensor
            # print(batch_input[:100])
            batch_output = model_G(batch_input, stage=s+1, mode='v')
            
            results = torch.round(torch.clamp(batch_output, -1, 1)
                                    * 127).cpu().data.numpy().astype(np.int8)
            outputs += [results]
        results = np.concatenate(outputs, 0)

        lut_path = os.path.join(opt.expDir,
                                "dnlut_LUT_{}bit_int8_s{}_{}.npy".format(opt.interval, str(s+1), 'V'))
        # print(results)
        np.save(lut_path, results)

        print("Resulting LUT size: ", results.shape, "Saved to", lut_path)

def build_q(model_G, input_tensor_3d, opt):
    if build_q:
        input_tensor = get_mode_input_tensor_3d(input_tensor_3d, 'q')
        
        for s in range(3):
            outputs = []
            with torch.no_grad():
                
                model_G.eval()
                batch_input = input_tensor
                # print(batch_input[:100])
                batch_output = model_G(batch_input, stage=s+1, mode='q')
                
                results = torch.round(torch.clamp(batch_output, -1, 1)
                                        * 127).cpu().data.numpy().astype(np.int8)
                outputs += [results]
            results = np.concatenate(outputs, 0)

            lut_path = os.path.join(opt.expDir,
                                    "dnlut_LUT_{}bit_int8_s{}_{}.npy".format(opt.interval, str(s+1), 'Q'))
            # print(results)
            np.save(lut_path, results)

            print("Resulting LUT size: ", results.shape, "Saved to", lut_path)

if __name__ == "__main__":
    opt_inst = TestOptions()
    opt = opt_inst.parse()

    # load model
    opt = TestOptions().parse()

    modes = [i for i in opt.modes]
    stages = opt.stages

    model = getattr(model, opt.model)

    model_G = model(nf=opt.nf, scale=opt.scale, modes=modes, stages=stages).cuda()

    lm = torch.load('/path/to/your/model')
    model_G.load_state_dict(lm.state_dict(), strict=True)



    def build(opt):
        input_tensor_meta_4d = get_input_tensor(opt)
        input_tensor_meta_3d = get_input_tensor_3d(opt)
        build_m(model_G, input_tensor_meta_4d, opt)
        build_v(model_G, input_tensor_meta_3d, opt)
        build_q(model_G, input_tensor_meta_3d, opt)
    
    def test(test_m=False, test_q=False, test_v=False):
        ## test M consistency <weight, lut>
        if test_m:
            inp = torch.FloatTensor([[
                [[64, 62]],
                [[44, 45]],
                [[130, 131]],
            ]]).cuda()

            inp = inp // 16 * 16
            print("Quantization input: ", inp)
            for i in range(2):
                out = model_G(inp/256, stage=i+1, mode='m')
                print("Network results: ", out*127)

                inp_np = inp.cpu().numpy()
                ids = 0
                LUT_RG = np.load(f'/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s{i+1}_RG.npy')
                index_RG = np.int32(np.sum(inp_np[ids, 0, 0, 0] * 16 + inp_np[ids, 0, 0, 1] * 16*17**1 + inp_np[ids, 1, 0, 1] *16 *17**2 + inp_np[ids, 1, 0, 0] * 16*17**3 ))
                out_RG = LUT_RG[index_RG//256]
                LUT_GB = np.load(f'/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s{i+1}_GB.npy')
                index_GB = np.int32(np.sum(inp_np[ids, 1, 0, 0] * 16 + inp_np[ids, 1, 0, 1] * 16*17**1 + inp_np[ids, 2, 0, 1] *16 *17**2 + inp_np[ids, 2, 0, 0] * 16*17**3 ))
                out_GB = LUT_GB[index_GB//256]
                LUT_BR = np.load(f'/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s{i+1}_BR.npy')
                index_BR = np.int32(np.sum(inp_np[ids, 2, 0, 0] * 16 + inp_np[ids, 2, 0, 1] * 16*17**1 + inp_np[ids, 0, 0, 1] *16*17**2 + inp_np[ids, 0, 0, 0] * 16*17**3 ))
                out_BR = LUT_BR[index_BR//256]
                print("LUT results: ", out_RG, out_GB, out_BR)

        ## test M consistency <weight, lut>
        if test_v:
            inp = torch.FloatTensor([[
                [[43, 34], [0, 123]],
                [[30, 55], [0, 213]],
                [[134, 153], [0, 12]],
            ]]).cuda()

            inp = inp // 16 * 16
            print("Quantization input: ", inp)
            for s in range(stages):
                out = model_G(inp/256, stage=s+1, mode='v')
                print("Network results: ", out*127)

                ids = 0
                inp_np = inp.cpu().numpy()
                LUT_V = np.load(f'/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s{s+1}_V.npy')
                
                index_V_R = np.int32(np.sum(inp_np[ids, 0, 0, 0] * 16*17**2 + inp_np[ids, 0, 0, 1] * 16*17**1 + inp_np[ids, 0, 1, 1] *16 ))
                out_V_R = LUT_V[index_V_R//256]

                index_V_G = np.int32(np.sum(inp_np[ids, 1, 0, 0] * 16*17**2 + inp_np[ids, 1, 0, 1] * 16*17**1 + inp_np[ids, 1, 1, 1] *16 ))
                out_V_G = LUT_V[index_V_G//256]

                index_V_B = np.int32(np.sum(inp_np[ids, 2, 0, 0] * 16*17**2 + inp_np[ids, 2, 0, 1] * 16*17**1 + inp_np[ids, 2, 1, 1] *16 ))
                out_V_B = LUT_V[index_V_B//256]
                print("LUT results: ", out_V_R, out_V_G, out_V_B)

        ## test Q consistency <weight, lut>
        if test_q:
            inp = torch.FloatTensor([[
                [[85]],
                [[36]],
                [[32]],
            ]]).cuda()

            inp = inp // 16 * 16
            print("Quantization input: ", inp)
            for s in range(3):
                out = model_G(inp/256, stage=s+1, mode='q')
                print("Network results: ", out*127)

                ids = 0
                inp_np = inp.cpu().numpy()
                LUT_V = np.load(f'/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s{s+1}_Q.npy')
                
                index_V = np.int32(np.sum(inp_np[ids, 0, 0, 0] * 16*17**2 + inp_np[ids, 1, 0, 0] * 16*17**1 + inp_np[ids, 2, 0, 0] *16 ))
                out_V = LUT_V[index_V//256]
                print("LUT results: ", out_V)

    def round_func(input):
        # Backward Pass Differentiable Approximation (BPDA)
        # This is equivalent to replacing round function (non-differentiable)
        # with an identity function (differentiable) only when backward,
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    test(True, True, True)
    img_lr = np.array(Image.open(
            '/home/styan/DNLUT/exp/dnlut_sidd_20241021/val/SIDD/0000-0000_net.png')).astype(
            np.float32)
        
        # Load GT image
    img_gt = np.array(Image.open('/home/styan/DNLUT/exp/dnlut_sidd_20241021/val/SIDD/0000-0000_gt.png'))

    LUT_M_1_RG = np.load('/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s1_RG.npy')
    LUT_M_1_GB = np.load('/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s1_GB.npy')
    LUT_M_1_BR = np.load('/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s1_BR.npy')
    LUT_M_2_RG = np.load('/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s2_RG.npy')
    LUT_M_2_GB = np.load('/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s2_GB.npy')
    LUT_M_2_BR = np.load('/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s2_BR.npy')
    LUT_V_1 = np.load('/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s1_V.npy')
    LUT_V_2 = np.load('/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s2_V.npy')
    LUT_V_3 = np.load('/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s3_V.npy')
    LUT_V_4 = np.load('/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s4_V.npy')
    LUT_V_5 = np.load('/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s5_V.npy')
    LUT_Q_1 = np.load('/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s1_Q.npy')
    LUT_Q_2 = np.load('/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s2_Q.npy')
    LUT_Q_3 = np.load('/home/styan/DNLUT/exp/dnlut_sidd_20241021/dnlut_LUT_4bit_int8_s3_Q.npy')
    LUT_dict = dict()
    LUT_dict['M_1_RG'] = LUT_M_1_RG
    LUT_dict['M_1_GB'] = LUT_M_1_GB
    LUT_dict['M_1_BR'] = LUT_M_1_BR
    LUT_dict['M_2_RG'] = LUT_M_2_RG
    LUT_dict['M_2_GB'] = LUT_M_2_GB
    LUT_dict['M_2_BR'] = LUT_M_2_BR
    LUT_dict['V1'] = LUT_V_1
    LUT_dict['V2'] = LUT_V_2
    LUT_dict['V3'] = LUT_V_3
    LUT_dict['V4'] = LUT_V_4
    LUT_dict['V5'] = LUT_V_5
    LUT_dict['Q1'] = LUT_Q_1
    LUT_dict['Q2'] = LUT_Q_2
    LUT_dict['Q3'] = LUT_Q_3



    # img_lr = np.array([[[23, 213, 54],[211, 223, 12]],
    #                    [[12, 32, 123],[233, 123, 33]]])
    # print(img_lr.shape)
    ## LUT results
    

    ### LUT M_1
    out_rgb =[]
    for k in ['M_1_RG', 'M_1_GB', 'M_1_BR']:
        pred = 0
        for r in [0, 1, 2, 3]:
            img_lr_rot = np.rot90(img_lr, r)
            h, w, _ = img_lr_rot.shape
            img_in = np.pad(img_lr_rot, ((0,1), (0,1), (0, 0)), mode='edge').transpose((2, 0, 1))
            # s1 = time.time()
            pred += FourSimplexInterpFaster(LUT_dict[k], img_in, h, w, 4, 4 - r,
                                            upscale=1, mode=k)
            # print(time.time()-s1)
        pred = np.clip((pred / 4) + 127, 0, 255)
        pred = np.round(np.clip(pred, 0, 255))
        out_rgb += [pred]
    results_m_1 = np.concatenate(out_rgb, 0)
    

    ### LUT V_1
    out_rgb =[]
    pred = 0
    x = results_m_1.transpose((1, 2, 0))
    for r in [0, 1, 2, 3]:
        img_lr_rot = np.rot90(x, r)
        img_in = np.pad(img_lr_rot, ((0,1), (0,1), (0,0)), mode='edge').transpose((2, 0, 1))
        s1 = time.time()
        pred += Tetrahedral(LUT_dict['V1'], img_in, h, w, 4, 4 - r,
                                        upscale=1, mode='v')
        # print(time.time()-s1)
    pred = np.clip((pred / 4) + 127, 0, 255)
    pred = np.round(np.clip(pred, 0, 255))
    out_rgb += [pred]
    results_v_1 = np.concatenate(out_rgb, 0)
    # print('LUT results V_1:', results_v_1[:,0:3,0:3])
    # print('*'*100)

    out_rgb =[]
    pred = 0
    x = results_v_1.transpose((1, 2, 0))
    for r in [0, 1, 2, 3]:
        img_lr_rot = np.rot90(x, r)
        img_in = np.pad(img_lr_rot, ((0,1), (0,1), (0,0)), mode='edge').transpose((2, 0, 1))
        pred += Tetrahedral(LUT_dict['V2'], img_in, h, w, 4, 4 - r,
                                        upscale=1, mode='v')
    pred = np.clip((pred / 4) + 127, 0, 255)
    pred = np.round(np.clip(pred, 0, 255))
    out_rgb += [pred]
    results_v_2 = np.concatenate(out_rgb, 0)
    # print('LUT results V_2:', results_v_2[:,0:3,0:3])
    # print('*'*100)

    out_rgb =[]
    pred = 0
    x = results_v_2.transpose((1, 2, 0))
    for r in [0, 1, 2, 3]:
        img_lr_rot = np.rot90(x, r)
        img_in = np.pad(img_lr_rot, ((0,1), (0,1), (0,0)), mode='edge').transpose((2, 0, 1))
        pred += Tetrahedral(LUT_dict['V3'], img_in, h, w, 4, 4 - r,
                                        upscale=1, mode='v')
    pred = np.clip((pred / 4) + 127, 0, 255)
    pred = np.round(np.clip(pred, 0, 255))
    out_rgb += [pred]
    results_v_3 = np.concatenate(out_rgb, 0)
    # print('LUT results V_3:', results_v_3[:,0:3,0:3])
    # print('*'*100)

    inp_r = np.concatenate([results_v_1[0:1,:,:], results_v_2[0:1,:,:], results_v_3[0:1,:,:]], 0)
    inp_g = np.concatenate([results_v_1[1:2,:,:], results_v_2[1:2,:,:], results_v_3[1:2,:,:]], 0)
    inp_b = np.concatenate([results_v_1[2:,:,:], results_v_2[2:,:,:], results_v_3[2:,:,:]], 0)

    # print('LUT inp_r: ', inp_r[:,0:3,0:3])

    r = 0
    out_rgb =[]
    pred = Tetrahedral(LUT_dict['Q1'], inp_r, h, w, 4, 4 - r,
                                        upscale=1, mode='q')
    pred = np.clip((pred / 1) + 127, 0, 255)
    pred = np.round(np.clip(pred, 0, 255))
    out_rgb += [pred]
    results_q_1 = np.concatenate(out_rgb, 0)

    r = 0
    out_rgb =[]
    pred = Tetrahedral(LUT_dict['Q2'], inp_g, h, w, 4, 4 - r,
                                        upscale=1, mode='q')
    pred = np.clip((pred / 1) + 127, 0, 255)
    pred = np.round(np.clip(pred, 0, 255))
    out_rgb += [pred]
    results_q_2 = np.concatenate(out_rgb, 0)

    r = 0
    out_rgb =[]
    pred = Tetrahedral(LUT_dict['Q3'], inp_b, h, w, 4, 4 - r,
                                        upscale=1, mode='q')
    pred = np.clip((pred / 1) + 127, 0, 255)
    pred = np.round(np.clip(pred, 0, 255))
    out_rgb += [pred]
    results_q_3 = np.concatenate(out_rgb, 0)
    results_q = np.concatenate([results_q_1, results_q_2, results_q_3], 0)
    # print('LUT Q:', x[:,0:3,0:3])

    out_rgb =[]
    pred = 0
    x = results_q.transpose((1, 2, 0))
    for r in [0, 1, 2, 3]:
        img_lr_rot = np.rot90(x, r)
        img_in = np.pad(img_lr_rot, ((0,1), (0,1), (0,0)), mode='edge').transpose((2, 0, 1))
        pred += Tetrahedral(LUT_dict['V4'], img_in, h, w, 4, 4 - r,
                                        upscale=1, mode='v')
    pred = np.clip((pred / 4) + 127, 0, 255)
    pred = np.round(np.clip(pred, 0, 255))
    out_rgb += [pred]
    results_v_4 = np.concatenate(out_rgb, 0)
    # print('LUT results V_4:', results_v_4[:,0:3,0:3])
    # print('*'*100)

    out_rgb =[]
    x = results_v_4.transpose((1, 2, 0))
    for k in ['M_2_RG', 'M_2_GB', 'M_2_BR']:
        pred = 0
        
        for r in [0, 1, 2, 3]:
            img_lr_rot = np.rot90(x, r)
            h, w, _ = img_lr_rot.shape
            img_in = np.pad(img_lr_rot, ((0,1), (0,1), (0, 0)), mode='edge').transpose((2, 0, 1))
            pred += FourSimplexInterpFaster(LUT_dict[k], img_in, h, w, 4, 4 - r,
                                            upscale=1, mode=k)
        pred = np.clip((pred / 4) + 127, 0, 255)
        pred = np.round(np.clip(pred, 0, 255))
        out_rgb += [pred]
    results_m_2 = np.concatenate(out_rgb, 0)
    # print('LUT results M_2:', results_m_2[:,0:3,0:3])
    # print('*'*100)
    

    out_rgb =[]
    pred = 0
    x = results_m_2.transpose((1, 2, 0))
    for r in [0, 1, 2, 3]:
        img_lr_rot = np.rot90(x, r)
        img_in = np.pad(img_lr_rot, ((0,1), (0,1), (0,0)), mode='edge').transpose((2, 0, 1))
        pred += Tetrahedral(LUT_dict['V5'], img_in, h, w, 4, 4 - r,
                                        upscale=1, mode='v')
    pred = np.clip((pred / 1) + 0, 0, 255)
    pred = np.round(np.clip(pred, 0, 255))
    out_rgb += [pred]
    results_v_5 = np.concatenate(out_rgb, 0)
    # print('LUT results V_5:', results_v_5[:,0:3,0:3])
    # print('*'*100)
    # s2 = time.time()
    # print(s2 - s1)

    Output = results_v_5.astype(np.uint8).transpose((1, 2, 0))
    Image.fromarray(Output).save('./net_denoised.png')

    





####################--------------- NETWORK -----------------------###################################################################
'''
    img_lr = torch.from_numpy(img_lr.transpose((2, 0, 1))).unsqueeze(0).cuda()
    print(img_lr)
    print('*'*100)
    pred = 0
    for r in [0, 1, 2, 3]:
        tmp = round_func(torch.rot90(model_G(F.pad(torch.rot90(img_lr/255, r, [
            2, 3]), (0, 1, 0, 0), mode='replicate'), stage=1, mode='m')[0], (4 - r) % 4, [2, 3]) * 127)
        # print(tmp.shape, r)
        pred += tmp
    avg_factor, bias, norm = 4, 127, 255.0
    x = round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))  / norm
    print('Network M_1: ', x[:,0:3,0:3]*255)


    s = 0
    pred = 0
    for mode in modes:
        for r in [0, 1, 2, 3]:
            pred += round_func(torch.rot90(model_G(F.pad(torch.rot90(x, r, [
                2, 3]), (0, 1, 0, 1), mode='replicate'), stage=s + 1, mode=mode), (4 - r) % 4, [2, 3]) * 127)
    if s + 1 == stages:
        avg_factor, bias, norm = len(modes), 0, 1
        x = round_func((pred / avg_factor) + bias)
        if phase == "train":
            x = x / 255.0
    else:
        avg_factor, bias, norm = len(modes) * 4, 127, 255.0
        x = round_func(torch.clamp((pred / avg_factor) + bias, 0, 255)) / norm
        x1 = x

    print('Network V_1: ', x[:,0:3,0:3]*255)

    s = 1
    pred = 0
    for mode in modes:
        for r in [0, 1, 2, 3]:
            pred += round_func(torch.rot90(model_G(F.pad(torch.rot90(x, r, [
                2, 3]), (0, 1, 0, 1), mode='replicate'), stage=s + 1, mode=mode), (4 - r) % 4, [2, 3]) * 127)
    if s + 1 == stages:
        avg_factor, bias, norm = len(modes), 0, 1
        x = round_func((pred / avg_factor) + bias)
        if phase == "train":
            x = x / 255.0
    else:
        avg_factor, bias, norm = len(modes) * 4, 127, 255.0
        x = round_func(torch.clamp((pred / avg_factor) + bias, 0, 255)) / norm
        x2 = x

    print('Network V_2: ', x[:,0:3,0:3]*255)

    s = 2
    pred = 0
    for mode in modes:
        for r in [0, 1, 2, 3]:
            pred += round_func(torch.rot90(model_G(F.pad(torch.rot90(x, r, [
                2, 3]), (0, 1, 0, 1), mode='replicate'), stage=s + 1, mode=mode), (4 - r) % 4, [2, 3]) * 127)
    if s + 1 == stages:
        avg_factor, bias, norm = len(modes), 0, 1
        x = round_func((pred / avg_factor) + bias)
        if phase == "train":
            x = x / 255.0
    else:
        avg_factor, bias, norm = len(modes) * 4, 127, 255.0
        x = round_func(torch.clamp((pred / avg_factor) + bias, 0, 255)) / norm
        x3 = x

    print('Network V_3: ', x[:,0:3,0:3]*255)

    x_r = torch.cat([x1[:, 0:1], x2[:, 0:1], x3[:, 0:1]], dim=1).to('cuda')
    x_g = torch.cat([x1[:, 1:2], x2[:, 1:2], x3[:, 1:2]], dim=1).to('cuda')
    x_b = torch.cat([x1[:, 2:], x2[:, 2:], x3[:, 2:]], dim=1).to('cuda')

    print('Network x_r:', x_r[:,0:3,0:3])

    r = 0
    pred = round_func(torch.rot90(model_G(F.pad(torch.rot90(x_r, r, [
    2, 3]), (0, 0, 0, 0), mode='replicate'), stage=1, mode='q'), (4 - r) % 4, [2, 3]) * 127)
    avg_factor, bias, norm = 1, 127, 255.0
    x_r = round_func(torch.clamp((pred / avg_factor) + bias, 0, 255)) / norm

    r = 0
    pred = round_func(torch.rot90(model_G(F.pad(torch.rot90(x_g, r, [
    2, 3]), (0, 0, 0, 0), mode='replicate'), stage=2, mode='q'), (4 - r) % 4, [2, 3]) * 127)
    avg_factor, bias, norm = 1, 127, 255.0
    x_g = round_func(torch.clamp((pred / avg_factor) + bias, 0, 255)) / norm

    r = 0
    pred = round_func(torch.rot90(model_G(F.pad(torch.rot90(x_b, r, [
    2, 3]), (0, 0, 0, 0), mode='replicate'), stage=3, mode='q'), (4 - r) % 4, [2, 3]) * 127)
    avg_factor, bias, norm = 1, 127, 255.0
    x_b = round_func(torch.clamp((pred / avg_factor) + bias, 0, 255)) / norm

    x = torch.cat([x_r, x_g, x_b], dim=1).to('cuda')

    print('Network Q: ', x[:,0:3,0:3]*255)

    s = 3
    pred = 0
    for mode in modes:
        for r in [0, 1, 2, 3]:
            pred += round_func(torch.rot90(model_G(F.pad(torch.rot90(x, r, [
                2, 3]), (0, 1, 0, 1), mode='replicate'), stage=s + 1, mode=mode), (4 - r) % 4, [2, 3]) * 127)
    if s + 1 == stages:
        avg_factor, bias, norm = len(modes), 0, 1
        x = round_func((pred / avg_factor) + bias)
        if phase == "train":
            x = x / 255.0
    else:
        avg_factor, bias, norm = len(modes) * 4, 127, 255.0
        x = round_func(torch.clamp((pred / avg_factor) + bias, 0, 255)) / norm

    print('Network V_4: ', x[:,0:3,0:3]*255)


    pred = 0
    for r in [0, 1, 2, 3]:
        tmp = round_func(torch.rot90(model_G(F.pad(torch.rot90(x, r, [
            2, 3]), (0, 1, 0, 0), mode='replicate'), stage=2, mode='m')[0], (4 - r) % 4, [2, 3]) * 127)
        # print(tmp.shape, r)
        pred += tmp
    avg_factor, bias, norm = 4, 127, 255.0
    x = round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))  / norm
    print('Network M_2: ', x[:,0:3,0:3]*255)


    s = 4
    pred = 0
    for mode in modes:
        for r in [0, 1, 2, 3]:
            pred += round_func(torch.rot90(model_G(F.pad(torch.rot90(x, r, [
                2, 3]), (0, 1, 0, 1), mode='replicate'), stage=s + 1, mode=mode), (4 - r) % 4, [2, 3]) * 127)
    if s + 1 == stages:
        avg_factor, bias, norm = len(modes), 0, 1
        x = round_func((pred / avg_factor) + bias)
    else:
        avg_factor, bias, norm = len(modes) * 4, 127, 255.0
        x = round_func(torch.clamp((pred / avg_factor) + bias, 0, 255)) / norm

    print('Network V_5: ', x[:,0:3,0:3])

'''