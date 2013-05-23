
from core import ClientCore

key = 0.7000000001, 3.600000000, 221
cc = ClientCore((key, key, key), (8, 8))

import numpy as np

_img = np.array(
    [[  16.,   11.,   10.,   16.,   24.,   40.,   51.,   61.],
     [  12.,   12.,   14.,   19.,   26.,   58.,   60.,   55.],
     [  14.,   13.,   16.,   24.,   40.,   57.,   69.,   56.],
     [  14.,   17.,   22.,   29.,   51.,   87.,   80.,   62.],
     [  18.,   22.,   37.,   56.,   68.,  109.,  103.,   77.],
     [  24.,   35.,   55.,   64.,   81.,  104.,  113.,   92.],
     [  49.,   64.,   78.,   87.,  103.,  121.,  120.,  101.],
     [  72.,   92.,   95.,   98.,  112.,  100.,  103.,   99.]],
    dtype=np.float64)

from struct import pack, unpack

def xor_float(_f):
    up_f = list(unpack('@BBBBBBBB', _f.tostring()))
    for i in range(6):
        up_f[i] ^= 0xd9
    up_f[6] ^= 0x0c
    return np.fromstring(pack('@BBBBBBBB', *up_f),
                         dtype=np.float64)[0]


_f = np.float64(23.4321)
assert _f == xor_float(xor_float(_f))

import cv2

_dct_img = cv2.dct(_img)

_dct_img = cc._transform_block(_dct_img, cc.block_perm_table)

# former_shape = _dct_img.shape

# _dct_img = _dct_img.reshape((8, 8))

for i in range(1, 64):
    _dct_img[i / 8, i % 8] = xor_float(_dct_img[i / 8, i % 8])

idct_img = cv2.idct(_dct_img).round()

print idct_img

cc.save_img('a1.jpg', idct_img)

inp = cc.open_img('a1.jpg')

tmp = np.ndarray((8, 8), dtype=np.float64)
tmp[:, :] = inp

_dct_img = cv2.dct(tmp) # .reshape(former_shape)

print _dct_img

for i in range(1, 64):
    _dct_img[i / 8, i % 8] = xor_float(_dct_img[i / 8, i % 8])

_dct_img = cc._transform_block(_dct_img, cc.inv_block_perm_table)

# _dct_img = _dct_img.reshape(former_shape)

print _dct_img

idct_img = cv2.idct(_dct_img).round()

print idct_img

print idct_img - _img

cc.save_img('a2.jpg', idct_img)


cc.save_img('t0.jpg', _img)

img = cc.enc_img(array=_img)

cc.save_img('t1.jpg', img)

__img = cc.open_img('t1.jpg')

tmp[:, :] = __img

dimg = cc.dec_img(array=__img)

cc.save_img('t2.jpg', dimg)

print dimg - _img


