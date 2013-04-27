# encoding: utf-8

"""
    使用方法
    在你的窗口类里新建一个ClientCore变量，例如
        self.core = ClientCore(key=0.7000001,
                               cwd=os.getcwd())
    其中key为密钥，cwd为当前工作目录（cwd设为默认值即可）
    ClientCore类主要方法有
        + open_img(path)
            按灰度模式打开图片，返回numpy.ndarray数组，
            其中数组元素为每一像素的灰度值
        + save_img(path)
            将灰度值数组保存为JPEG文件
        + enc_img(path) -> img_enc 或
          enc_img(img) -> img_enc 其中img和img_enc是numpy.ndarray数组
            加密置换图片，并不改变img的内容
        + dec_img(path) -> img_dec 或
          dec_img(img) -> img_dec 其中img和img_dec是numpy.ndarray数组
            解密置换图片，并不改变img的内容
        + send_img_by_path(path, max_count=10) -> response
            将指定路径的JPEG文件发送至服务器进行检索
        + send_img(img, max_count=10) -> response
            将指定的numpy.ndarray类型的数据发送至服务器进行检索
        + parse_result(response) -> int
            解析服务器返回的数据，
            密变换后保存至cwd/results目录下，
            按返回的顺序从零开始编号，
            返回值为服务器返回数据个数
    具体的使用例子可以看最后的if __name__ == '__main__':段。

    ClientCore模块使用方法建议：
        在ui工程下新建一个文件夹libs，然后把core.py，exc.py，logistic.py拷进去，
        然后在你的ui代码里的第一行加上
            # encoding: utf-8
        换一行之后写
            from libs.core import ClientCore
    应该就没什么问题了，直接给各种按钮绑定ClientCore实例的方法就行了。
"""

import base64
import cv2
from itertools import product
import logging
import os
import uuid
import requests
from exc import ImageSizeNotMatchException, ServerError
import logistic
import numpy as np


class ClientCore(object):
    DCT_CONTAINER_TYPE = np.int64
    GRAYSCALE_CONTAINER_TYPE = np.int16
    SEND_ENC_IMAGE_URL = 'send'

    def __init__(self, key,
                 (size_h, size_w)=(480, 640),
                 server_addr='http://127.0.0.1:5000',
                 cwd=os.getcwd()):
        self.cwd = cwd

        self.key = key
        self.size_h = size_h
        self.size_w = size_w
        self.size_b_h = size_h / 8
        self.size_b_w = size_w / 8

        self.perm_table = logistic.get_encrypt_mat(self.key, (self.size_b_h, self.size_b_w))
        self.inv_perm_table = logistic.get_decrypt_mat(self.key, (self.size_b_h, self.size_b_w))

        self.block_perm_table = logistic.get_block_perm_mat(self.key)
        self.inv_block_perm_table = logistic.get_block_inv_perm_mat(self.key)

        self.server_addr = server_addr

        block_row_idx, block_col_idx = map(lambda end: range(0, end, 8), (size_h, size_w))
        self.block_coordinates = tuple(product(block_row_idx, block_col_idx))

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


    def _dct_img(self, img):
        ret = np.ndarray((self.size_b_h, self.size_b_w),
                         dtype=np.dtype((ClientCore.DCT_CONTAINER_TYPE,
                                         (8, 8))))
        mat = np.zeros((8, 8), dtype=np.float64)

        for r, c in self.block_coordinates:
            mat[:8, :8] = img[r:r + 8, c:c + 8] # dct argument must be matrix of float
            ret[r / 8, c / 8][:, :] = cv2.dct(mat)

        return ret


    def _idct_img(self, mat):
        ret = np.zeros((self.size_h, self.size_w),
                       dtype=ClientCore.GRAYSCALE_CONTAINER_TYPE)
        mat = mat.reshape((self.size_b_h, self.size_b_w, 8, 8))
        block_float = np.zeros((8, 8), dtype=np.float64)

        for i in range(self.size_b_h):
            for j in range(self.size_b_w):
                block_float[:, :] = mat[i, j]
                ret[i * 8:i * 8 + 8, j * 8:j * 8 + 8] = cv2.idct(block_float)

        return ret


    def open_img(self, path):
        img = cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        h, w = img.shape
        if (h, w) != (self.size_h, self.size_w):
            raise ImageSizeNotMatchException

        return img


    def save_img(self, path, img):
        cv2.imwrite(path, img)


    def __transform_block(self, block, table):
        ret = np.zeros(block.shape, dtype=ClientCore.DCT_CONTAINER_TYPE)
        ret[0, 0] = block[0, 0]
        for i, p in zip(range(1, 64), table):
            ret[p / 8, p % 8] = block[i / 8, i % 8]

        return ret


    def _transform(self, img, table, block_table):
        tmp = np.zeros((self.size_b_h, self.size_b_w),
                       dtype=np.dtype((ClientCore.DCT_CONTAINER_TYPE,
                                       (8, 8))))
        ret = np.zeros((self.size_b_h, self.size_b_w),
                       dtype=np.dtype((ClientCore.DCT_CONTAINER_TYPE,
                                       (8, 8))))
        for i in range(self.size_b_h):
            for j in range(self.size_b_w):
                tmp[i, j][:, :] = self.__transform_block(img[i, table[i][j]],
                                                       block_table)
        for i in range(self.size_b_h):
            ret[i, :] = tmp[table[-1][i]]

        ret = ret.reshape((self.size_h, self.size_w))

        return ret


    def _enc(self, img):
        return self._transform(img,
                              self.perm_table,
                              self.block_perm_table)


    def _dec(self, img):
        return self._transform(img,
                              self.inv_perm_table,
                              self.inv_block_perm_table)


    def send_img_by_path(self, path, max_count=10):
        post_url = '%s/%s' % (self.server_addr,
                              ClientCore.SEND_ENC_IMAGE_URL)
        with open(path, 'rb') as f:
            return self._send_str(post_url,
                                  img=base64.standard_b64encode(f.read()),
                                  max_count=max_count)


    def send_img(self, img, max_count=10):
        _, img_buf = cv2.imencode('.jpg', img)
        post_url = '%s/%s' % (self.server_addr,
                              ClientCore.SEND_ENC_IMAGE_URL)
        return self._send_str(post_url,
                              img=base64.standard_b64encode(img_buf.data),
                              max_count=max_count)


    def _send_str(self, post_url, **d):
        self.logger.info('posting to %s', post_url)
        return requests.post(post_url,
                             data=d)


    def _transform_img(self, func, path='', array=''):
        if path:
            img = self.open_img(path)
        elif array:
            img = array
        else:
            raise TypeError('no fatal argument provided')

        return self._idct_img(func(self._dct_img(img)))


    def enc_img(self, path='', array=''):
        return self._transform_img(self._enc, path=path, array=array)


    def dec_img(self, path='', array=''):
        return self._transform_img(self._dec, path=path, array=array)


    def parse_result(self, response):
        r = response.json()
        if r['status'] != 'ok':
            raise ServerError
        rnd_path = '%s.jpg' % uuid.uuid1().get_hex()
        for idx, data in enumerate(r['results']):
            with open(rnd_path, 'wb') as tmp:
                print >> tmp, base64.standard_b64decode(data)
                tmp.flush()
                self.save_img('%s/results/res%s.jpg' % (self.cwd, idx),
                              self.dec_img(path=rnd_path))
        if os.path.isfile(rnd_path):
            os.remove(rnd_path)

        return len(r['results'])


if __name__ == '__main__':
    cc = ClientCore(np.float32(.7000001))
    r = cc.send_img(cc.enc_img(path='7.jpg'))
    r = cc.parse_result(r)
    cc.logger.info('processed %s results', r)

