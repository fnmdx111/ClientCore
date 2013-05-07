# encoding: utf-8

import base64
import cv2
from itertools import product
import logging
import os
import uuid
import requests
from exc import ImageSizeNotMatchException
import logistic
import numpy as np


class ClientCore(object):
    DCT_CONTAINER_TYPE = np.float64
    GRAYSCALE_CONTAINER_TYPE = np.int16
    SEND_ENC_IMAGE_URL = 'send'
    UPLOAD_ENC_IMAGE_URL = 'add'
    LOGIN_URL = 'login'
    LOGOUT_URL = 'logout'
    RETRIEVE_URL = 'retrieve'

    def __init__(self, keys,
                 (size_h, size_w)=(480, 640),
                 server_addr='http://127.0.0.1:5000',
                 cwd=os.getcwd()):
        self.cwd = cwd

        self.size_h = size_h
        self.size_w = size_w

        self.size_b_h = size_h / 8
        self.size_b_w = size_w / 8
        size_b_pair = self.size_b_h, self.size_b_w

        key1, key2, key3 = keys
        self.logistic = logistic.LogisticPermutation(key1, key2, key3)

        self.perm_table = self.logistic.get_encrypt_mat(size_b_pair)
        self.inv_perm_table = self.logistic.get_decrypt_mat(size_b_pair)

        self.block_perm_table = self.logistic.get_block_perm_mat()
        self.inv_block_perm_table = self.logistic.get_block_inv_perm_mat()

        self.server_addr = server_addr

        block_row_idx, block_col_idx = map(lambda end: range(0, end, 8), (size_h, size_w))
        self.block_coordinates = tuple(product(block_row_idx, block_col_idx))

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.session = requests.Session()


    def init_core(self):
        client_id = uuid.uuid1().get_hex()
        return self.session.post(self._gen_url(self.LOGIN_URL),
                                 {'id': client_id}).json()


    def finalize_core(self):
        self.session.post(self._gen_url(self.LOGOUT_URL))


    def _gen_url(self, sub_url):
        return '%s/%s' % (self.server_addr, sub_url)


    def _dct_img(self, img):
        """
        return dct of specified matrix
        :param img: np.ndarray of self.GRAYSCALE_CONTAINER_TYPE
        :return: np.ndarray of self.DCT_CONTAINER_TYPE
        """
        ret = np.ndarray((self.size_b_h, self.size_b_w),
                         dtype=np.dtype((ClientCore.DCT_CONTAINER_TYPE,
                                         (8, 8))))
        mat = np.zeros((8, 8), dtype=self.DCT_CONTAINER_TYPE)

        for r, c in self.block_coordinates:
            mat[:8, :8] = img[r:r + 8, c:c + 8] # dct argument must be matrix of float
            ret[r / 8, c / 8][:, :] = cv2.dct(mat)

        return ret


    def _idct_img(self, mat):
        """
        return inverse dct of specified matrix
        :param mat: np.ndarray of self.DCT_CONTAINER_TYPE
        :return: np.ndarray of self.GRAYSCALE_CONTAINER_TYPE
        """
        ret = np.zeros((self.size_h, self.size_w),
                       dtype=ClientCore.GRAYSCALE_CONTAINER_TYPE)
        mat = mat.reshape((self.size_b_h, self.size_b_w, 8, 8))
        block_float = np.zeros((8, 8), dtype=self.DCT_CONTAINER_TYPE)

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


    def save_img_m(self, img):
        _, buf = cv2.imencode('.jpg', img)
        return str(buf.data)


    def _transform_block(self, block, table):
        ret = np.zeros(block.shape, dtype=ClientCore.DCT_CONTAINER_TYPE)
        ret[0, 0] = block[0, 0]
        for i, p in zip(range(1, 64), table):
            ret[p / 8, p % 8] = block[i / 8, i % 8]

        return ret


    def _transform(self, img, table, block_table):
        """
        transform (permutation) :img: according to :table: and :block_table:
        :param img: np.ndarray of self.DCT_CONTAINER_TYPE
                    that holds the matrix of image after DCT, size is (60, 80, 8, 8)
        :param table: matrix permutation table
        :param block_table: block permutation table
        :return: np.ndarray of self.DCT_CONTAINER_TYPE that holds the matrix
                 after permutation, size if (480, 640)
        """
        tmp = np.zeros((self.size_b_h, self.size_b_w),
                       dtype=np.dtype((ClientCore.DCT_CONTAINER_TYPE,
                                       (8, 8))))
        ret = np.zeros((self.size_b_h, self.size_b_w),
                       dtype=np.dtype((ClientCore.DCT_CONTAINER_TYPE,
                                       (8, 8))))
        for i in range(self.size_b_h):
            for j in range(self.size_b_w):
                tmp[i, j][:, :] = self._transform_block(img[i, table[i][j]],
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
        with open(path, 'rb') as f:
            return self._send_str(self._gen_url(self.SEND_ENC_IMAGE_URL),
                                  img=base64.standard_b64encode(f.read()),
                                  max_count=max_count)


    def send_img(self, img, max_count=10):
        _, img_buf = cv2.imencode('.jpg', img)
        return self._send_str(self._gen_url(self.SEND_ENC_IMAGE_URL),
                              img=base64.standard_b64encode(img_buf.data),
                              max_count=max_count)


    def send_img_raw(self, raw, max_count=10):
        return self._send_str(self._gen_url(self.SEND_ENC_IMAGE_URL),
                              img=base64.standard_b64encode(raw),
                              max_count=max_count)


    def upload_img_by_path(self, path):
        with open(path, 'rb') as f:
            return self._send_str(self._gen_url(self.UPLOAD_ENC_IMAGE_URL),
                                  img=base64.standard_b64encode(f.read()))


    def upload_img(self, img):
        _, img_buf = cv2.imencode('.jpg', img)
        return self._send_str(self._gen_url(self.UPLOAD_ENC_IMAGE_URL),
                              img=base64.standard_b64encode(img_buf.data))


    def upload_img_raw(self, raw):
        return self._send_str(self._gen_url(self.UPLOAD_ENC_IMAGE_URL),
                              img=base64.standard_b64encode(raw))


    def _send_str(self, post_url, **d):
        self.logger.info('posting to %s', post_url)
        return self.session.post(post_url,
                                 data=d).json()


    def _transform_img(self, func, path='', array=None):
        """
        do transformation on image specified by array or path
        :param func: op func e.g. self._enc or self._dec
        :param path: str that holds image path
        :param array: np.ndarray of self.GRAYSCALE_CONTAINER_TYPE
                      that holds source's grayscale array
        :return: np.ndarray of self.GRAYSCALE_CONTAINER_TYPE
        :raise: TypeError
        """
        if path:
            img = self.open_img(path)
        elif isinstance(array, np.ndarray):
            if array.any():
                img = array
            else:
                raise TypeError('no fatal argument provided')
        else:
            raise TypeError('no fatal argument provided')

        return self._idct_img(func(self._dct_img(img)))


    def enc_img(self, path='', array=''):
        """for signature, see self._transform_img"""
        return self._transform_img(self._enc, path=path, array=array)


    def dec_img(self, path='', array=''):
        """for signature, see self._transform_img"""
        return self._transform_img(self._dec, path=path, array=array)


    def _from_raw_to_grayscale(self, raw):
        """
        convert binary data into image's grayscale array
        :param raw: binary data of image
        :return: np.ndarray of self.GRAYSCALE_CONTAINER_TYPE
        """
        return cv2.imdecode(np.fromstring(raw, dtype=np.uint8),
                            cv2.CV_LOAD_IMAGE_GRAYSCALE)


    def write_dec_result(self, data, i):
        """
        write data into local file after decryption (i.e. inverse permutation)
        :param data: binary data of image
        :param i: number of image
        :return: image file path
        """
        return self.write_result(self.dec_img(array=self._from_raw_to_grayscale(data)), i)


    def write_result(self, img, i):
        """
        write data directly into local file
        :param img: np.ndarray of self.GRAYSCALE_CONTAINER_TYPE
        :param i: number of image
        :return: image file path
        """
        f_path = os.path.join(self.cwd, 'results', 'res%s.jpg' % i)
        self.save_img(f_path, img)
        return f_path


    def parse_result(self, _):
        """
        retrieve single result of the result set
        :return: str of image's binary data and distance
        """
        r = self.session.post(self._gen_url(self.RETRIEVE_URL)).json()
        if r['status'] != 'ok':
            return r['result'], ''

        raw, norm = r['result']
        return base64.standard_b64decode(raw), norm



if __name__ == '__main__':
    print os.getcwd()
    key = np.float64(.7000000000000001), np.float64(3.6000000000000001), 211
    cc = ClientCore((key, key, key))
    buf = cc.save_img_m(cc.enc_img(path='8.jpg'))


