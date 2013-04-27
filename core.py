import cv2
from itertools import product
from exc import ImageSizeNotMatchException
import logistic
import numpy as np


class ClientCore(object):
    DCT_CONTAINER_TYPE = np.int64
    GRAYSCALE_CONTAINER_TYPE = np.int16
    SEND_ENC_IMAGE_ADDR = 'send'

    def __init__(self, key, (size_h, size_w)=(480, 640), server_addr='http://127.0.0.1'):
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


    def dct_img(self, img):
        ret = np.ndarray((self.size_b_h, self.size_b_w),
                         dtype=np.dtype((ClientCore.DCT_CONTAINER_TYPE,
                                         (8, 8))))
        mat = np.zeros((8, 8), dtype=np.float64)

        for r, c in self.block_coordinates:
            mat[:8, :8] = img[r:r + 8, c:c + 8] # dct argument must be matrix of float
            ret[r / 8, c / 8][:, :] = cv2.dct(mat)

        return ret


    def idct_img(self, mat):
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


    def transform_block(self, block, table):
        ret = np.zeros(block.shape, dtype=ClientCore.DCT_CONTAINER_TYPE)
        ret[0, 0] = block[0, 0]
        for i, p in zip(range(1, 64), table):
            ret[p / 8, p % 8] = block[i / 8, i % 8]

        return ret


    def transform(self, img, table, block_table):
        ret = np.zeros((self.size_b_h, self.size_b_w),
                       dtype=np.dtype((ClientCore.DCT_CONTAINER_TYPE,
                                       (8, 8))))
        for i in range(self.size_b_h):
            for j in range(self.size_b_w):
                ret[i, j][:, :] = self.transform_block(img[i, table[i][j]],
                                                       block_table)

        ret = ret.reshape((self.size_h, self.size_w))

        return ret


    def enc(self, img):
        return self.transform(img,
                              self.perm_table,
                              self.block_perm_table)


    def dec(self, img):
        return self.transform(img,
                              self.inv_perm_table,
                              self.inv_block_perm_table)



if __name__ == '__main__':
    cc = ClientCore(np.float32(.7000001))
    img = cc.open_img('7.jpg')
    mat = cc.dct_img(img)
    mat = cc.enc(mat)
    img = cc.idct_img(mat)
    cc.save_img('0.jpg', img)

    img = cc.open_img('0.jpg')
    mat = cc.dct_img(img)
    mat = cc.dec(mat)
    img = cc.idct_img(mat)
    cc.save_img('8.jpg', img)

