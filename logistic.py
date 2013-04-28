# encoding: utf-8

import numpy as np

float_t = np.float64

class LogisticPermutation(object):

    def __init__(self, key1, key2, key3):
        self.key1 = key1
        self.key2 = key2
        self.key3 = key3
        self.float_1 = float_t(1.)


    def _f(self, x, mu):
        return mu * x * (self.float_1 - x)


    def _get_inverse_dict(self, l):
        return {x: i for i, x in enumerate(l)}


    def _gen_list(self, start, mu, size=80):
        float_l = []
        it = start
        for i in range(size):
            it = self._f(it, mu)
            float_l.append(it)

        float_l_sorted = sorted(float_l)
        float_d = self._get_inverse_dict(float_l_sorted)
        # float_d = {0.521245: 1, 0.654521: 2, 0.791253: 3, ...}

        return [float_d[x] for x in float_l], it


    def _skip_x(self, start, mu, skip):
        ret = start
        for _ in range(skip):
            ret = self._f(ret, mu)
        return ret


    def get_encrypt_mat(self, (height, width)=(60, 80)):
        mat = []

        x0, mu, skip = self.key1
        next_key = self._skip_x(x0, mu, skip)

        for i in range(height):
            row_perm_table, next_key = self._gen_list(next_key, mu, size=width)
            mat.append(row_perm_table)

        x0, mu, skip = self.key2
        next_key = self._skip_x(x0, mu, skip)

        col_perm_table, _ = self._gen_list(next_key, mu, size=height)
        mat.append(col_perm_table)

        return mat


    def get_decrypt_mat(self, (height, width)=(60, 80)):
        enc_mat = self.get_encrypt_mat((height, width))
        dec_mat = [[] for _ in range(height)]

        row_inv_table = self._get_inverse_dict(enc_mat[-1]).values()
        for i, row in enumerate(enc_mat[:-1]):
            dec_mat[row_inv_table[i]] = self._get_inverse_dict(row).values()
        dec_mat.append(row_inv_table)

        return dec_mat


    def get_block_perm_mat(self):
        x0, mu, skip = self.key3
        next_key = self._skip_x(x0, mu, skip)

        return [i + 1 for i in self._gen_list(next_key, mu, size=63)[0]]


    def get_block_inv_perm_mat(self):
        return [i + 1 for i in {x: i for i, x in enumerate(self.get_block_perm_mat())}.values()]



