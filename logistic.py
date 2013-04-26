# encoding: utf-8

import numpy as np

npf32_1 = np.float32(1.)
npf32_2 = np.float32(2.)

def f(x):
    return npf32_1 - npf32_2 * x * x


def gen_list(key, size=80):
    float_l = []
    it = key
    for i in range(size):
        it = f(it)
        float_l.append(it)

    float_l_sorted = sorted(float_l)
    float_d = {x : i for i, x in enumerate(float_l_sorted)}
    # float_d = {0.521245: 1, 0.654521: 2, 0.791253: 3, ...}
    return [float_d[x] for x in float_l], it


def get_encrypt_mat(key, (height, width)=(60, 80)):
    mat = []
    next_key = key
    for i in range(height):
        row_perm_table, next_key = gen_list(next_key, size=width)
        mat.append(row_perm_table)

    col_perm_table, _ = gen_list(next_key, size=height)
    mat.append(col_perm_table)

    return mat


def get_decrypt_mat(key, (height, width)=(60, 80)):
    enc_mat = get_encrypt_mat(key, (height, width))
    mat = []
    for row in enc_mat:
        mat.append({x: i for i, x in enumerate(row)}.values())

    return mat


def get_block_perm_mat(key):
    return gen_list(key, size=63)[0]


def get_block_inv_perm_mat(key):
    return {x: i for i, x in enumerate(get_block_perm_mat(key))}.values()



if __name__ == '__main__':
    # print get_block_perm_mat(np.float32(.7000001))
    print get_encrypt_mat(np.float32(.7000001), (60, 80))
    # print get_block_inv_perm_mat(np.float32(.7000001))
    print get_decrypt_mat(np.float32(.7000001), (60, 80))

