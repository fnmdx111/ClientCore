# encoding: utf-8

import numpy as np

npf32_1 = np.float32(1.)
npf32_2 = np.float32(2.)

def f(x):
    """
    >>> f(np.float32(.7000001))
    0.019999683
    """
    return npf32_1 - npf32_2 * x * x


def gen_list(key, size=80):
    """
    >>> gen_list(np.float32(.7000001), size=10)
    ([6, 9, 0, 1, 2, 3, 5, 8, 4, 7], 0.68547022)
    """
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
    """
    >>> get_encrypt_mat(np.float32(.7000001), (4, 4))
    [[2, 3, 0, 1], [0, 1, 2, 3], [0, 2, 1, 3], [0, 1, 2, 3], [2, 1, 3, 0]]
    """
    mat = []
    next_key = key
    for i in range(height):
        row_perm_table, next_key = gen_list(next_key, size=width)
        mat.append(row_perm_table)

    col_perm_table, _ = gen_list(next_key, size=height)
    mat.append(col_perm_table)

    return mat


def get_decrypt_mat(key, (height, width)=(60, 80)):
    """
    >>> get_decrypt_mat(np.float32(.7000001), (4, 4))
    [[2, 3, 0, 1], [0, 1, 2, 3], [0, 2, 1, 3], [0, 1, 2, 3], [3, 1, 0, 2]]
    """
    enc_mat = get_encrypt_mat(key, (height, width))
    mat = []
    for row in enc_mat:
        mat.append({x: i for i, x in enumerate(row)}.values())

    return mat


def get_block_perm_mat(key):
    return [i + 1 for i in gen_list(key, size=63)[0]]


def get_block_inv_perm_mat(key):
    return [i + 1 for i in {x: i for i, x in enumerate(get_block_perm_mat(key))}.values()]


def test():
    mat = np.array(range(20)).reshape((4, 5))
    table = get_encrypt_mat(np.float32(.7000001), (4, 5))
    inv_table = get_decrypt_mat(np.float32(.7000001), (4, 5))

    def transform(mat, table):
        ret = np.zeros((4, 5))
        for i in range(4):
            for j in range(5):
                ret[i, j] = mat[table[-1][i], table[i][j]]
        return ret

    print mat
    t_mat = transform(mat, table)
    print t_mat
    t_mat = transform(t_mat, inv_table)
    print t_mat


if __name__ == '__main__':
    # # print get_block_perm_mat(np.float32(.7000001))
    # print get_encrypt_mat(np.float32(.7000001), (60, 80))
    # # print get_block_inv_perm_mat(np.float32(.7000001))
    # print get_decrypt_mat(np.float32(.7000001), (60, 80))
    test()

