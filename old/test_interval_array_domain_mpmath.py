import numpy as np
import pytest
from mpmath import iv
from interval_array_domain_mpmath import AReLU, seq_AReLU, abstract
import tensorflow as tf


@pytest.mark.parametrize('intv, expected_intv', [(iv.mpf([-3, -1]), iv.mpf(0)), (iv.mpf([2, 5]), iv.mpf([2, 5])), (iv.mpf([-2, 2]), iv.mpf([0, 2]))])
def test_AReLU(intv, expected_intv):
    assert AReLU(intv) == expected_intv


@pytest.mark.parametrize('intv_arr, expected_intv_arr', [(iv.matrix([[iv.mpf([1, 2]), iv.mpf([-1, -0.5]), iv.mpf([-2, 2])]]),
                                                          iv.matrix([[iv.mpf([1, 2]), iv.mpf([0, 0]), iv.mpf([0, 2])]]))])
def test_seq_AReLU(intv_arr, expected_intv_arr):
    np.testing.assert_array_equal(seq_AReLU(intv_arr), expected_intv_arr)


@pytest.mark.parametrize('t, d, m', [(tf.constant([[1, 2], [3, 4], [-5, 6]]), 0.1, iv.matrix([[iv.mpf([0.9, 1.1]), iv.mpf([1.9, 2.1])],
                                                                                              [iv.mpf([2.9, 3.1]), iv.mpf([3.9, 4.1])],
                                                                                              [iv.mpf([-5.1, -4.9]), iv.mpf([5.9, 6.1])]]))])
def test_tensor_to_intv(t, d, m):
    np.testing.assert_array_equal(abstract(t, d), m)

