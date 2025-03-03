'''
本文件对第一阶段工作进行测试
'''

import unittest
import numpy as np
from step.step04 import numerical_diff
from step.step09 import Variable, Square, Exp, square, exp


# 测试 square() 函数的正向/反向传播功能
class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    # 梯度检验
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

# 测试 test() 函数的正向/反向传播功能
class ExpTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = exp(x)
        expected = np.exp(2)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(2.0))
        y = exp(x)
        y.backward()
        expected = np.exp(2)
        self.assertEqual(y.data, expected)

    # 梯度检验
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = exp(x)
        y.backward()
        num_grad = numerical_diff(exp, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)