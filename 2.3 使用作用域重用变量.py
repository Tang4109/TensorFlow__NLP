'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    : 2019/12/4 14:09
@Author  :Zhangyunjia
@FileName:  2.3 使用作用域重用变量.py
@Software: PyCharm
'''

import tensorflow.compat.v1 as tf

graph = tf.Graph()
session = tf.InteractiveSession(graph=graph)


# 1.变量非重用示例：
def very_simple_computation(w):
    x = tf.Variable(tf.constant(5.0, shape=None, dtype=tf.float32), name='x')
    y = tf.Variable(tf.constant(2.0, shape=None, dtype=tf.float32), name='y')
    z = x * w + y ** 2
    return z


z1 = very_simple_computation(2)

for _ in range(10):
    z1 = very_simple_computation(2)

tf.global_variables_initializer().run()
print(session.run(z1))

print([v.name for v in tf.global_variables()])
session.close()


# 2.使用作用域重用变量示例：
# tf.reset_default_graph()
graph=tf.get_default_graph()
# Defining the graph and session
session = tf.InteractiveSession(graph=graph)  # Creates a session


def not_so_simple_computation(w):
    x = tf.get_variable('x', initializer=tf.constant(5.0, shape=None, dtype=tf.float32))
    y = tf.get_variable('y', initializer=tf.constant(2.0, shape=None, dtype=tf.float32))
    z = x * w + y ** 2
    return z


def another_not_so_simple_computation(w):
    x = tf.get_variable('x', initializer=tf.constant(5.0, shape=None, dtype=tf.float32))
    y = tf.get_variable('y', initializer=tf.constant(2.0, shape=None, dtype=tf.float32))
    z = w * x * y
    return z


# Since this is the first call, the variables will be created with following names
# x => scopeA/x, y => scopeA/y
with tf.variable_scope('scopeA'):
    z1 = not_so_simple_computation(tf.constant(1.0, dtype=tf.float32))
# scopeA/x and scopeA/y alread created we reuse them
with tf.variable_scope('scopeA', reuse=True):
    z2 = another_not_so_simple_computation(z1)

# Since this is the first call, the variables will be created with following names
# x => scopeB/x, y => scopeB/y
with tf.variable_scope('scopeB'):
    a1 = not_so_simple_computation(tf.constant(1.0, dtype=tf.float32))
# scopeB/x and scopeB/y alread created we reuse them
with tf.variable_scope('scopeB', reuse=True):
    a2 = another_not_so_simple_computation(a1)

# Say we want to reuse the "scopeA" scope again, since variables are already created
# we should set "reuse" argument to True when invoking the scope
with tf.variable_scope('scopeA', reuse=True):
    zz1 = not_so_simple_computation(tf.constant(1.0, dtype=tf.float32))
    zz2 = another_not_so_simple_computation(z1)

tf.global_variables_initializer().run()
print(session.run([z1, z2, a1, a2, zz1, zz2]))
print([v.name for v in tf.global_variables()])

session.close()











