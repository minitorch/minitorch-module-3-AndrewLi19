"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> bool:
    return x < y


def eq(x: float, y: float) -> bool:
    return x == y


def max(x: float, y: float) -> float:
    if x >= y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> bool:
    return math.fabs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    if x < 0:
        return 0
    else:
        return x


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    return y / x


def inv_back(x: float, y: float) -> float:
    return -1.0 * (y / math.pow(x, 2))


def relu_back(x: float, y: float) -> float:
    if x > 0:
        return y
    else:
        return 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(function: Callable, listx: Iterable) -> Iterable:
    for item in listx:
        yield function(item)


def zipWith(function: Callable, listx: Iterable, listy: Iterable) -> Iterable:
    listxiter = iter(listx)
    listyiter = iter(listy)
    while True:
        try:
            item1 = next(listxiter)
            item2 = next(listyiter)
            yield function(item1, item2)
        except StopIteration:
            break


def reduce(function: Callable, listx: Iterable) -> float:
    listxiter = iter(listx)
    try:
        ans = next(listxiter)
    except StopIteration:
        return 0.0
    while True:
        try:
            item = next(listxiter)
            ans = function(ans, item)
        except StopIteration:
            break
    return ans


def negList(arg: list) -> list:
    argiter = iter(arg)
    ans = map(neg, argiter)
    return list(ans)


def addLists(arg1: list, arg2: list) -> list:
    arg1iter = iter(arg1)
    arg2iter = iter(arg2)
    ans = zipWith(add, arg1iter, arg2iter)
    return list(ans)


def sum(arg: list) -> float:
    argiter = iter(arg)
    ans = reduce(add, argiter)
    return ans


def prod(arg: list) -> float:
    argiter = iter(arg)
    ans = reduce(mul, argiter)
    return ans
