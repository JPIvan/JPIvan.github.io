---
layout: page
title: Boyd - Convex Optimisation
permalink: /boyd-convex-optimisation/
---

# Overview

During my masters I took a course on convex optimisation which used the book Convex Optimisation by Stephen Boyd and Lieven Vandenberghe as a reference text. Aside from really enjoying the course I thought the book itself was worderful (I can't stress how well it is written), but noticed many of my fellow students struggled with using and applying the material covered in the book.

The goal of this project is to supplement anyone working through the book with some select examples and discussions. The source for all the notebooks used in these discussions is [available](https://github.com/JPIvan/optimisation) and contributions are welcome.

Likewise, this entire website is [open-source](https://github.com/JPIvan/JPIvan.github.io). As such, any errors in my writing are also open for correction.

Once again, these notes are intended as an aid to anyone working through Boyd's book. I doubt they will be even fractionally as instructive as a standalone reference. I hope you will find these resources useful.

The links below will take you to a jupyter notebook made for each section, beneath this you will find some extra discussion of the contents of the notebooks.

1. [Overview](/boyd-convex-optimisation) - You are here.
2. [Chapter 1 - Introduction](https://nbviewer.jupyter.org/github/JPIvan/optimisation/blob/main/notebooks/boyd_ch1.ipynb)

# Discussion: Chapter 1 - Introduction

This chapter does not need too much external support, but in the notebook there are some details that are glossed over which are important for anyone who looks to begin using the techniques in the textbook.

Looking at the least_squares class, if we pull away the extra details, the callable looks something like this:

```python
def __call__(self, x):
    """ Given x, returns the norm of Ax - b.
    Args:
        x: 2-d numpy matrix (column vector)
    Returns:
        || A*x - b ||_2^2
    """
    return np.linalg.norm(self.A @ x - self.b)**2
```

One of the most popular libraries for performing optimisation in python, scipy.optimise, is simplest to use if you find a way to write your functions as functions of a single vector variable. Typically you will use a numpy array. If you would like to optimise

```python
def func(x, y):
    return x**2 + y**2
```

it will save you a lot of headache later to become accustomed to writing this as

```python
def func(x):
    """ We assume we will recieve a numpy array with 2 elements.
    """
    return np.sum(x**2)
```

instead.

The other thing I would like to point out is that, in this case, I used the numpy least-squares solver. Why? Implementing an efficient, numerically stable optimiser is challenging.

Even for something as simple as least-squares there are pitfalls. Matrix inversion is not a numerically stable operation. I only learned this when my course project for the course that inspured this series of notes failed to converge and my lecturer pointed out during a consultation, somewhat horrified, that I was explicitly inverting a matrix in my solution and that this was likely the cause of the issues. A solution to least-squares, however simple, must at the very least address this problem: some invertible matrices will behave just fine if we use the analytical solution given by Boyd. Others will not.

If you're interested in exactly how the least-squares problem in the example was solved efficiently and stably, you can give "_umath_linalg.lstsq_m" and "_umath_linalg.lstsq_n" a google and go for a deep-dive into the internals of numpy although you will probably be better served by searching for some information on the general principles involed.

The key point, however, is this: there is some extra work required to bridge the gap between mathematical results and implementation. One of the biggest challenges for people that do not happen to be experts in numerical methods is that the pitfalls involved in this process are not always immediately obvious. We will have to deal with these problems, but it will save time and frustration to use libraries that others have painstakingly crafted, optimised, and tested wherever we can permit it.