---
layout: page
title: Boyd - Convex Optimisation
permalink: /boyd-convex-optimisation/ch1/
---

# Discussion: Chapter 1 - Introduction

This chapter does not need too much external support, but if it serves as a launching pad for the book there's no reason it can't be a launching pad for our discussion.

## Section 1.1 - Mathematical Optimisation

Boyd begins by presenting the general form of a *mathematical optimisation problem*,

$$\begin{align}
\text{minimise} \quad & f_0(x) \\
\text{s.t.} \quad & f_i(x) \leq b_i, \quad i = 1, \dots, m \qquad (1.1)
\end{align}
$$

He then gives the formal definition of a *linear program*,

$$f_i(\alpha x + \beta y) = \alpha f_i(x) + \beta f_i(y)\text{,} \qquad (1.2)$$

and defines *non-linear programs* in the obvious manner. A *convex optimisation problem* is defined as one where the objective and constraints are convex, that is,

$$f_i(\alpha x + \beta y) \leq \alpha f_i(x) + \beta f_i(y)\text{,} \qquad (1.3)$$

where $$\alpha, \beta \in [0, 1]$$ and $$\alpha + \beta = 1$$. This definition is the workhorse in solving many of the examples and problems in the book. A surprising number of problems can be solved with very little imagination by directly proving that this inequality holds or by assuming it does not and deriving a contradiction.

## Section 1.2 - Least-Squares and Linear Programming
### Least Squares

An unconstrained least squares problem is a problem with no constraints (surprise!) and objective function of the form

$$f_0(x) = \Vert Ax - b \Vert^2_2. \qquad (1.4)$$

The notebook shows how we might construct a class to create and solve instances of the least-squares problem. This example is not terribly interesting from the perspective of implementing an optimiser - since we directly call the `numpy.lstsq` solver - but serves to show an example of how we might construct objects, functions, and interfaces using the theory from the book.

Why not implement a least-squares solver? Surprisingly, even for this simple problem, an efficient and numerically stable solver is not straightforward to implement. Matrix inversion is not a numerically stable operation. I only learned this when my course project for the course that inspired this series of notes failed to converge and my lecturer pointed out during a consultation, somewhat horrified, that I was explicitly inverting a matrix in my solution and that this was likely the cause of the issues. A solution to least-squares, however simple, must at the very least address this problem: some invertible matrices will behave just fine if we use the analytical solution given by Boyd, $$x^* = (A^TA)^{-1}A^Tb$$. Others will not.

If you’re interested in exactly how the least-squares problem in the example was solved efficiently and stably, you can give “_umath_linalg.lstsq_m” and “_umath_linalg.lstsq_n” a google and go for a deep-dive into the internals of numpy although you will probably be better served by searching for some information on the general principles involved. I will probably come back to this problem at some point and discuss implementing a least-squares solver but, for now, let us move on.

The key point to this digression is this: there is some extra work required to bridge the gap between mathematical results and implementation. One of the biggest challenges for people that do not happen to be experts in numerical methods is that the pitfalls involved in this process are not always immediately obvious.

The least-squares problem appears in many areas; anyone who has ever done any regression, parameter estimation, or control should recognise the name. The problem has been studied in a great deal of detail and highly efficient solvers exist. Looking at the contour plot in the notebook it might come as little surprise that these problems are relatively easy to solve given their simple "shape". Some modifications can make the method more flexible.

One of the most natural extensions is that of weighted least-squares. In weighted least-squares a weighted version of the cost function is minimised. In normal least-squares we can write $$\Vert Ax - b \Vert^2_2$$ as $$\sum^k_{i=1} (a_i^Tx - b_i)\text{,}$$ and a weighted version of the problem as $$\sum^k_{i=1} w_i(a_i^Tx - b_i)\text{,}$$ with $$w_1, \dots, w_k \geq 0$$.

*Regularisation* is another relatively common extension to least-squares problems where extra terms are added to the least-squares cost function. Boyd gives the simplest example $$\sum^k_{i=1}(a_i^Tx - b_i) + \rho\sum^k_{i=1}x_i^2\text{,}$$ which we could also write $$\Vert Ax - b \Vert^2_2 + x^Tx$$. Regularisation appears in statistical estimation when $$x$$ has a prior estimate. Regularisation can also be used as a tool to try and squeeze out acceptable solutions from an ill-posed problem or to prevent overfitting. Like with the basic least-squares problem I hope to come back and look at some details on this topic.

### Linear Programs

Linear programs are another very well-known class of optimisation problems. In a linear program the objective function and constraints are all linear functions of the optimisation variable.

$$\begin{align}
\text{minimise} \quad & c^Tx \\
\text{s.t.} \quad & a_i^Tx \leq b_i, \quad i = 1, \dots, m \qquad (1.5)
\end{align}
$$

Where $$a_i \in \mathbb{R}^n$$ and $$b_i \in \mathbb{R}$$; evidently we could write the constraints as $$Ax \leq b$$ with $$A \in \mathbb{R}^{m\times n}$$ and $$b \in \mathbb{R^m}$$. Linear programs have also been very deeply studied, and although we have no simple analytical formula for their solution there are many effective solvers for this class of problems.

Boyd chooses the *Chebyshev approximation problem* as his example of an interesting problem which can be formulated as a linear program.

$$\begin{align}
\text{minimise} \max_{i=1,\dots,k} \vert a_i^Tx - b_i \vert \qquad (1.6)
\end{align}
$$

I will offer a little bit of explanation since I had to dig a little to find out what exactly is meant by "the" Chebyshev approximation problem. Chebyshev approximation in general has to do with using Chebyshev polynomials to approximate a given function, but what Boyd refers to here is the question of determining a linear approximation $$Ax$$ which minimises the largest error of approximation between $$Ax$$ and $$b$$ - that is, a Chebyshev approximation *of order 1* specifically. This can be phrased as a linear program with variables $$x \in \mathbb{R}^n$$ and $$t \in \mathbb{R}$$:

$$\begin{align}
\text{minimise} \quad & t \\
\text{s.t.} \quad & a_i^Tx -t \leq b_i, &&\quad i = 1, \dots, k \\
                  & -a_i^Tx -t \leq -b_i, &&\quad i = 1, \dots, k \qquad (1.7)
\end{align}
$$

More details on this can be found in my notes on linear programming, where it is solved as an example problem.

### Sections 1.3 and 1.4: Convex Optimisation and Non-Linear Optimisation

There is nothing in this section that I feel I can add significant value to, but as a key point for the rest of this series of notes and examples I want to repeat a point that Boyd makes which I believe captures the value of this area of study wonderfully: first, techniques for convex (and quasiconvex) optimisation cover a sufficiently large problem domain that a surprising number of problems can be formulated as convex optimisation problems, and second, the techniques used for convex optimisation are general enough to be useful in non-convex problems.

I leave the details of this up to Boyd, he does a far better job in exposition than I could hope to do here. I encourage readers to not skip what he has written in the remainder of Chapter 1, since some of it really is quite relevant in a holistic sense.