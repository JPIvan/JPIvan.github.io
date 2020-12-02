---
layout: page
title: Boyd - Convex Optimisation
permalink: /boyd-convex-optimisation/ch9/
---

# Discussion: Chapter 9 - Unconstrained Minimisation

## 9.2 Descent Methods

The first step in implementing our own algorithm using some of the descent methods Boyd discusses is implementing a line search algorithm. Here we will discuss the idea behind the two line search methods implemented at time of writing. The [notebook](https://nbviewer.jupyter.org/github/JPIvan/optimisation/blob/main/notebooks/boyd_ch9_line_search_golden_section.ipynb) associated with this discussion goes over the details of the implementations found in [line_search.py](https://github.com/JPIvan/optimisation/blob/main/src/line_search.py).

The idea of a line search is to minimise a function in a single direction. If function evaluations are expensive we may content ourselves with an approximation of the minimum. We implement one *exact* line search method, golden section search, and one *inexact* line search method, backtracking line search.

### Exact Line Search - The Method of Golden Sections

Assuming a convex function $$f: \mathbb{R} \to \mathbb{R}$$ has a minimum $$x^*$$ on the interval $$(x_1, x_4)$$. Let $$x_1 < x_2 < x_3 < x_4$$ and $$f(x_2) < f(x_3)$$, we can show that $$x^* \in (x_1, x_3)$$.

If $$x^* \in (x_3, x_4)$$ then there will be some $$\alpha + \beta = 1$$ with non-negative $$\alpha$$ and $$\beta$$ such that $$\alpha x_2 + \beta x^* = x_3$$. Since $$f(x^*)$$ is a minimum there is an $$s$$ such that $$f(x^*)=f(x_2) - s$$. With these and the convexity of $$f$$:

$$\begin{aligned}%
    f(\alpha x_2 + \beta x^*) &\leq \alpha f(x_2) + \beta f(x^*) \\%
    f(x_3) &\leq \alpha f(x_2) + (1-\alpha)(f(x_2) - s) \\%
    f(x_3) &\leq f(x_2) - (1-\alpha) s \\%
    f(x_3) &< f(x_2).
\end{aligned}$$

But we began with $$f(x_2) < f(x_3)$$, so $$x^* \not\in (x_3, x_4)$$. So we have for convex $$f$$ that $$f(x_2) < f(x_3) \rightarrow x^* \in (x_1,x_3)$$ and we can likewise show $$f(x_3) < f(x_2) \rightarrow x^* \in (x_2,x_4)$$.

Excellent! We have a method that, given a bracketed minimum, can narrow the bracketing interval. We can repeat the argument on the new, smaller, interval and continue until the interval is narrow enough for our purposes. We could divide up the interval into thirds, evaluate $$f(x_2)$$ and $$f(x_3)$$ and move on; but I think continuing a little with this line of thought is interesting.

The main challenge of this sort of exact line search is that usually a not-insignificant number of function evaluations is required to hone in on $$x^*$$. We can address this problem a little. If $$f(x_2) < f(x_3)$$ the interval is subsequently reduced from $$(x_1, x_4)$$ to $$(x_1, x_3)$$. If $$f(x_3) < f(x_2)$$ the interval is reduced from $$(x_1, x_4)$$ to $$(x_2, x_4)$$. In the former case the new interval contains $$x_2$$, and in the latter the new interval contains $$x_3$$. The idea of the method of golden sections is to divide up the interval $$(x_1, x_4)$$ such that these points can be reused on the subsequent iteration since we will already have calculated values for $$f(x_2)$$ and $$f(x_3)$$.

A typical shorthand used at this point is to let $$a = x_2-x_1$$, $$b = x_4-x_2$$, $$c = b - a = x_3-x_2$$. If our interval is reduced to $$(x_1, x_3)$$ we would like to have

$$\begin{aligned}%
    \frac{x_4-x_1}{x_3-x_1} &= \frac{x_3-x_1}{x_2-x_1} \\[3pt]
    \frac{a + b}{a + c} &= \frac{a + c}{a} \\[3pt]
    \frac{a + b}{b} &= \frac{b}{a}.
\end{aligned}$$

If our interval is reduced to $$(x_2, x_4)$$ we would like to have

$$\begin{aligned}%
    \frac{x_4-x_1}{x_4-x_2} &= \frac{x_4-x_2}{x_4-x_3} \\[3pt]
    \frac{a + b}{b} &= \frac{b}{b-c} \\[3pt]
    \frac{a + b}{b} &= \frac{b}{a}.
\end{aligned}$$

To clarify, if we have this property then our first interval containing $$(x_1, x_2, x_3, x_4)$$ will be narrowed to either $$(x_1, y, x_2, x_3)$$ or  $$(x_2, x_3, y, x_4)$$ and provided we save the values of $$f(x_2)$$ and $$f(x_3)$$ our next iteration will only need to evaluate $$f(y)$$ in order to continue narrowing the interval.

Rearranging our result we obtain $$a^2+ab-b^2=0$$ and thus $$a = b(-1\pm\sqrt{5})/2$$. Restricting to positive ratios we have $$a = b\varphi^{-1}$$ where $$\varphi$$ is the golden ratio. Some rearranging, taking note of the identity $$\varphi = 1 + \varphi^{-1}$$, gives $$b = a+c = \varphi^{-1}(a+b)$$.

From the equality with $$a+c$$ we thus have $$x_3-x_1 = \varphi^{-1}(x_4-x_1)$$ or, as appears in the implementation, $$x_3 = x_1 + \varphi^{-1}(x_4-x_1)$$. Using the same method for the interval $$(x_1, x_3)$$ yields $$ x_2 = x_1 + \varphi^{-2}(x_4-x_1)$$.

It is from the appearance of $$\varphi$$ as the requisite ratio of the subdivisions of the interval that the method of golden sections derives its name.

With the derivation complete, how then to go about actually implementing this method? Let us describe the procedure somewhat loosely in pseudocode, and then switch over to the [notebook](https://nbviewer.jupyter.org/github/JPIvan/optimisation/blob/main/notebooks/boyd_ch9_line_search_golden_section.ipynb) to look at an implementation in Python.  

```
function goldensection:
    given:
        f: function to minimise
        x_1, x_4: interval including minimum
        precision: return when x_4 - x_1 is below this value
    optional:
        h: width of interval
        x_2, x_3: points in interval to use for search
        fx_2, fx_3: previously calculated values of f(x_2), f(x_3)
    procedure:
        # get everything we need
        if h not given: h = x_4 - x_1
        if h < precision: return something from the interval (x_1, x_4)
        if x_2 not given: x_2 = x_1 + h/(phi^2)
        if x_3 not given: x_3 = x_1 + h/phi
        if fx_2 not given: fx_2 = f(x_2)
        if fx_3 not given: fx_3 = f(x_3)

        # decrease interval size
        if fx_2 < fx_3:
            return goldensection(f, x_1, x_3, precision, # replace x_4 with x_3
                h = x_3 - x_1, x_3 = x_2, fx_3 = f_x2 # reuse x_2
            )
        if fx_3 < fx_2:
            return goldensection(f, x_2, x_4, precision, # replace x_1 with x_2
                h = x_4 - x_2, x_2 = x_3, fx_2 = f_x3 # reuse x_3
            )
```

### Inexact Line Search - Backtracking

(coming soon)