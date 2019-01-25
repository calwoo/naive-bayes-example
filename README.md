### naive-bayes-example

This is just a note to remind myself what naive bayes is. It is one of the simplest probabilistic graphical models. The key feature of the model is the strong independence relations imposed on the model.

## what does it do?

It is a standard classification method. As a probabilistic graphical model it looks like:

![nb-pgm](https://ermongroup.github.io/cs228-notes/assets/img/naive-bayes.png)

The joint distribution is given by $p(x,y) = p(y)\prod{p(x_i|y)}$, and our goal for these models is often to compute the posterior

$$ p(y|x_i) = \frac{p(x_i,y)}{p(x_i)}$$


