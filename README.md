## naive-bayes-example

This is just a note to remind myself what naive bayes is. It is one of the simplest probabilistic graphical models. The key feature of the model is the strong independence relations imposed on the model.

## what does it do?

It is a standard classification method. As a probabilistic graphical model it looks like:

![nb-pgm](https://ermongroup.github.io/cs228-notes/assets/img/naive-bayes.png)

## how to compute

The joint distribution is given by $p(x,y) = p(y)\prod{p(x_i|y)}$, and our goal for these models is often to compute the posterior

$$ p(y|x) = \frac{p(x,y)}{p(x)}$$

where here, $y$ is often the latent/hidden variable we want to compute. In classification problems, $y$ could be discrete, and so we might impose (Gaussian) priors for our data, eg $p(x|y=1)\sim N(x|\mu, \sigma^2)$ where the mean and variance is computed from the given training data.

We note that the marginal

$$ p(x) = \sum_{\text{classes}}{p(x|y=k)p(y=k)} $$

is expensive, but in this case, computable (as there is a small amount of classification categories). In essence, this computability makes naive bayes algorithms relatively easy, and doesn't make us go crazy things like variational inference.


