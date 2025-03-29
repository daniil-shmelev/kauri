# RootedTrees

<p align="center">
<img src="example_plots/antipode.png" width="350">
</p>

An implementation of the Butcher-Connes-Kreimer Hopf algebra of rooted trees [[Connes & Kreimer, 1999](#reference)], 
commonly used for the analysis of B-series and Runge-Kutta schemes. The Hopf algebra is given by
$(\mathcal{H}, \Delta,\mu,\varepsilon, \emptyset, S)$, where<br>
- $\mathcal{H}$ is the set of all linear combinations of forests of trees
- Multiplication $\mu$ is defined as the commutative juxtaposition of trees
- Comultiplication $\Delta$ is defined by
$$\Delta(t) = t \otimes I + I \otimes t + \sum_{s \subset t} s \otimes [t\setminus s]$$
where the sum is over proper rootes subtrees $s$ of $t$, and $[t\setminus s]$ is the product of all branches fromed when
$s$ is erased from $t$.
- The unit $\emptyset$ is the empty tree
- The counit $\varepsilon$ is given by $\varepsilon(\tau) = 1$ if $\tau = \emptyset$ and $0$ otherwise
- The antipode $S$ is defined by
$$S(t) = -t - \sum_{s \subset t} S([t \setminus s])s, \quad S(\bullet) = -\bullet.$$

Given two maps $f,g : \mathcal{H} \to \mathcal{H}$, we define their product map by
$$(f\cdot g)(\tau) = \mu \circ (f \otimes g) \circ \Delta(\tau).$$

## Objects

The implementation works with three objects,
- Tree
- Forest - a product of trees
- ForestSum - a linear combination of forests

and supports standard arithmetic operations *, +, - with any combination of the above structures and scalars. For example:

```python
t0 = Tree(None) #The empty tree
t1 = Tree([]) #The single-node tree
t2 = Tree([[]]) #Tree with 2-nodes
t3 = Tree([[],[]]) #Cherry tree

f1 = Forest([t1, t2])
f2 = Forest([t3])

print(f1 == (t1 * t2)) #This will evaluate to True

s1 = ForestSum([f1, f2], [2, -1]) #2*f1 - f2
s2 = s1 + 5*t0

print(s1 == (2*f1 - t3)) #This will evaluate to True

print(s2 == (2*f1 - t3 + 5)) #This will evaluate to True, since the empty tree t0
                             #is treated as equivalent to the scalar 1.
```

There are two ways of viewing the above objects. `print(...)` will show the list representation, whilst `display(...)`
will plot the trees. For example, `print(s2)` will output
```python
2*[] [[]] + -1*[[], []] + 5*∅
```
whilst `display(s2)` plots:<br><br>
<img src="example_plots/example.png" width="150">

## Functions

```python
t.numNodes() #Returns the number of nodes in a tree or forest t
t.factorial() #Returns the tree factorial of a tree or forest t
t.sorted() #Returns the sorted representation of the tree, with the heaviest branches moved to the left
t.join() #For a forest t, returns the tree formed by joining the trees of the forest with a common root
t.unjoin() #For a tree t, returns the forest formed by deleting the root
t.split() #For a tree t, returns a list of truncs and a list of corresponding branches, split according to the coproduct Delta
t.antipode() #Returns the antipode of a tree, forest or forest sum t
t.sign() #Returns t if t.numNodes() is even, otherwise -t
t.signed_antipode() #Returns the composition of the sign and antipode functions
t.asForest() #For a tree t, returns t as a forest
t.asForestSum() #For a tree or forest t, returns t as a forest sum
t.singleton_reduced() #For a forest or forest sum t, removes redundant occurences of Tree([]) in each forest
```

Additionally, given two functions `func1`, `func2` defined as maps from trees to scalars, trees, forests or forest sums, the
following functions will work with `func1`, `func2` as if they were multiplicative linear maps.

```python
t.apply(func1) #Applies func1 to a tree, forest or forest sum t
t.apply_product(func1, func2) #Applies the product map func1*func2 to t
t.apply_power(func1, n) #Applys the n^th power of func1 to t
```

For convenience, we provide a sample set of functions to use in place of func1, func2 listed below, although one may also
define custom functions.

```python
ident(t) #Returns t
counit(t) #Returns 1 if t is the empty tree and 0 otherwise
S(t) #Returns t.antipode()
exact_weights(t) #Returns 1 / t.factorial()
RK_elementary_weights(t, A, b) #Returns the elementary weights for an RK scheme with parameters (A,b)
```

## Citation

```bibtex
@misc{shmelev2025rootedtrees,
  title={RootedTrees: A Python implementation of the Butcher-Connes-Kreimer Hopf algebra of planar rooted trees},
  author={Shmelev, Daniil},
  year={2025},
  howpublished={\url{https://github.com/daniil-shmelev/RootedTrees}}
}
```

## References
<a name="reference"></a>
- Connes, A., & Kreimer, D. (1999). *Hopf algebras, renormalization and noncommutative geometry*. In *Quantum field theory: perspective and prospective* (pp. 59–109). Springer.