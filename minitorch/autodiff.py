from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # raise NotImplementedError("Need to implement for Task 1.1")
    tmp = list(vals)
    tmp[arg] += epsilon
    tmp = tuple(tmp)
    originf = f(*vals)
    deltaf = f(*tmp)
    if type(deltaf) == list or type(deltaf) == tuple:
        if len(deltaf) == 1:
            return (deltaf[0] - originf[0]) / epsilon
        else:
            ans = []
            for i in range(len(deltaf)):
                ans.append((deltaf[i] - originf[i]) / epsilon)
    elif type(deltaf) == float or type(deltaf) == int:
        return (deltaf - originf) / epsilon
    # return 0.0


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")
    ans = []
    ansstack = []
    DFS(variable, ansstack)
    for var in ansstack:
        ans.append(var)
    return iter(ans)


def DFS(variable: Variable, stack: List[Variable]) -> None:
    if variable.is_leaf():
        stack.append(variable)
        return
    if variable.is_constant():
        return
    parlist = variable.parents()
    for var in parlist:
        DFS(var, stack)
    stack.append(variable)
    return


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")
    DFS4BP(variable, deriv)


def DFS4BP(variable: Variable, deriv: Any) -> None:
    if variable.is_leaf():
        variable.accumulate_derivative(deriv)
        return
    if variable.is_constant():
        return
    parlist = variable.chain_rule(deriv)
    for var, delta in parlist:
        DFS4BP(var, delta)
    return


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
