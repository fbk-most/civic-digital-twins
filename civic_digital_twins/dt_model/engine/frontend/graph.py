"""Computation Graph Building.

This module allows to build an abstract computation graph using TensorFlow-like
computation primitives and concepts. These primitives and concepts are also similar
to NumPy primitives, albeit with minor naming differences.

This module provides:

1. Basic node types for constants and placeholders
2. Arithmetic operations (add, subtract, multiply, divide)
3. Comparison operations (equal, not_equal, less, less_equal, greater, greater_equal)
4. Logical operations (and, or, xor, not)
5. Mathematical operations (exp, power, log)
6. Shape manipulation operations (expand_dims, squeeze)
7. Reduction operations (sum, mean)
8. Built-in debug operations (tracepoint, breakpoint)
9. Support for infix and unary operators (e.g., `a + b`, `~a`)

The nodes form a directed acyclic graph (DAG) that represents computations
to be performed. Each node implements a specific operation and stores its
inputs as attributes. The graph can then be evaluated by traversing the nodes
and performing their operations using NumPy, TensorFlow, etc.

We anticipate using NumPy/TensorFlow to perform computation based on matrices
of diverse shapes ("tensors"), therefore, we have included operations for shape
manipulation including expanding the dimensions and projecting over the axes. For
example, expand_dims allows to add new axes of size 1 to a tensor's shape, while
project_using_sum allows to compute the sum of tensor elements along specified
axes, thus projecting the tensor onto a lower-dimensional space.

To allow for uniform manipulation, we define the following operation groups:

1. BinaryOp: Operations that take two graph nodes as input
2. UnaryOp: Operations that take one graph node as input
3. AxisOp: Operations that take a graph node and an axis specification as input
and either expand to a higher-dimensional space or reduce to a lower-dimensional
space by projecting over one or more axes using a specific reduction operation.

Here's an example of what you can do with this module:

    >>> from civic_digital_twins.dt_model.engine.frontend import graph
    >>>
    >>> a = graph.placeholder("a", 1.0)
    >>> b = graph.constant(2.0)
    >>> c = a + b
    >>> d = c * c + 1
    >>>
    >>> # Expand to a higher-dimensional space
    >>> e = graph.expand_dims(d, axis=(1,2))
    >>>
    >>> # Project to a lower-dimensional space by summing over axis 0
    >>> f = graph.project_using_sum(e, axis=0)

Like TensorFlow, we support placeholders. That is, variables with a given
name that can be filled in at execution time with concrete values. We also
support constants, which must be bool, float, or int scalars.

Because our goal is to *capture* the arguments provided to function invocations
for later evaluation, we are using classes instead of functions. (We could
alternatively have used closures, but it would have been more clumsy.) To keep
the invoked entities names as close as possible to TensorFlow, we named the
classes using snake_case rather than CamelCase. This is a pragmatic and conscious
choice: violating PEP8 to produce code that reads like TensorFlow.

The main type in this module is the `Node`, representing a node in the
computation graph. Each operation (e.g., `add`) is a subclass of the `Node`
capturing the arguments it has been provided on construction.

Design Decisions
----------------

1. Class-based vs Function-based:
   - Classes capture operation arguments naturally
   - Enable visitor pattern for transformations
   - Allow future addition of operation-specific attributes

2. Snake Case Operation Names:
   - Match NumPy/TensorFlow conventions
   - Improve readability in mathematical context

3. Node Identity:
   - Nodes are identified by their instance identity
   - Enables graph traversal and transformation

4. Gradual, Static Typing:
    - Produce static type errors for explicitly-typed nodes
    - No static type errors otherwise

5. Node Representation:
    - The __repr__ of a node emits the Python code that would generate the node
    - Therefore, the __repr__ representation is round-trippable

Gradual, Static Typing
----------------------

The `Node` class is actually `Node[T]`. If you explicitly
assign types using `placeholder` and `constant`, the
type checker will produce type errors for mismatched operations.

If at least a node is untyped, then operations are always
possible without getting static type errors.

For example:

    class TimeDimension:
        '''Represents the time dimension.'''

    class EnsembleDimension:
        '''Represents the ensemble dimension.'''

    a = graph.constant[TimeDimension](14)
    b = graph.constant[EnsembleDimension](117)
    c = a + b  # This line produces a static type error due to incompatible types

    d = graph.constant[TimeDimension](14)
    e = graph.constant(117)
    f = d + e  # No static type error: untyped nodes default to Unknown

A stricter version of type checking, if desired, will become available
when the minimum Python version becomes 3.13.

Node Representation
-------------------

Given this code:

    a = graph.placeholder("a")
    b = graph.placeholder("b")
    c = a + b

If you print each node, you obtain:

    n1 = graph.placeholder(name="a", default_value=None)
    n2 = graph.placeholder(name="b", default_value=None)
    n3 = graph.add(left=n1, right=n2, name="")

That is, the `__repr__` (and `__str__`) of each node is an SSA representation
where you see the node IDs (the format is `nX` where `X` is the ID) and the
arguments with which they were constructed.

Crucially, if you execute the `__repr__` of each node, you get back the
same graph that you have constructred with the original code.

Node Identity and Equality
--------------------------

Nodes in this module override Python's standard equality operators (`==`, `!=`, etc.)
to create new graph operations rather than test for object equality.

For example:
    x == y   # Creates a graph.equal operation node
    x < y    # Creates a graph.less operation node

When you need to check if two nodes are the same object (identity comparison),
use Python's `is` operator instead:

    x is y   # Tests if x and y are the same object

This behavior impacts code that needs to find nodes in collections like lists:

    # Won't work as expected:
    nodes.index(my_node)  # Uses `==` internally

    # Correct approach:
    next(i for i, n in enumerate(nodes) if n is my_node)
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Generic, Sequence, TypeVar

from .. import atomic, compileflags

Axis = int | tuple[int, ...]
"""Type alias for axis specifications in shape operations."""

Scalar = bool | float | int
"""Type alias for supported scalar value types."""


NODE_FLAG_TRACE = compileflags.TRACE
"""Inserts a tracepoint at the corresponding graph node."""

NODE_FLAG_BREAK = compileflags.BREAK
"""Inserts a breakpoint at the corresponding graph node."""


_id_generator = atomic.Int()
"""Atomic integer generator for unique node IDs."""

T = TypeVar("T")
"""Type associated with a Node."""

C = TypeVar("C")
"""Type associated with boolean conditions."""


def ensure_node(value: Node[T] | Scalar) -> Node[T]:
    """Convert a scalar value to a constant node if necessary."""
    return value if isinstance(value, Node) else constant(value)


class Node(Generic[T]):
    """
    Base class for all computation graph nodes.

    Design Notes
    ------------

    1. Identity Semantics:
        - Nodes use identity-based hashing and equality
        - This allows graph traversal algorithms to work correctly
        - Enables use of nodes as dictionary and sets keys

    2. Debug Support:
        - Nodes carry flags for debugging (trace/break)
        - Names for better error reporting
        - Extensible flag system for future debug features
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.flags = 0
        self.id = _id_generator.add(1)

    def __str__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return repr(self)  # subclasses define suitable __repr__

    def __hash__(self) -> int:
        """Override hash to use identity-based hashing.

        We need to do this because we override the `__eq__` method to support lazy equality.
        """
        return id(self)

    # Arithmetic operators
    def __add__(self, other: Node[T] | Scalar) -> Node[T]:
        """Add two nodes or a node and a scalar."""
        return add(self, ensure_node(other))

    def __radd__(self, other: Node[T] | Scalar) -> Node[T]:
        """Add two nodes or a node and a scalar."""
        return add(ensure_node(other), self)

    def __sub__(self, other: Node[T] | Scalar) -> Node[T]:
        """Subtract two nodes or a node and a scalar."""
        return subtract(self, ensure_node(other))

    def __rsub__(self, other: Node[T] | Scalar) -> Node[T]:
        """Subtract two nodes or a node and a scalar."""
        return subtract(ensure_node(other), self)

    def __mul__(self, other: Node[T] | Scalar) -> Node[T]:
        """Multiply two nodes or a node and a scalar."""
        return multiply(self, ensure_node(other))

    def __rmul__(self, other: Node[T] | Scalar) -> Node[T]:
        """Multiply two nodes or a node and a scalar."""
        return multiply(ensure_node(other), self)

    def __truediv__(self, other: Node[T] | Scalar) -> Node[T]:
        """Divide two nodes or a node and a scalar."""
        return divide(self, ensure_node(other))

    def __rtruediv__(self, other: Node[T] | Scalar) -> Node[T]:
        """Divide two nodes or a node and a scalar."""
        return divide(ensure_node(other), self)

    # Comparison operators
    #
    # See the companion `__hash__` comment.
    def __eq__(self, other: Node[T] | Scalar) -> Node[T]:  # type: ignore
        """Lazily check whether two nodes are equal."""
        return equal(self, ensure_node(other))

    def __ne__(self, other: Node[T] | Scalar) -> Node[T]:  # type: ignore
        """Lazily check whether two nodes are not equal."""
        return not_equal(self, ensure_node(other))

    def __lt__(self, other: Node[T] | Scalar) -> Node[T]:
        """Lazily check whether one node is less than another."""
        return less(self, ensure_node(other))

    def __le__(self, other: Node[T] | Scalar) -> Node[T]:
        """Lazily check whether one node is less than or equal to another."""
        return less_equal(self, ensure_node(other))

    def __gt__(self, other: Node[T] | Scalar) -> Node[T]:
        """Lazily check whether one node is greater than another."""
        return greater(self, ensure_node(other))

    def __ge__(self, other: Node[T] | Scalar) -> Node[T]:
        """Lazily check whether one node is greater than or equal to another."""
        return greater_equal(self, ensure_node(other))

    # Logical operators
    def __and__(self, other: Node[T] | Scalar) -> Node[T]:
        """Lazily check whether one node is logically and with another."""
        return logical_and(self, ensure_node(other))

    def __rand__(self, other: Node[T] | Scalar) -> Node[T]:
        """Lazily check whether one node is logically and with another."""
        return logical_and(ensure_node(other), self)

    def __or__(self, other: Node[T] | Scalar) -> Node[T]:
        """Lazily check whether one node is logically or with another."""
        return logical_or(self, ensure_node(other))

    def __ror__(self, other: Node[T] | Scalar) -> Node[T]:
        """Lazily check whether one node is logically or with another."""
        return logical_or(ensure_node(other), self)

    def __xor__(self, other: Node[T] | Scalar) -> Node[T]:
        """Lazily check whether one node is logically xor with another."""
        return logical_xor(self, ensure_node(other))

    def __rxor__(self, other: Node[T] | Scalar) -> Node[T]:
        """Lazily check whether one node is logically xor with another."""
        return logical_xor(ensure_node(other), self)

    def __invert__(self) -> Node[T]:
        """Lazily check whether one node is logically not."""
        return logical_not(self)


class constant(Generic[T], Node[T]):
    """A constant scalar value in the computation graph.

    Args:
        value: The scalar value to store in this node.
    """

    def __init__(self, value: Scalar, name: str = "") -> None:
        super().__init__(name)
        self.value = value

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.constant(value={self.value}, name='{self.name}')"


class placeholder(Generic[T], Node[T]):
    """Named placeholder for a value to be provided during evaluation.

    Args:
        default_value: Optional default scalar value to use for the
        placeholder if no type is provided at evaluation time.
    """

    def __init__(self, name: str, default_value: Scalar | None = None) -> None:
        super().__init__(name)
        self.default_value = default_value

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.placeholder(name='{self.name}', default_value={self.default_value})"


class BinaryOp(Generic[T], Node[T]):
    """Base class for binary operations.

    Args:
        left: First input node
        right: Second input node
    """

    def __init__(self, left: Node[T], right: Node[T], name="") -> None:
        super().__init__(name)
        self.left = left
        self.right = right


# Arithmetic operations


class add(Generic[T], BinaryOp[T]):
    """Element-wise addition of two tensors."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.add(left=n{self.left.id}, right=n{self.right.id}, name='{self.name}')"


class subtract(Generic[T], BinaryOp[T]):
    """Element-wise subtraction of two tensors."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.subtract(left=n{self.left.id}, right=n{self.right.id}, name='{self.name}')"


class multiply(Generic[T], BinaryOp[T]):
    """Element-wise multiplication of two tensors."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.multiply(left=n{self.left.id}, right=n{self.right.id}, name='{self.name}')"


class divide(Generic[T], BinaryOp[T]):
    """Element-wise division of two tensors."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.divide(left=n{self.left.id}, right=n{self.right.id}, name='{self.name}')"


# Comparison operations


class equal(Generic[T], BinaryOp[T]):
    """Element-wise equality comparison of two tensors."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.equal(left=n{self.left.id}, right=n{self.right.id}, name='{self.name}')"


class not_equal(Generic[T], BinaryOp[T]):
    """Element-wise inequality comparison of two tensors."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.not_equal(left=n{self.left.id}, right=n{self.right.id}, name='{self.name}')"


class less(Generic[T], BinaryOp[T]):
    """Element-wise less-than comparison of two tensors."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.less(left=n{self.left.id}, right=n{self.right.id}, name='{self.name}')"


class less_equal(Generic[T], BinaryOp[T]):
    """Element-wise less-than-or-equal comparison of two tensors."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.less_equal(left=n{self.left.id}, right=n{self.right.id}, name='{self.name}')"


class greater(Generic[T], BinaryOp[T]):
    """Element-wise greater-than comparison of two tensors."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.greater(left=n{self.left.id}, right=n{self.right.id}, name='{self.name}')"


class greater_equal(Generic[T], BinaryOp[T]):
    """Element-wise greater-than-or-equal comparison of two tensors."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.greater_equal(left=n{self.left.id}, right=n{self.right.id}, name='{self.name}')"


# Logical operations


class logical_and(Generic[T], BinaryOp[T]):
    """Element-wise logical AND of two boolean tensors."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.logical_and(left=n{self.left.id}, right=n{self.right.id}, name='{self.name}')"


class logical_or(Generic[T], BinaryOp[T]):
    """Element-wise logical OR of two boolean tensors."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.logical_or(left=n{self.left.id}, right=n{self.right.id}, name='{self.name}')"


class logical_xor(Generic[T], BinaryOp[T]):
    """Element-wise logical XOR of two boolean tensors."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.logical_xor(left=n{self.left.id}, right=n{self.right.id}, name='{self.name}')"


class UnaryOp(Generic[T], Node[T]):
    """Base class for unary operations.

    Args:
        node: Input node
    """

    def __init__(self, node: Node[T], name="") -> None:
        super().__init__(name)
        self.node = node


class logical_not(Generic[T], UnaryOp[T]):
    """Element-wise logical NOT of a boolean tensor."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.logical_not(node=n{self.node.id}, name='{self.name}')"


# Math operations


class exp(Generic[T], UnaryOp[T]):
    """Element-wise exponential of a tensor."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.exp(node=n{self.node.id}, name='{self.name}')"


class power(Generic[T], BinaryOp[T]):
    """Element-wise power operation (first tensor raised to power of second)."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.power(left=n{self.left.id}, right=n{self.right.id}, name='{self.name}')"


pow = power
"""Name alias for power, for compatibility with NumPy naming."""


class log(Generic[T], UnaryOp[T]):
    """Element-wise natural logarithm of a tensor."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.log(node=n{self.node.id}, name='{self.name}')"


class maximum(Generic[T], BinaryOp[T]):
    """Element-wise maximum of two tensors."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.maximum(left=n{self.left.id}, right=n{self.right.id}, name='{self.name}')"


# Conditional operations


class where(Generic[C, T], Node[T]):
    """Selects elements from tensors based on a condition.

    Args:
        condition: Boolean tensor
        then: Values to use where condition is True
        otherwise: Values to use where condition is False
    """

    def __init__(self, condition: Node[C], then: Node[T], otherwise: Node[T], name="") -> None:
        super().__init__(name)
        self.condition = condition
        self.then = then
        self.otherwise = otherwise

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.where(condition=n{self.condition.id}, then=n{self.then.id}, otherwise=n{self.otherwise.id}, name='{self.name}')"


class multi_clause_where(Generic[C, T], Node[T]):
    """Selects elements from tensors based on multiple conditions.

    Args:
        clauses: List of (condition, value) pairs
        default_value: Value to use when no condition is met
    """

    def __init__(self, clauses: Sequence[tuple[Node[C], Node[T]]], default_value: Node[T], name="") -> None:
        super().__init__(name)
        self.clauses = clauses
        self.default_value = default_value

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        clauses = [(f"n{condition.id}", f"n{then.id}") for condition, then in self.clauses]
        return f"n{self.id} = graph.multi_clause_where(clauses={clauses}, default_value=n{self.default_value}, name='{self.name}')"


# Shape-changing operations


class AxisOp(Generic[T], Node[T]):
    """Base class for axis manipulation operations.

    We use these operations to expand a tensor to a higher-dimensional
    space or to reduce its dimensionality by projecting over one or more
    axes using a specific reduction operation.

    Args:
        node: Input tensor
        axis: Axis specification
    """

    def __init__(self, node: Node[T], axis: Axis, name="") -> None:
        super().__init__(name)
        self.node = node
        self.axis = axis


class expand_dims(Generic[T], AxisOp[T]):
    """Adds new axes of size 1 to a tensor's shape.

    This expands the tensor to a higher-dimensional space.
    """

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.expand_dims(node=n{self.node.id}, axis={self.axis}, name='{self.name}')"


class squeeze(Generic[T], AxisOp[T]):
    """Removes axes of size 1 from a tensor's shape."""

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.squeeze(node=n{self.node.id}, axis={self.axis}, name='{self.name}')"


class project_using_sum(Generic[T], AxisOp[T]):
    """Computes sum of tensor elements along specified axes.

    This projects the tensor to a lower-dimensional space.
    """

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.project_using_sum(node=n{self.node.id}, axis={self.axis}, name='{self.name}')"


reduce_sum = project_using_sum
"""Name alias for project_using_sum, for compatibility with yakof, which still
uses this name. We will remove this symbol once the merge of yakof into
the dt-model is complete."""


class project_using_mean(Generic[T], AxisOp[T]):
    """Computes mean of tensor elements along specified axes.

    This projects the tensor to a lower-dimensional space.
    """

    def __repr__(self) -> str:
        """Return a round-trippable SSA representation of the node."""
        return f"n{self.id} = graph.project_using_mean(node=n{self.node.id}, axis={self.axis}, name='{self.name}')"


reduce_mean = project_using_mean
"""Name alias for project_using_mean, for compatibility with yakof, which
still uses this name. We will remove this symbol once the merge of
yakof into the dt-model is complete."""


# Debug operations


def tracepoint(node: Node[T]) -> Node[T]:
    """
    Mark the node as a tracepoint and returns it.

    The tracepoint will take effect while evaluating the node. We will
    print information before evaluating the node, evaluate it, then
    print the result.

    This function acts like the unit in the category with semantic side
    effects depending on the debug operation that is requested.
    """
    node.flags |= NODE_FLAG_TRACE
    return node


def breakpoint(node: Node[T]) -> Node[T]:
    """
    Mark the node as a breakpoint and returns it.

    The breakpoint will cause the interpreter to stop before
    evaluating the node.

    This function acts like the unit in the category with semantic side
    effects depending on the debug operation that is requested.
    """
    node.flags |= NODE_FLAG_TRACE | NODE_FLAG_BREAK
    return node
