"""
Backpropagation from Scratch
==============================

THE BIG IDEA:
    Backpropagation is NOT a mysterious algorithm. It is simply the CHAIN RULE
    from calculus, applied systematically through a graph of operations. That's it.

    If you understand the chain rule, you understand backpropagation.
    This module builds that understanding from the ground up.

WHAT IS BACKPROPAGATION?
    Training a neural network means finding weights that minimize a loss function.
    To minimize the loss, we need the GRADIENT of the loss with respect to every
    weight — that tells us which direction to adjust each weight to reduce the loss.

    Backpropagation ("backward propagation of errors") computes these gradients
    efficiently by working backward through the network, applying the chain rule
    at each step.

    Forward pass:  input --> hidden layers --> output --> loss  (compute predictions)
    Backward pass: input <-- hidden layers <-- output <-- loss  (compute gradients)

THE CHAIN RULE — THE MATHEMATICAL FOUNDATION:
    +-----------------------------------------------------------------+
    |                                                                   |
    |   If y = f(g(x)), then:                                          |
    |                                                                   |
    |       dy/dx = dy/dg * dg/dx                                      |
    |                                                                   |
    |   In words: the derivative of a COMPOSITION of functions equals   |
    |   the PRODUCT of the derivatives of each function.                |
    |                                                                   |
    |   More generally, for a chain x -> a -> b -> ... -> L:            |
    |                                                                   |
    |       dL/dx = dL/db * db/da * da/dx                              |
    |                                                                   |
    |   Each factor is a LOCAL derivative — easy to compute.            |
    |   The chain rule MULTIPLIES them together.                        |
    |                                                                   |
    +-----------------------------------------------------------------+

    WORKED EXAMPLE WITH NUMBERS:
        Let f(x) = (2x + 3)^2.  What is df/dx at x = 1?

        Step 1: Decompose into simple operations.
            a = 2x        (multiply)
            b = a + 3     (add)
            f = b^2       (square)

        Step 2: Forward pass — compute values.
            x = 1
            a = 2 * 1 = 2
            b = 2 + 3 = 5
            f = 5^2 = 25

        Step 3: Backward pass — compute gradients using chain rule.
            df/df = 1                          (seed the backward pass)
            df/db = 2b = 2 * 5 = 10           (derivative of b^2 is 2b)
            df/da = df/db * db/da = 10 * 1 = 10   (derivative of a+3 w.r.t a is 1)
            df/dx = df/da * da/dx = 10 * 2 = 20   (derivative of 2x w.r.t x is 2)

        Verify: f(x) = (2x+3)^2 = 4x^2 + 12x + 9
                f'(x) = 8x + 12
                f'(1) = 8 + 12 = 20  <-- matches!

COMPUTATIONAL GRAPHS — HOW NEURAL NETWORKS ORGANIZE COMPUTATIONS:
    A computational graph represents a mathematical expression as a directed
    acyclic graph (DAG), where:

        - Each NODE is an operation (add, multiply, sigmoid, etc.)
        - Each EDGE carries a value (forward) or a gradient (backward)
        - The FORWARD PASS computes values from inputs to output
        - The BACKWARD PASS computes gradients from output to inputs

    Example: f(x, w, b) = sigmoid(w * x + b)

                  x ----+
                         |
                  w ----[*]----> wx ----[+]----> z ----[sigmoid]----> y
                                         |
                  b --------------------+

        Forward pass:  x=0.5, w=2.0, b=0.1
            wx = 2.0 * 0.5 = 1.0
            z  = 1.0 + 0.1 = 1.1
            y  = sigmoid(1.1) = 0.7502601

        Backward pass:  (assume dy/dy = 1)
            dy/dz   = y * (1 - y)       = 0.7503 * 0.2497 = 0.1874
            dy/db   = dy/dz * dz/db     = 0.1874 * 1      = 0.1874
            dy/d(wx)= dy/dz * dz/d(wx)  = 0.1874 * 1      = 0.1874
            dy/dw   = dy/d(wx) * d(wx)/dw = 0.1874 * x    = 0.1874 * 0.5 = 0.0937
            dy/dx   = dy/d(wx) * d(wx)/dx = 0.1874 * w    = 0.1874 * 2.0 = 0.3749

KEY GRADIENT RULES FOR COMMON OPERATIONS:
    +-----------------------------------------------------------------------+
    | Operation       | Forward       | Local Gradient (backward)            |
    +-----------------------------------------------------------------------+
    | Addition: a+b   | c = a + b     | dc/da = 1,  dc/db = 1               |
    |                 |               | Gradient passes through UNCHANGED    |
    +-----------------------------------------------------------------------+
    | Multiply: a*b   | c = a * b     | dc/da = b,  dc/db = a               |
    |                 |               | Gradient SWAPS the values            |
    +-----------------------------------------------------------------------+
    | Square: a^2     | c = a^2       | dc/da = 2a                           |
    +-----------------------------------------------------------------------+
    | ReLU: max(0,a)  | c = max(0,a)  | dc/da = 1 if a > 0, else 0          |
    |                 |               | Gradient is a GATE: pass or block    |
    +-----------------------------------------------------------------------+
    | Sigmoid: s(a)   | c = 1/(1+e^-a)| dc/da = c * (1 - c)                 |
    |                 |               | Gradient depends on OUTPUT value     |
    +-----------------------------------------------------------------------+
    | MatMul: W @ x   | y = W @ x     | dy/dW = outer(grad, x)              |
    |                 |               | dy/dx = W^T @ grad                   |
    +-----------------------------------------------------------------------+

VANISHING AND EXPLODING GRADIENTS:
    Because backpropagation MULTIPLIES gradients at each layer (chain rule),
    deep networks face two dangers:

    1. VANISHING GRADIENTS: If local gradients are small (< 1), the product
       shrinks exponentially with depth. Result: early layers learn NOTHING.
       - Sigmoid and tanh saturate for large inputs --> tiny gradients
       - Fix: use ReLU, residual connections, batch normalization

    2. EXPLODING GRADIENTS: If local gradients are large (> 1), the product
       grows exponentially. Result: weights blow up to NaN.
       - Fix: gradient clipping, careful weight initialization

    Example: 10 layers, each with gradient 0.5:
        Product = 0.5^10 = 0.00097  (vanished! almost zero)

    Example: 10 layers, each with gradient 2.0:
        Product = 2.0^10 = 1024.0   (exploded! way too large)

LEARNING RESOURCES:
    - Andrej Karpathy, "Micrograd" (build autograd in ~100 lines):
      https://github.com/karpathy/micrograd
    - VIDEO: Andrej Karpathy, "The spelled-out intro to neural networks
      and backpropagation":
      https://www.youtube.com/watch?v=VMj-3S1tku0
    - 3Blue1Brown, "What is backpropagation really doing?":
      https://www.youtube.com/watch?v=Ilg3gGewQ5U
    - CS231n Backpropagation Notes (Stanford):
      https://cs231n.github.io/optimization-2/
    - Deep Learning Book, Chapter 6.5 (Goodfellow et al.):
      https://www.deeplearningbook.org/contents/mlp.html
    - Original paper: Rumelhart, Hinton & Williams (1986),
      "Learning representations by back-propagating errors":
      https://www.nature.com/articles/323533a0
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from agentexplorr.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# NODE CLASS — One operation in the computational graph
# =============================================================================


class Node:
    """Represents a single operation in a computational graph.

    Each Node holds:
        - value: the result of the forward computation (a scalar float or numpy array)
        - grad: the gradient of the FINAL output with respect to this node's value
                 (accumulated during the backward pass)
        - children: list of input Nodes that feed into this operation
        - op: a human-readable name for the operation (e.g., "mul", "add", "sigmoid")
        - label: optional human-readable name for this node (e.g., "w1", "x1", "loss")
        - _backward: a function that computes and propagates gradients to children

    THE KEY INSIGHT:
        During the forward pass, each Node computes its value from its children's values.
        During the backward pass, each Node receives its gradient (from its parent)
        and uses the LOCAL derivative to send gradients to its children.

        This is the chain rule in action: dL/dchild = dL/dself * dself/dchild.

    EXAMPLE:
        Suppose we compute c = a * b.  Then:
            - c.children = [a, b]
            - c.op = "mul"
            - Forward:  c.value = a.value * b.value
            - Backward: a.grad += c.grad * b.value   (dL/da = dL/dc * dc/da = dL/dc * b)
                        b.grad += c.grad * a.value   (dL/db = dL/dc * dc/db = dL/dc * a)

        The '+=' is important: a node might be used in multiple places.
        Gradients ACCUMULATE (multivariate chain rule).
    """

    def __init__(
        self,
        value: float | np.ndarray,
        children: Sequence[Node] = (),
        op: str = "",
        label: str = "",
    ) -> None:
        """Create a new Node in the computational graph.

        Args:
            value: The numeric value this node holds (result of forward computation).
                   Can be a scalar float or a numpy array for vector/matrix operations.
            children: Tuple of input Nodes that were used to compute this node's value.
                      Leaf nodes (inputs, weights) have no children.
            op: Name of the operation that produced this node (e.g., "add", "mul").
                Leaf nodes have op = "".
            label: Human-readable name for display purposes (e.g., "w1", "bias").
        """
        # Store value as float or numpy array
        self.value: float | np.ndarray = value

        # Gradient of the final loss with respect to this node.
        # Initialized to 0.0; will be filled in during backward().
        # For numpy arrays, the gradient has the SAME SHAPE as the value.
        if isinstance(value, np.ndarray):
            self.grad: float | np.ndarray = np.zeros_like(value, dtype=np.float64)
        else:
            self.grad = 0.0

        # The nodes that were inputs to this operation.
        # We store these so the backward pass can propagate gradients to them.
        self.children: tuple[Node, ...] = tuple(children)

        # Human-readable metadata for visualization
        self.op: str = op
        self.label: str = label

        # The backward function for THIS specific operation.
        # Each operation (add, mul, sigmoid, etc.) defines its own backward function
        # that knows how to compute the local gradient and propagate it.
        # Default: do nothing (for leaf nodes that have no children).
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        """String representation showing label, value, and gradient."""
        label_str = f"{self.label}=" if self.label else ""
        if isinstance(self.value, np.ndarray):
            return f"Node({label_str}array(shape={self.value.shape}), grad=...)"
        return f"Node({label_str}{self.value:.6f}, grad={self.grad:.6f})"

    # -----------------------------------------------------------------
    # ARITHMETIC OPERATIONS — Each one creates a new Node in the graph
    # -----------------------------------------------------------------

    def __add__(self, other: Node | float) -> Node:
        """Addition: c = a + b.

        MATH:
            Forward:  c = a + b
            Backward: dc/da = 1   (adding doesn't change the gradient)
                      dc/db = 1

            So: a.grad += c.grad * 1 = c.grad   (gradient passes through!)
                b.grad += c.grad * 1 = c.grad

        WHY THE GRADIENT PASSES THROUGH:
            Addition distributes the gradient equally to both inputs.
            If the output needs to increase by some amount dL, then BOTH
            inputs need to increase by that same amount (since a+b would
            increase by dL if either a or b increases by dL).
        """
        # Allow adding raw floats: wrap them in a Node
        other = other if isinstance(other, Node) else Node(value=other, label=str(other))

        # Forward: compute the sum
        out = Node(value=self.value + other.value, children=(self, other), op="+")

        # Backward: define how gradients flow through addition
        def _backward() -> None:
            # dL/d(self) += dL/d(out) * d(out)/d(self)  =  dL/d(out) * 1
            self.grad = self.grad + out.grad
            # dL/d(other) += dL/d(out) * d(out)/d(other)  =  dL/d(out) * 1
            other.grad = other.grad + out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float | Node) -> Node:
        """Handle float + Node (reversed addition)."""
        return self.__add__(other)

    def __mul__(self, other: Node | float) -> Node:
        """Multiplication: c = a * b.

        MATH:
            Forward:  c = a * b
            Backward: dc/da = b   (the OTHER input's value!)
                      dc/db = a

            So: a.grad += c.grad * b   (gradient is scaled by the other value)
                b.grad += c.grad * a

        WHY THE GRADIENT SWAPS:
            Think about it: if b is large, then a small change in a causes
            a LARGE change in the product (because c = a*b). So the gradient
            of c with respect to a is proportional to b. And vice versa.

        THIS IS PROFOUND FOR NEURAL NETWORKS:
            In a neuron, output = weight * input.
            - Gradient w.r.t weight = upstream_grad * input
              --> Weights learn faster when inputs are large
            - Gradient w.r.t input = upstream_grad * weight
              --> Inputs get larger gradients when weights are large
        """
        other = other if isinstance(other, Node) else Node(value=other, label=str(other))

        out = Node(value=self.value * other.value, children=(self, other), op="*")

        def _backward() -> None:
            # dL/d(self) += dL/d(out) * d(out)/d(self) = dL/d(out) * other.value
            self.grad = self.grad + out.grad * other.value
            # dL/d(other) += dL/d(out) * d(out)/d(other) = dL/d(out) * self.value
            other.grad = other.grad + out.grad * self.value

        out._backward = _backward
        return out

    def __rmul__(self, other: float | Node) -> Node:
        """Handle float * Node (reversed multiplication)."""
        return self.__mul__(other)

    def __neg__(self) -> Node:
        """Negation: -a = a * (-1)."""
        return self * (-1.0)

    def __sub__(self, other: Node | float) -> Node:
        """Subtraction: a - b = a + (-b)."""
        return self + (-other)

    def __rsub__(self, other: float) -> Node:
        """Handle float - Node."""
        return Node(value=other, label=str(other)) + (-self)

    def __pow__(self, exponent: float) -> Node:
        """Power: c = a^n.

        MATH:
            Forward:  c = a^n
            Backward: dc/da = n * a^(n-1)

            This is the standard power rule from calculus.
            For n=2 (squaring): dc/da = 2a
            For n=0.5 (sqrt):   dc/da = 0.5 * a^(-0.5) = 1/(2*sqrt(a))
        """
        assert isinstance(exponent, (int, float)), "Only constant exponents supported."

        out = Node(
            value=self.value ** exponent,
            children=(self,),
            op=f"**{exponent}",
        )

        def _backward() -> None:
            # dL/d(self) += dL/d(out) * n * self.value^(n-1)
            self.grad = self.grad + out.grad * exponent * (self.value ** (exponent - 1))

        out._backward = _backward
        return out

    # -----------------------------------------------------------------
    # ACTIVATION FUNCTIONS — Non-linear operations used in neural nets
    # -----------------------------------------------------------------

    def relu(self) -> Node:
        """ReLU (Rectified Linear Unit): c = max(0, a).

        MATH:
            Forward:  c = max(0, a)
                      = a  if a > 0
                      = 0  if a <= 0

            Backward: dc/da = 1 if a > 0
                      dc/da = 0 if a <= 0

        WHY RELU IS POPULAR:
            1. Gradient is either 0 or 1 — no shrinking! (helps with vanishing gradients)
            2. Very fast to compute (just a comparison)
            3. Induces sparsity (some neurons output 0, effectively "turning off")

        THE "DYING RELU" PROBLEM:
            If a neuron's input is always negative, its gradient is always 0,
            so it can never recover. The neuron is "dead" — it will never activate
            again. This is why variants like Leaky ReLU and GELU exist.
        """
        out_val = max(0.0, self.value) if isinstance(self.value, (int, float)) else np.maximum(0, self.value)
        out = Node(value=out_val, children=(self,), op="relu")

        def _backward() -> None:
            # Gradient is 1 where input > 0, else 0 (the "gate")
            if isinstance(self.value, np.ndarray):
                gate = (self.value > 0).astype(np.float64)
            else:
                gate = 1.0 if float(self.value) > 0 else 0.0  # type: ignore[assignment]
            self.grad = self.grad + out.grad * gate

        out._backward = _backward
        return out

    def sigmoid(self) -> Node:
        """Sigmoid activation: c = 1 / (1 + exp(-a)).

        MATH:
            Forward:  c = sigma(a) = 1 / (1 + e^(-a))

            Backward: dc/da = sigma(a) * (1 - sigma(a))
                            = c * (1 - c)

        DERIVING THE SIGMOID DERIVATIVE (step by step):
            Let s = 1 / (1 + e^(-a))

            ds/da = -1 / (1 + e^(-a))^2  *  d/da(1 + e^(-a))
                  = -1 / (1 + e^(-a))^2  *  (-e^(-a))
                  = e^(-a) / (1 + e^(-a))^2

            Now notice:  e^(-a) / (1 + e^(-a))^2
                       = [1 / (1 + e^(-a))] * [e^(-a) / (1 + e^(-a))]
                       = s * [(1 + e^(-a) - 1) / (1 + e^(-a))]
                       = s * [1 - 1/(1 + e^(-a))]
                       = s * (1 - s)

            Beautiful! The derivative of sigmoid is sigmoid * (1 - sigmoid).

        WHY SIGMOID CAUSES VANISHING GRADIENTS:
            The maximum value of s*(1-s) is 0.25 (when s = 0.5, i.e., a = 0).
            So each sigmoid layer SHRINKS the gradient by at least 4x!
            After 10 layers: 0.25^10 = 9.5e-7 (practically zero).
            This is why deep networks DON'T use sigmoid in hidden layers.
        """
        if isinstance(self.value, np.ndarray):
            # Numerically stable sigmoid using np.clip to prevent overflow
            clipped = np.clip(self.value, -500, 500)
            s = 1.0 / (1.0 + np.exp(-clipped))
        else:
            clipped = max(-500.0, min(500.0, float(self.value)))  # type: ignore[arg-type]
            s = 1.0 / (1.0 + math.exp(-clipped))

        out = Node(value=s, children=(self,), op="sigmoid")

        def _backward() -> None:
            # dL/d(self) += dL/d(out) * sigmoid(self) * (1 - sigmoid(self))
            self.grad = self.grad + out.grad * out.value * (1.0 - out.value)

        out._backward = _backward
        return out

    def tanh_act(self) -> Node:
        """Hyperbolic tangent activation: c = tanh(a).

        MATH:
            Forward:  c = tanh(a) = (e^a - e^(-a)) / (e^a + e^(-a))

            Backward: dc/da = 1 - tanh(a)^2 = 1 - c^2

        NOTE: tanh is a shifted and scaled sigmoid:
            tanh(a) = 2 * sigmoid(2a) - 1

            Its output range is [-1, 1] (vs sigmoid's [0, 1]).
            The gradient at a=0 is 1.0 (vs sigmoid's 0.25), so it suffers
            less from vanishing gradients than sigmoid — but still suffers
            compared to ReLU.
        """
        t = np.tanh(self.value) if isinstance(self.value, np.ndarray) else math.tanh(self.value)

        out = Node(value=t, children=(self,), op="tanh")

        def _backward() -> None:
            # dL/d(self) += dL/d(out) * (1 - tanh^2)
            self.grad = self.grad + out.grad * (1.0 - out.value ** 2)

        out._backward = _backward
        return out

    # -----------------------------------------------------------------
    # MATRIX / VECTOR OPERATIONS — For layers in neural networks
    # -----------------------------------------------------------------

    def matmul(self, other: Node) -> Node:
        """Matrix multiplication: C = A @ B.

        MATH:
            Forward:  C = A @ B   (matrix product)

            Backward (for loss L):
                dL/dA = dL/dC @ B^T
                dL/dB = A^T @ dL/dC

        WHY THESE FORMULAS?
            Consider C_ij = sum_k A_ik * B_kj

            dL/dA_ik = sum_j dL/dC_ij * dC_ij/dA_ik
                     = sum_j dL/dC_ij * B_kj
                     = (dL/dC @ B^T)_ik

            Similarly for dL/dB.

        SHAPES:
            If A is (m, n) and B is (n, p):
                C = A @ B         --> (m, p)
                dL/dA = dL/dC @ B^T  --> (m, p) @ (p, n) = (m, n)  (same as A!)
                dL/dB = A^T @ dL/dC  --> (n, m) @ (m, p) = (n, p)  (same as B!)

            The gradients always have the SAME SHAPE as the original matrix.
            This is a useful sanity check when debugging.

        FOR VECTOR INPUTS (1D):
            When dealing with 1D vectors (common for single-sample inference),
            we reshape to 2D, perform the matmul, then reshape back.
        """
        # Ensure we are working with numpy arrays
        a_val = np.atleast_2d(np.array(self.value, dtype=np.float64))
        b_val = np.atleast_2d(np.array(other.value, dtype=np.float64))

        out_val = a_val @ b_val

        out = Node(value=out_val, children=(self, other), op="matmul")

        def _backward() -> None:
            out_grad = np.atleast_2d(np.array(out.grad, dtype=np.float64))
            # dL/dA = dL/dC @ B^T
            grad_a = out_grad @ b_val.T
            # dL/dB = A^T @ dL/dC
            grad_b = a_val.T @ out_grad

            # Accumulate gradients, reshaping back if needed
            if isinstance(self.grad, np.ndarray):
                self.grad = self.grad + grad_a.reshape(self.grad.shape)
            else:
                self.grad = grad_a

            if isinstance(other.grad, np.ndarray):
                other.grad = other.grad + grad_b.reshape(other.grad.shape)
            else:
                other.grad = grad_b

        out._backward = _backward
        return out


# =============================================================================
# COMPUTATIONAL GRAPH CLASS — Orchestrates forward and backward passes
# =============================================================================


class ComputationalGraph:
    """Orchestrates forward and backward passes through a graph of Nodes.

    A ComputationalGraph ties together all the Nodes that make up a computation
    (e.g., a neural network layer or an entire network). It handles:

        1. FORWARD PASS: Computing values from inputs to outputs
        2. BACKWARD PASS: Computing gradients from outputs to inputs
        3. VISUALIZATION: Printing the graph structure with values and gradients

    HOW THE BACKWARD PASS WORKS:
        We need to process nodes in REVERSE TOPOLOGICAL ORDER — that is,
        we must compute a node's gradient BEFORE computing gradients for
        its children. This ensures that when a child's _backward() runs,
        all upstream gradients have already been accumulated.

        Algorithm:
            1. Start at the output node (usually the loss)
            2. Set output.grad = 1.0 (dL/dL = 1 — the loss's gradient w.r.t itself)
            3. Process nodes in reverse topological order
            4. For each node, call node._backward() to propagate gradients to children

    TOPOLOGICAL SORT (why it matters):
        In a DAG, topological order means: every node comes AFTER all nodes
        it depends on. Reversed, it means: every node comes BEFORE all nodes
        that depend on it. This is exactly what we need for the backward pass.

        Example:  a --> c --> e
                  b ---/  \
                           d --> f

        Topological order: a, b, c, d, e, f
        Reverse (for backward): f, e, d, c, b, a

        This ensures when we process c, we have ALREADY processed e and d
        (which contribute to c.grad).
    """

    def __init__(self, output_node: Node, label: str = "graph") -> None:
        """Create a computational graph rooted at the given output node.

        Args:
            output_node: The final node in the computation (e.g., the loss).
                         The backward pass will compute gradients starting from here.
            label: Human-readable name for this graph.
        """
        self.output = output_node
        self.label = label

        # Build the topological ordering of all nodes reachable from output.
        # We will traverse this in reverse during backward().
        self._topo_order: list[Node] = []
        self._build_topo()

        logger.info(
            "computational_graph_created",
            label=label,
            num_nodes=len(self._topo_order),
        )

    def _build_topo(self) -> None:
        """Build a topological ordering of all nodes via depth-first search.

        ALGORITHM (Kahn-style DFS post-order):
            1. Start from the output node
            2. Recursively visit all children first (DFS)
            3. After visiting all children, append the current node
            4. Result: nodes are in topological order (children before parents)

        We reverse this list for the backward pass so parents come first.
        """
        visited: set[int] = set()
        topo: list[Node] = []

        def _dfs(node: Node) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            # Visit all children first (these are the inputs to this node)
            for child in node.children:
                _dfs(child)
            # Post-order: add this node AFTER all its dependencies
            topo.append(node)

        _dfs(self.output)
        self._topo_order = topo

    def forward(self) -> float | np.ndarray:
        """Execute the forward pass and return the output value.

        NOTE: In our implementation, the forward pass is computed eagerly
        when Nodes are created (each operation computes its value immediately).
        This method exists for API clarity and to return the final value.

        In frameworks like PyTorch, the forward pass can be lazy (deferred
        until .backward() or .item() is called), enabling optimizations
        like operator fusion.

        Returns:
            The computed value at the output node.
        """
        return self.output.value

    def backward(self) -> None:
        """Execute the backward pass — compute all gradients via the chain rule.

        This is the heart of backpropagation. The algorithm:

            1. SEED: Set the output node's gradient to 1.0.
               Why? Because dL/dL = 1 (the derivative of L with respect to itself).

            2. REVERSE: Walk through nodes in reverse topological order.
               This ensures every node's gradient is FULLY computed before
               we use it to compute gradients for earlier nodes.

            3. PROPAGATE: For each node, call _backward() which uses the
               chain rule to send gradient contributions to its children.

        After backward() completes, every Node in the graph has its .grad
        attribute filled with the gradient of the output w.r.t. that node.
        """
        # Step 0: Reset all gradients to zero to avoid accumulation from
        # previous backward passes. This is important when running backward()
        # multiple times (e.g., during training iterations).
        for node in self._topo_order:
            if isinstance(node.value, np.ndarray):
                node.grad = np.zeros_like(node.value, dtype=np.float64)
            else:
                node.grad = 0.0

        # Step 1: Seed the output gradient
        # dL/dL = 1.0 — the loss's gradient with respect to itself is 1.
        if isinstance(self.output.value, np.ndarray):
            self.output.grad = np.ones_like(self.output.value, dtype=np.float64)
        else:
            self.output.grad = 1.0

        # Step 2 & 3: Walk in REVERSE topological order and propagate gradients.
        # Reverse order ensures that when we call node._backward(), the node's
        # grad has already been fully accumulated from all its consumers.
        for node in reversed(self._topo_order):
            node._backward()

        logger.debug(
            "backward_pass_complete",
            graph=self.label,
            num_nodes=len(self._topo_order),
        )

    def zero_grad(self) -> None:
        """Reset all gradients to zero.

        Call this before each backward pass during training to prevent
        gradient accumulation from previous iterations.

        WHY IS THIS NECESSARY?
            Our _backward() methods use += to accumulate gradients (to handle
            nodes used in multiple places). Without zeroing, gradients from
            the previous training step would be added to the current step's
            gradients, giving incorrect updates.

            PyTorch has the same requirement: you must call optimizer.zero_grad()
            before each backward pass.
        """
        for node in self._topo_order:
            if isinstance(node.value, np.ndarray):
                node.grad = np.zeros_like(node.value, dtype=np.float64)
            else:
                node.grad = 0.0

    def visualize(self) -> str:
        """Print the computational graph with values and gradients.

        Shows each node's:
            - Label and operation
            - Forward value
            - Gradient (after backward pass)
            - Children (inputs)

        Returns:
            A formatted string representation of the graph.
        """
        lines: list[str] = []
        lines.append(f"\n{'=' * 70}")
        lines.append(f"  COMPUTATIONAL GRAPH: {self.label}")
        lines.append(f"{'=' * 70}")
        lines.append(f"  {'Node':<20} {'Op':<10} {'Value':<18} {'Gradient':<18}")
        lines.append(f"  {'-' * 20} {'-' * 10} {'-' * 18} {'-' * 18}")

        for node in self._topo_order:
            name = node.label if node.label else f"node_{id(node) % 10000:04d}"

            # Format value
            if isinstance(node.value, np.ndarray):
                val_str = f"arr{node.value.shape}"
            else:
                val_str = f"{node.value:>12.6f}"

            # Format gradient
            if isinstance(node.grad, np.ndarray):
                grad_str = f"arr{node.grad.shape}"
            else:
                grad_str = f"{node.grad:>12.6f}"

            op_str = node.op if node.op else "input"
            lines.append(f"  {name:<20} {op_str:<10} {val_str:<18} {grad_str:<18}")

        lines.append(f"{'=' * 70}\n")

        result = "\n".join(lines)
        return result

    # -----------------------------------------------------------------
    # FACTORY METHODS — Build common network structures
    # -----------------------------------------------------------------

    @staticmethod
    def build_linear_layer(
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray,
        activation: str = "none",
    ) -> tuple[Node, list[Node]]:
        """Build a single linear layer: y = activation(W @ x + b).

        This is the fundamental building block of neural networks.

        MATH:
            z = W @ x + b        (linear transformation)
            y = activation(z)    (non-linear activation)

        WHERE:
            x: input vector   (shape: [input_dim])
            W: weight matrix  (shape: [output_dim, input_dim])
            b: bias vector    (shape: [output_dim])
            z: pre-activation (shape: [output_dim])
            y: output         (shape: [output_dim])

        WHY THE BIAS?
            Without bias, the hyperplane always passes through the origin.
            Bias allows the network to shift the decision boundary.

        WHY THE ACTIVATION?
            Without activation, stacking layers gives: W2 @ (W1 @ x) = (W2 @ W1) @ x
            which is just ANOTHER linear transformation. Activations introduce
            non-linearity, allowing the network to learn complex functions.

        Args:
            x: Input array of shape (input_dim,) or (input_dim, 1).
            w: Weight array of shape (output_dim, input_dim).
            b: Bias array of shape (output_dim,) or (output_dim, 1).
            activation: One of "none", "relu", "sigmoid", "tanh".

        Returns:
            Tuple of (output_node, list_of_parameter_nodes) where parameter_nodes
            contains the weight and bias Nodes (for accessing gradients).
        """
        # Create leaf nodes (inputs — these are the starting points of the graph)
        x_node = Node(value=np.array(x, dtype=np.float64).reshape(-1, 1), label="x")
        w_node = Node(value=np.array(w, dtype=np.float64), label="W")
        b_node = Node(value=np.array(b, dtype=np.float64).reshape(-1, 1), label="b")

        # Forward: z = W @ x + b
        # Step 1: Compute W @ x (matrix multiplication)
        wx_node = w_node.matmul(x_node)
        wx_node.label = "Wx"

        # Step 2: Add bias: z = Wx + b
        z_node = wx_node + b_node
        z_node.label = "z"

        # Step 3: Apply activation
        if activation == "relu":
            y_node = z_node.relu()
            y_node.label = "y=relu(z)"
        elif activation == "sigmoid":
            y_node = z_node.sigmoid()
            y_node.label = "y=sigmoid(z)"
        elif activation == "tanh":
            y_node = z_node.tanh_act()
            y_node.label = "y=tanh(z)"
        else:
            y_node = z_node
            y_node.label = "y=z"

        return y_node, [w_node, b_node, x_node]

    @staticmethod
    def build_simple_network(
        x: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        hidden_activation: str = "relu",
    ) -> tuple[Node, list[Node]]:
        """Build a 2-layer neural network: y = W2 @ activation(W1 @ x + b1) + b2.

        ARCHITECTURE:
            Input (x)  -->  Linear Layer 1  -->  Activation  -->  Linear Layer 2  -->  Output (y)
              [n_in]       [n_hidden, n_in]     [n_hidden]      [n_out, n_hidden]     [n_out]

        This is the simplest "deep" network. With just 2 layers and a non-linear
        activation, it can approximate ANY continuous function (Universal
        Approximation Theorem). In practice, deeper networks learn more
        efficiently, but this 2-layer network illustrates all the key concepts.

        Args:
            x: Input vector of shape (input_dim,).
            w1: Layer 1 weights, shape (hidden_dim, input_dim).
            b1: Layer 1 biases, shape (hidden_dim,).
            w2: Layer 2 weights, shape (output_dim, hidden_dim).
            b2: Layer 2 biases, shape (output_dim,).
            hidden_activation: Activation for the hidden layer ("relu", "sigmoid", "tanh").

        Returns:
            Tuple of (output_node, list_of_all_parameter_nodes).
        """
        # -- Layer 1: hidden = activation(W1 @ x + b1) --
        x_node = Node(value=np.array(x, dtype=np.float64).reshape(-1, 1), label="x")
        w1_node = Node(value=np.array(w1, dtype=np.float64), label="W1")
        b1_node = Node(value=np.array(b1, dtype=np.float64).reshape(-1, 1), label="b1")

        # z1 = W1 @ x + b1
        w1x = w1_node.matmul(x_node)
        w1x.label = "W1x"
        z1 = w1x + b1_node
        z1.label = "z1"

        # Apply hidden activation
        if hidden_activation == "relu":
            h = z1.relu()
        elif hidden_activation == "sigmoid":
            h = z1.sigmoid()
        elif hidden_activation == "tanh":
            h = z1.tanh_act()
        else:
            h = z1
        h.label = "h"

        # -- Layer 2: y = W2 @ h + b2 (no activation for output layer) --
        w2_node = Node(value=np.array(w2, dtype=np.float64), label="W2")
        b2_node = Node(value=np.array(b2, dtype=np.float64).reshape(-1, 1), label="b2")

        w2h = w2_node.matmul(h)
        w2h.label = "W2h"
        y = w2h + b2_node
        y.label = "y"

        return y, [w1_node, b1_node, w2_node, b2_node, x_node]


# =============================================================================
# HELPER: Numerical gradient checking
# =============================================================================


def numerical_gradient(
    f: Callable[..., float],
    args: list[float],
    arg_index: int,
    epsilon: float = 1e-7,
) -> float:
    """Compute the gradient of f with respect to args[arg_index] using finite differences.

    MATH:
        df/dx approx= [f(x + epsilon) - f(x - epsilon)] / (2 * epsilon)

    This is the "central difference" formula. It has O(epsilon^2) error,
    which is much better than the forward difference [f(x+eps) - f(x)] / eps
    that only has O(epsilon) error.

    WHY IS THIS USEFUL?
        Numerical gradients are SLOW (need 2 function evaluations per parameter)
        but EASY TO IMPLEMENT CORRECTLY. We use them to VERIFY that our
        analytical gradients (computed by backprop) are correct. This is
        called "gradient checking" and is essential when implementing
        backpropagation from scratch.

    Args:
        f: A function that takes *args and returns a scalar float.
        args: The list of arguments to f.
        arg_index: Which argument to differentiate with respect to.
        epsilon: Step size for finite differences (default 1e-7).

    Returns:
        The numerical gradient df/d(args[arg_index]).
    """
    args_plus = list(args)
    args_minus = list(args)

    args_plus[arg_index] = args[arg_index] + epsilon
    args_minus[arg_index] = args[arg_index] - epsilon

    return (f(*args_plus) - f(*args_minus)) / (2.0 * epsilon)


# =============================================================================
# DEMO: If run directly, show worked examples with full explanations
# =============================================================================

if __name__ == "__main__":
    # We use plain print() for the demo to avoid structlog formatting overhead.
    # In production code, you would use the logger.

    print("\n" + "=" * 72)
    print("   BACKPROPAGATION FROM SCRATCH — Interactive Demo")
    print("   Understanding the chain rule through computational graphs")
    print("=" * 72)

    # =================================================================
    # EXAMPLE 1: Simple Chain Rule
    # f(x) = (2x + 3)^2
    # =================================================================

    print("\n" + "-" * 72)
    print("  EXAMPLE 1: Simple Chain Rule — f(x) = (2x + 3)^2")
    print("-" * 72)
    print("""
    We decompose f(x) = (2x + 3)^2 into elementary operations:

        a = 2 * x       (multiplication)
        b = a + 3       (addition)
        f = b ** 2      (squaring)

    Computational graph:

        x --[*2]--> a --[+3]--> b --[**2]--> f

    Let's trace through with x = 1:
    """)

    # Build the computational graph
    x = Node(value=1.0, label="x")

    # Step 1: a = 2 * x
    a = x * 2.0
    a.label = "a=2x"

    # Step 2: b = a + 3
    b = a + 3.0
    b.label = "b=a+3"

    # Step 3: f = b^2
    f = b ** 2.0
    f.label = "f=b^2"

    print("    FORWARD PASS (computing values left to right):")
    print(f"      x = {x.value:.1f}")
    print(f"      a = 2 * x = 2 * {x.value:.1f} = {a.value:.1f}")
    print(f"      b = a + 3 = {a.value:.1f} + 3 = {b.value:.1f}")
    print(f"      f = b^2  = {b.value:.1f}^2 = {f.value:.1f}")

    # Run backward pass
    graph = ComputationalGraph(output_node=f, label="f(x) = (2x+3)^2")
    graph.backward()

    print("\n    BACKWARD PASS (computing gradients right to left):")
    print("      df/df = 1.0                           (seed)")
    print(f"      df/db = 2 * b = 2 * {b.value:.1f} = {f.grad * 2 * b.value / (2 * b.value):.1f}  x  df/df = {b.grad:.1f}")
    print(f"      df/da = df/db * db/da = {b.grad:.1f} * 1 = {a.grad:.1f}    (addition passes gradient through)")
    print(f"      df/dx = df/da * da/dx = {a.grad:.1f} * 2 = {x.grad:.1f}   (multiplication by constant 2)")

    print(f"\n    RESULT: df/dx at x=1 is {x.grad:.1f}")

    # Verify analytically
    # f(x) = (2x+3)^2 = 4x^2 + 12x + 9
    # f'(x) = 8x + 12
    # f'(1) = 20
    analytical = 8.0 * 1.0 + 12.0
    print("\n    VERIFICATION (analytical):")
    print("      f(x) = (2x+3)^2 = 4x^2 + 12x + 9")
    print("      f'(x) = 8x + 12")
    print(f"      f'(1) = 8(1) + 12 = {analytical:.1f}")
    print(f"      Match: {abs(float(x.grad) - analytical) < 1e-10}")

    # Also verify with numerical gradient
    def f_func(x_val: float) -> float:
        return (2 * x_val + 3) ** 2

    num_grad = numerical_gradient(f_func, [1.0], 0)
    print(f"\n    VERIFICATION (numerical gradient): {num_grad:.6f}")
    print(f"    Backprop gradient:                  {x.grad:.6f}")
    print(f"    Difference:                         {abs(x.grad - num_grad):.2e}")

    print(graph.visualize())

    # =================================================================
    # EXAMPLE 2: Single Neuron
    # y = sigmoid(w1*x1 + w2*x2 + b)
    # =================================================================

    print("\n" + "-" * 72)
    print("  EXAMPLE 2: A Single Neuron — y = sigmoid(w1*x1 + w2*x2 + b)")
    print("-" * 72)
    print("""
    This is the building block of ALL neural networks.
    A single neuron computes a weighted sum of inputs, adds a bias,
    and passes the result through a non-linear activation function.

    Computational graph:

        x1 ---[*w1]---+
                       |
                      [+]----> s --[+b]--> z --[sigmoid]--> y
                       |
        x2 ---[*w2]---+

    where:
        s  = w1*x1 + w2*x2      (weighted sum)
        z  = s + b               (add bias)
        y  = sigmoid(z)          (activation)

    We'll use: x1=0.5, x2=-1.0, w1=2.0, w2=-1.0, b=0.5
    """)

    # Create input and parameter nodes
    x1 = Node(value=0.5, label="x1")
    x2 = Node(value=-1.0, label="x2")
    w1 = Node(value=2.0, label="w1")
    w2 = Node(value=-1.0, label="w2")
    bias = Node(value=0.5, label="b")

    # Forward pass: build the graph step by step
    # Step 1: weighted inputs
    w1x1 = w1 * x1
    w1x1.label = "w1*x1"

    w2x2 = w2 * x2
    w2x2.label = "w2*x2"

    # Step 2: sum of weighted inputs
    s = w1x1 + w2x2
    s.label = "s=w1x1+w2x2"

    # Step 3: add bias
    z = s + bias
    z.label = "z=s+b"

    # Step 4: sigmoid activation
    y = z.sigmoid()
    y.label = "y=sigmoid(z)"

    print("    FORWARD PASS:")
    print(f"      w1*x1 = {w1.value:.1f} * {x1.value:.1f}   = {w1x1.value:.4f}")
    print(f"      w2*x2 = {w2.value:.1f} * {x2.value:.1f} = {w2x2.value:.4f}")
    print(f"      s     = {w1x1.value:.4f} + {w2x2.value:.4f} = {s.value:.4f}")
    print(f"      z     = {s.value:.4f} + {bias.value:.1f}   = {z.value:.4f}")
    print(f"      y     = sigmoid({z.value:.4f}) = {y.value:.6f}")

    # Backward pass
    neuron_graph = ComputationalGraph(output_node=y, label="single neuron")
    neuron_graph.backward()

    print("\n    BACKWARD PASS (all partial derivatives):")
    print("      dy/dy    = 1.0                       (seed)")
    sig_val = y.value
    sig_deriv = sig_val * (1.0 - sig_val)
    print(f"      dy/dz    = y*(1-y) = {sig_val:.6f} * {1 - sig_val:.6f} = {z.grad:.6f}")
    print(f"      dy/ds    = dy/dz * dz/ds = {z.grad:.6f} * 1 = {s.grad:.6f}")
    print(f"      dy/db    = dy/dz * dz/db = {z.grad:.6f} * 1 = {bias.grad:.6f}")
    print(f"      dy/d(w1x1) = dy/ds * 1 = {w1x1.grad:.6f}")
    print(f"      dy/dw1   = dy/d(w1x1) * x1 = {w1x1.grad:.6f} * {x1.value:.1f} = {w1.grad:.6f}")
    print(f"      dy/dx1   = dy/d(w1x1) * w1 = {w1x1.grad:.6f} * {w1.value:.1f} = {x1.grad:.6f}")
    print(f"      dy/dw2   = dy/d(w2x2) * x2 = {w2x2.grad:.6f} * {x2.value:.1f} = {w2.grad:.6f}")
    print(f"      dy/dx2   = dy/d(w2x2) * w2 = {w2x2.grad:.6f} * {w2.value:.1f} = {x2.grad:.6f}")

    # Verify with numerical gradients
    print("\n    GRADIENT VERIFICATION (numerical vs. backprop):")

    def neuron_func(w1_v: float, w2_v: float, b_v: float, x1_v: float, x2_v: float) -> float:
        z_val = w1_v * x1_v + w2_v * x2_v + b_v
        return 1.0 / (1.0 + math.exp(-z_val))

    param_names = ["w1", "w2", "b", "x1", "x2"]
    param_vals = [2.0, -1.0, 0.5, 0.5, -1.0]
    backprop_grads = [w1.grad, w2.grad, bias.grad, x1.grad, x2.grad]

    for i, (name, bp_grad) in enumerate(zip(param_names, backprop_grads, strict=False)):
        num_g = numerical_gradient(neuron_func, param_vals, i)
        match = abs(float(bp_grad) - num_g) < 1e-5
        print(f"      d/d{name:>2s}: backprop={bp_grad:>10.6f}  numerical={num_g:>10.6f}  match={match}")

    print(neuron_graph.visualize())

    # =================================================================
    # EXAMPLE 3: Verify with PyTorch Autograd
    # =================================================================

    print("\n" + "-" * 72)
    print("  EXAMPLE 3: Verify Against PyTorch Autograd")
    print("-" * 72)
    print("""
    The gold standard for gradient verification is to compare against
    PyTorch's autograd engine. PyTorch computes exact analytical gradients
    using the same chain rule approach, but with a battle-tested
    implementation. If our gradients match, our backprop is correct!
    """)

    try:
        import torch

        print("    PyTorch is available. Running verification...\n")

        # ---- Test 1: f(x) = (2x + 3)^2 at x=1 ----
        print("    TEST 1: f(x) = (2x + 3)^2 at x=1")

        x_t = torch.tensor(1.0, requires_grad=True)
        f_t = (2.0 * x_t + 3.0) ** 2
        f_t.backward()

        our_grad = 20.0  # From Example 1 above
        torch_grad = x_t.grad.item()
        print(f"      Our gradient:     {our_grad:.6f}")
        print(f"      PyTorch gradient: {torch_grad:.6f}")
        print(f"      Match: {abs(our_grad - torch_grad) < 1e-6}")

        # ---- Test 2: Single neuron ----
        print("\n    TEST 2: y = sigmoid(w1*x1 + w2*x2 + b)")

        w1_t = torch.tensor(2.0, requires_grad=True)
        w2_t = torch.tensor(-1.0, requires_grad=True)
        b_t = torch.tensor(0.5, requires_grad=True)
        x1_t = torch.tensor(0.5, requires_grad=True)
        x2_t = torch.tensor(-1.0, requires_grad=True)

        y_t = torch.sigmoid(w1_t * x1_t + w2_t * x2_t + b_t)
        y_t.backward()

        print(f"      {'Parameter':<8} {'Ours':>12} {'PyTorch':>12} {'Match':>8}")
        print(f"      {'-'*8} {'-'*12} {'-'*12} {'-'*8}")

        torch_grads = {
            "w1": w1_t.grad.item(),
            "w2": w2_t.grad.item(),
            "b": b_t.grad.item(),
            "x1": x1_t.grad.item(),
            "x2": x2_t.grad.item(),
        }

        our_grads = {
            "w1": w1.grad,
            "w2": w2.grad,
            "b": bias.grad,
            "x1": x1.grad,
            "x2": x2.grad,
        }

        all_match = True
        for name in ["w1", "w2", "b", "x1", "x2"]:
            ours = our_grads[name]
            theirs = torch_grads[name]
            match = abs(ours - theirs) < 1e-5
            all_match = all_match and match
            print(f"      {name:<8} {ours:>12.6f} {theirs:>12.6f} {match!s:>8}")

        print(f"\n    ALL GRADIENTS MATCH: {all_match}")

        # ---- Test 3: 2-layer network ----
        print("\n    TEST 3: 2-layer network with ReLU hidden layer")

        np.random.seed(42)
        x_np = np.array([1.0, 2.0])
        w1_np = np.array([[0.5, -0.3], [0.2, 0.8], [-0.1, 0.4]])  # (3, 2)
        b1_np = np.array([0.1, -0.2, 0.3])                          # (3,)
        w2_np = np.array([[0.6, -0.5, 0.3]])                         # (1, 3)
        b2_np = np.array([0.1])                                      # (1,)

        # Our implementation
        y_node, params = ComputationalGraph.build_simple_network(
            x=x_np, w1=w1_np, b1=b1_np, w2=w2_np, b2=b2_np, hidden_activation="relu"
        )
        net_graph = ComputationalGraph(output_node=y_node, label="2-layer network")
        net_graph.backward()

        # PyTorch implementation
        # NOTE: We must create tensors already in the right shape BEFORE
        # setting requires_grad=True, because reshape() on a requires_grad
        # tensor creates a non-leaf tensor whose .grad would be None.
        x_pt = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float64)
        w1_pt = torch.tensor(w1_np, dtype=torch.float64, requires_grad=True)
        b1_pt = torch.tensor(b1_np.reshape(-1, 1), dtype=torch.float64, requires_grad=True)
        w2_pt = torch.tensor(w2_np, dtype=torch.float64, requires_grad=True)
        b2_pt = torch.tensor(b2_np.reshape(-1, 1), dtype=torch.float64, requires_grad=True)

        z1_pt = w1_pt @ x_pt + b1_pt
        h_pt = torch.relu(z1_pt)
        y_pt = w2_pt @ h_pt + b2_pt
        y_pt.sum().backward()

        # Compare outputs
        our_output = y_node.value.flatten()
        torch_output = y_pt.detach().numpy().flatten()
        print("\n      Output values:")
        print(f"        Ours:    {our_output}")
        print(f"        PyTorch: {torch_output}")
        print(f"        Match:   {np.allclose(our_output, torch_output, atol=1e-6)}")

        # Compare W1 gradients
        # params = [w1_node, b1_node, w2_node, b2_node, x_node]
        w1_node_ref = params[0]
        b1_node_ref = params[1]
        w2_node_ref = params[2]
        b2_node_ref = params[3]

        print("\n      W1 gradients:")
        print(f"        Ours:    \n{w1_node_ref.grad}")
        print(f"        PyTorch: \n{w1_pt.grad.numpy()}")
        print(f"        Match:   {np.allclose(w1_node_ref.grad, w1_pt.grad.numpy(), atol=1e-6)}")

        print("\n      W2 gradients:")
        print(f"        Ours:    {w2_node_ref.grad}")
        print(f"        PyTorch: {w2_pt.grad.numpy()}")
        print(f"        Match:   {np.allclose(w2_node_ref.grad, w2_pt.grad.numpy(), atol=1e-6)}")

        print("\n      b1 gradients:")
        print(f"        Ours:    {b1_node_ref.grad.flatten()}")
        print(f"        PyTorch: {b1_pt.grad.numpy().flatten()}")
        print(f"        Match:   {np.allclose(b1_node_ref.grad, b1_pt.grad.numpy(), atol=1e-6)}")

        print("\n      b2 gradients:")
        print(f"        Ours:    {b2_node_ref.grad.flatten()}")
        print(f"        PyTorch: {b2_pt.grad.numpy().flatten()}")
        print(f"        Match:   {np.allclose(b2_node_ref.grad, b2_pt.grad.numpy(), atol=1e-6)}")

    except ImportError:
        print("    PyTorch is not installed. Skipping PyTorch verification.")
        print("    Install with: pip install torch")
        print()
        print("    We already verified correctness using numerical gradients above.")
        print("    Numerical gradient checking is the standard approach when")
        print("    a reference implementation is unavailable.")

    # =================================================================
    # BONUS: Demonstrating Vanishing Gradients
    # =================================================================

    print("\n" + "-" * 72)
    print("  BONUS: Vanishing Gradients in Deep Sigmoid Networks")
    print("-" * 72)
    print("""
    Let's see vanishing gradients in action. We'll chain multiple sigmoid
    activations and watch the gradient shrink exponentially.

    Remember: sigmoid's max gradient is 0.25 (at input = 0).
    After N layers: gradient <= 0.25^N
    """)

    print(f"    {'Depth':<8} {'Gradient at input':<25} {'Ratio to depth 1'}")
    print(f"    {'-'*8} {'-'*25} {'-'*20}")

    first_grad = None
    for depth in [1, 2, 5, 10, 20, 50]:
        # Chain: sigmoid(sigmoid(sigmoid(...sigmoid(x)...)))
        node = Node(value=0.0, label="x")  # x=0 gives max sigmoid gradient
        for _i in range(depth):
            node = node.sigmoid()
        node.label = f"depth_{depth}"

        g = ComputationalGraph(output_node=node, label=f"depth_{depth}")
        g.backward()

        # Find the input node's gradient (it's at the beginning of topo order)
        input_grad = g._topo_order[0].grad
        if first_grad is None:
            first_grad = input_grad

        ratio = input_grad / first_grad if first_grad != 0 else 0.0
        print(f"    {depth:<8} {input_grad:<25.15f} {ratio:<20.10f}")

    print("""
    OBSERVATION: The gradient shrinks EXPONENTIALLY with depth!
    At depth 50, the gradient is essentially zero (< 1e-15).
    This is why deep networks with sigmoid activations cannot train
    their early layers — the gradient signal vanishes before it reaches them.

    SOLUTIONS:
      1. Use ReLU activations (gradient is 0 or 1, never shrinks)
      2. Use residual connections (skip connections add the gradient directly)
      3. Use batch normalization (keeps activations in a good range)
      4. Careful weight initialization (Xavier/He initialization)
    """)

    print("=" * 72)
    print("   KEY TAKEAWAY")
    print("=" * 72)
    print("""
    Backpropagation is the CHAIN RULE applied systematically through
    a computational graph:

      1. FORWARD: Compute values from inputs to output
      2. BACKWARD: Compute gradients from output to inputs

    Each node only needs to know its LOCAL derivative.
    The chain rule MULTIPLIES these together to get the global gradient.

    That's it. No magic. Just calculus.
    """)
