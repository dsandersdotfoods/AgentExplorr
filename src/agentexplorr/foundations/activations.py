"""
Activation Functions from Scratch — Why Neural Networks Need Non-Linearity
===========================================================================

WHY ACTIVATION FUNCTIONS EXIST:

    Without activation functions, a neural network is just ONE BIG LINEAR
    TRANSFORMATION — no matter how many layers you stack.

    PROOF (composing two linear layers):
        Layer 1:  h = W1 * x + b1
        Layer 2:  y = W2 * h + b2

        Substitute:
            y = W2 * (W1 * x + b1) + b2
            y = (W2 * W1) * x + (W2 * b1 + b2)
            y = W' * x + b'

        Where W' = W2 * W1 and b' = W2 * b1 + b2.

        Result: Two linear layers collapse into ONE linear layer!
        A 100-layer linear network is equivalent to a single matrix multiply.
        This means a deep linear network cannot learn XOR, cannot learn circles,
        cannot learn ANYTHING that a single-layer network cannot.

    THE FIX: Insert a non-linear function (activation) between layers:
        h = activation(W1 * x + b1)
        y = W2 * h + b2

        Now the composition is NO LONGER linear, and the network can learn
        arbitrarily complex functions.

    UNIVERSAL APPROXIMATION THEOREM (Cybenko, 1989; Hornik, 1991):
        A feed-forward network with a SINGLE hidden layer and a non-linear
        activation function can approximate ANY continuous function on a
        compact subset of R^n, given enough neurons.

        In plain English: with non-linearity, even a one-hidden-layer network
        can learn ANY function (in theory). In practice, deeper networks learn
        more efficiently — but the non-linearity is what makes it POSSIBLE.

WHAT THIS MODULE COVERS:
    1. ReLU          — The workhorse (simple, fast, used almost everywhere)
    2. Leaky ReLU    — Fixes dying ReLU (small gradient for negatives)
    3. Sigmoid        — The original (squishes to [0, 1], used for probabilities)
    4. Tanh           — Zero-centered sigmoid (better gradients, still saturates)
    5. GELU           — The transformer standard (smooth, probabilistic)
    6. Softmax        — Turns logits into probability distributions

HISTORICAL TIMELINE:
    1943  McCulloch-Pitts neuron (step function)
    1986  Backpropagation popularized (Rumelhart et al.) — sigmoid era begins
    1991  Universal Approximation Theorem proved (Hornik)
    2010  ReLU shown to work well for deep networks (Nair & Hinton)
    2011  ReLU widely adopted (Glorot et al.)
    2015  Batch Normalization + ReLU becomes standard (Ioffe & Szegedy)
    2016  GELU proposed (Hendrycks & Gimpel)
    2017  Transformers use GELU (Vaswani et al.)
    2018  Swish/SiLU explored (Ramachandran et al.)
    2020+ GELU dominant in LLMs (GPT, BERT, LLaMA)

LEARNING RESOURCES:
    - CS231n Activation Functions: https://cs231n.github.io/neural-networks-1/#actfun
    - Deep Learning Book Ch. 6: https://www.deeplearningbook.org/contents/mlp.html
    - VIDEO: "Activation Functions Explained" (StatQuest) — https://www.youtube.com/watch?v=68BZ5f7P94Q
    - VIDEO: "Why ReLU?" (3Blue1Brown, Neural Networks series) — https://www.youtube.com/watch?v=aircAruvnKk
    - Original ReLU paper: Glorot et al., 2011, "Deep Sparse Rectifier Neural Networks"
    - GELU paper: Hendrycks & Gimpel, 2016, https://arxiv.org/abs/1606.08415
    - Softmax and temperature: https://en.wikipedia.org/wiki/Softmax_function
    - Universal Approximation: Hornik, 1991, "Approximation capabilities of multilayer feedforward networks"
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from agentexplorr.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# BASE CLASS: ActivationFunction
# =============================================================================

class ActivationFunction(ABC):
    """Abstract base class for all activation functions.

    Every activation function needs two things:
        1. forward(x)  — Compute the activation value f(x)
        2. backward(x) — Compute the derivative f'(x) for backpropagation

    WHY WE NEED THE DERIVATIVE:
        During backpropagation, gradients flow backward through the network
        via the chain rule. At each activation layer, the incoming gradient
        is multiplied by the LOCAL derivative of the activation function:

            dL/dx = dL/dy * dy/dx = upstream_gradient * activation_derivative

        If the derivative is very small (e.g., sigmoid at extremes), the
        gradient "vanishes" — this is the VANISHING GRADIENT PROBLEM.
        If the derivative is zero (e.g., ReLU for x < 0), the gradient
        is completely blocked — this is the DYING RELU PROBLEM.
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the activation function value.

        Args:
            x: Input array of any shape (pre-activation values, aka logits).

        Returns:
            Activated values, same shape as x.
        """

    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Compute the derivative of the activation function.

        This is the LOCAL gradient used during backpropagation.
        The chain rule multiplies this by the upstream gradient.

        Args:
            x: Input array (same x used in forward, NOT the activated value).

        Returns:
            Derivative values, same shape as x.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the activation function."""


# =============================================================================
# 1. ReLU — Rectified Linear Unit (The Workhorse)
# =============================================================================

class ReLU(ActivationFunction):
    """Rectified Linear Unit — the most widely used activation function.

    FORMULA:
        f(x) = max(0, x)

        That's it. If x is positive, pass it through. If negative, output zero.

    DERIVATIVE:
        f'(x) = 1   if x > 0
        f'(x) = 0   if x < 0
        f'(x) is undefined at x = 0 (we use 0 by convention)

    WHY RELU WORKS SO WELL:
        1. COMPUTATIONAL SIMPLICITY: Just a comparison and a max — no exp()!
           Orders of magnitude faster than sigmoid or tanh.
        2. NO VANISHING GRADIENT (for x > 0): derivative is exactly 1, so
           gradients flow through unchanged. Compare to sigmoid where the
           max derivative is only 0.25.
        3. SPARSITY: Roughly 50% of neurons output zero for random inputs.
           Sparse representations are more efficient and often more
           interpretable (like how only some neurons "fire" in the brain).

    THE DYING RELU PROBLEM:
        If a neuron's input is always negative (due to bad initialization or
        a large learning rate), the gradient is ALWAYS ZERO. The neuron
        never updates and is effectively dead — it never activates again.

        In practice: ~10-40% of neurons can die during training.
        Fixes: Leaky ReLU, parametric ReLU, careful initialization (He init).

    PAPER: Glorot, X., Bordes, A., & Bengio, Y. (2011).
           "Deep Sparse Rectifier Neural Networks." AISTATS.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x) = max(0, x)"""
        return np.maximum(0, x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """f'(x) = 1 if x > 0, else 0"""
        return (x > 0).astype(x.dtype)

    @property
    def name(self) -> str:
        return "ReLU"


# =============================================================================
# 2. Leaky ReLU — Fixing the Dying ReLU Problem
# =============================================================================

class LeakyReLU(ActivationFunction):
    """Leaky ReLU — allows a small gradient when x < 0.

    FORMULA:
        f(x) = x        if x > 0
        f(x) = alpha * x if x <= 0

        Where alpha is a small positive constant (default 0.01).

    DERIVATIVE:
        f'(x) = 1      if x > 0
        f'(x) = alpha   if x <= 0

    WHY THIS EXISTS:
        Standard ReLU has ZERO gradient for negative inputs. Once a neuron
        outputs negative values consistently, it is "dead" — no gradient
        flows through, so it can never recover.

        Leaky ReLU fixes this by allowing a SMALL gradient (alpha) for
        negative inputs. The neuron is never fully dead.

    CHOOSING ALPHA:
        - alpha = 0.01: Standard Leaky ReLU (most common)
        - alpha = 0.2:  Aggressive leak (used in some GANs)
        - alpha = learned: Parametric ReLU (PReLU) — He et al., 2015

    PAPER: Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013).
           "Rectifier Nonlinearities Improve Neural Network Acoustic Models."

    Args:
        alpha: Slope for negative inputs. Default 0.01.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x) = x if x > 0, else alpha * x"""
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """f'(x) = 1 if x > 0, else alpha"""
        return np.where(x > 0, 1.0, self.alpha)

    @property
    def name(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"


# =============================================================================
# 3. Sigmoid — The Original Activation (Squishes to [0, 1])
# =============================================================================

class Sigmoid(ActivationFunction):
    """Sigmoid — maps any real number to (0, 1).

    FORMULA:
        sigma(x) = 1 / (1 + exp(-x))

    DERIVATIVE (beautifully self-referential):
        sigma'(x) = sigma(x) * (1 - sigma(x))

        DERIVATION:
            sigma(x)  = (1 + exp(-x))^(-1)
            sigma'(x) = -1 * (1 + exp(-x))^(-2) * (-exp(-x))     [chain rule]
                       = exp(-x) / (1 + exp(-x))^2
                       = [1 / (1 + exp(-x))] * [exp(-x) / (1 + exp(-x))]
                       = sigma(x) * [1 - 1/(1 + exp(-x))]
                       = sigma(x) * [1 - sigma(x)]

        This means the derivative only depends on the OUTPUT, not the input!
        Very efficient to compute during backpropagation.

    WHY SIGMOID FELL OUT OF FAVOR FOR HIDDEN LAYERS:

        1. VANISHING GRADIENT:
           The maximum value of sigma'(x) is 0.25 (at x = 0).
           For a 10-layer network, gradients shrink by ~0.25^10 = 9.5e-7.
           That's 0.000095% of the original gradient — effectively ZERO.
           The early layers learn almost nothing. This is why deep sigmoid
           networks were considered impossible to train before ~2010.

        2. NOT ZERO-CENTERED:
           Sigmoid outputs are always POSITIVE (between 0 and 1).
           This means the gradients for the weights are always the same
           sign, causing "zigzag" gradient updates that slow convergence.

           Why? If h = sigmoid(x) is always positive, then dL/dW = dL/dh * x,
           and the gradient direction for W is constrained.

        3. COMPUTATIONAL COST:
           exp() is much slower than max() (ReLU). In large networks with
           millions of activations per forward pass, this adds up.

    WHERE SIGMOID IS STILL USED:
        - OUTPUT layer for BINARY CLASSIFICATION (probability of class 1)
        - GATES in LSTMs and GRUs (controlling information flow)
        - ATTENTION mechanisms (sometimes)
        - Anywhere you need a [0, 1] probability interpretation

    NUMERICAL STABILITY:
        For very large negative x, exp(-x) overflows. We handle this using:
            For x >= 0:  sigma(x) = 1 / (1 + exp(-x))
            For x < 0:   sigma(x) = exp(x) / (1 + exp(x))
        numpy's implementation handles this internally with np.clip.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """sigma(x) = 1 / (1 + exp(-x)), numerically stable."""
        # Clip to prevent overflow in exp()
        x_safe = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_safe))

    def backward(self, x: np.ndarray) -> np.ndarray:
        """sigma'(x) = sigma(x) * (1 - sigma(x))"""
        s = self.forward(x)
        return s * (1.0 - s)

    @property
    def name(self) -> str:
        return "Sigmoid"


# =============================================================================
# 4. Tanh — Zero-Centered Sigmoid
# =============================================================================

class Tanh(ActivationFunction):
    """Hyperbolic Tangent — maps to (-1, 1), zero-centered.

    FORMULA:
        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    RELATIONSHIP TO SIGMOID:
        tanh(x) = 2 * sigmoid(2x) - 1

        This means tanh is just a rescaled, recentered sigmoid!
        sigmoid maps to (0, 1); tanh maps to (-1, 1).

    DERIVATIVE:
        tanh'(x) = 1 - tanh(x)^2

        DERIVATION:
            Let t = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

            Using quotient rule:
                d/dx [(e^x - e^(-x)) / (e^x + e^(-x))]
                = [(e^x + e^(-x))(e^x + e^(-x)) - (e^x - e^(-x))(e^x - e^(-x))]
                  / (e^x + e^(-x))^2
                = [(e^x + e^(-x))^2 - (e^x - e^(-x))^2] / (e^x + e^(-x))^2
                = 1 - [(e^x - e^(-x)) / (e^x + e^(-x))]^2
                = 1 - tanh(x)^2

    ADVANTAGES OVER SIGMOID:
        1. ZERO-CENTERED: outputs range from -1 to 1, centered at 0.
           This means gradients are not all-positive, avoiding the zigzag
           update problem that sigmoid has.
        2. STRONGER GRADIENTS: max derivative is 1.0 (at x = 0) vs
           sigmoid's 0.25. Gradients are 4x stronger at the peak.

    DISADVANTAGES:
        - Still suffers from VANISHING GRADIENTS at the extremes.
          For |x| > 3, tanh'(x) is very close to 0.
        - Still uses exp() (slower than ReLU's max()).

    COMMON USES:
        - Hidden layers in RNNs and LSTMs (before ReLU became standard)
        - Output layer when you need values in [-1, 1]
        - Normalizing inputs to zero-centered range
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
        return np.tanh(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """tanh'(x) = 1 - tanh(x)^2"""
        t = np.tanh(x)
        return 1.0 - t ** 2

    @property
    def name(self) -> str:
        return "Tanh"


# =============================================================================
# 5. GELU — Gaussian Error Linear Unit (The Transformer Standard)
# =============================================================================

class GELU(ActivationFunction):
    """GELU — the activation function used in GPT, BERT, and most LLMs.

    FORMULA (exact):
        GELU(x) = x * Phi(x)

        Where Phi(x) is the CDF of the standard normal distribution:
            Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))

        So: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

    FORMULA (tanh approximation, used in practice):
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

        This approximation is used in GPT-2 and many other models because
        erf() is not available on all hardware (especially early GPUs).

    DERIVATIVE (exact):
        GELU'(x) = Phi(x) + x * phi(x)

        Where phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2) is the standard
        normal PDF (the derivative of the CDF).

        Expanded:
            GELU'(x) = 0.5 * (1 + erf(x / sqrt(2)))
                      + x * (1 / sqrt(2 * pi)) * exp(-x^2 / 2)

    INTUITION — WHY GELU WORKS:
        GELU can be thought of as a "soft" version of ReLU:
        - For large positive x: GELU(x) ≈ x     (like ReLU)
        - For large negative x: GELU(x) ≈ 0     (like ReLU)
        - Near zero: smooth transition (UNLIKE ReLU's hard kink)

        The probabilistic interpretation: GELU multiplies x by the
        probability that x is greater than other inputs (assuming
        Gaussian distribution). Larger inputs are more likely to be
        "kept" and smaller inputs more likely to be "dropped".

        This is like a SMOOTH, LEARNABLE dropout!

    WHY TRANSFORMERS USE GELU INSTEAD OF RELU:
        1. SMOOTH: No hard kink at x = 0. The derivative is continuous,
           which gives smoother optimization landscapes.
        2. NON-MONOTONIC: GELU is slightly negative near x ≈ -0.17,
           which gives it more expressive power than ReLU.
        3. PROBABILISTIC: The Gaussian CDF weighting has a natural
           statistical interpretation.
        4. EMPIRICALLY BETTER: Consistently outperforms ReLU in
           transformer architectures across many benchmarks.

    PAPER: Hendrycks, D. & Gimpel, K. (2016).
           "Gaussian Error Linear Units (GELUs)."
           https://arxiv.org/abs/1606.08415

    Args:
        approximate: If True, use the tanh approximation (faster, standard
                     in GPT-2). If False, use the exact erf formulation.
    """

    def __init__(self, approximate: bool = True) -> None:
        self.approximate = approximate

    def forward(self, x: np.ndarray) -> np.ndarray:
        """GELU(x) = x * Phi(x), with optional tanh approximation."""
        if self.approximate:
            # Tanh approximation (used in GPT-2, PyTorch default)
            # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
            return 0.5 * x * (1.0 + np.tanh(inner))
        else:
            # Exact formulation using the error function
            # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            from scipy.special import erf  # type: ignore[import-untyped,unused-ignore]
            return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivative of GELU.

        For the EXACT form:
            GELU'(x) = Phi(x) + x * phi(x)
            Where:
                Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))   -- standard normal CDF
                phi(x) = (1 / sqrt(2*pi)) * exp(-x^2/2)  -- standard normal PDF

        For the APPROXIMATE (tanh) form, we differentiate:
            f(x) = 0.5 * x * (1 + tanh(u))
            where u = sqrt(2/pi) * (x + 0.044715 * x^3)

            By product rule:
            f'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx

            where du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
                        = sqrt(2/pi) * (1 + 0.134145 * x^2)

            and sech^2(u) = 1 - tanh^2(u)
        """
        if self.approximate:
            # Derivative of the tanh approximation (matches forward exactly)
            coeff = np.sqrt(2.0 / np.pi)
            inner = coeff * (x + 0.044715 * x ** 3)
            tanh_inner = np.tanh(inner)
            sech2 = 1.0 - tanh_inner ** 2
            du_dx = coeff * (1.0 + 3.0 * 0.044715 * x ** 2)
            return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * du_dx
        else:
            # Exact derivative using CDF and PDF of standard normal
            phi_cdf = 0.5 * (1.0 + _erf_approx(x / np.sqrt(2.0)))
            phi_pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x ** 2)
            return phi_cdf + x * phi_pdf

    @property
    def name(self) -> str:
        mode = "approx" if self.approximate else "exact"
        return f"GELU({mode})"


def _erf_approx(x: np.ndarray) -> np.ndarray:
    """Approximate the error function using tanh.

    This avoids importing scipy for the derivative computation.
    The approximation erf(x) ≈ tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
    is accurate to ~1e-4 for most practical values.
    """
    return np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3))


# =============================================================================
# 6. Softmax — Turning Logits into Probabilities
# =============================================================================

class Softmax:
    """Softmax — converts a vector of real numbers into a probability distribution.

    FORMULA:
        softmax(x_i) = exp(x_i) / sum_j(exp(x_j))

    PROPERTIES:
        1. All outputs are POSITIVE (because exp() > 0)
        2. Outputs SUM TO 1 (it's a probability distribution)
        3. PRESERVES ORDERING: if x_i > x_j, then softmax(x_i) > softmax(x_j)
        4. AMPLIFIES DIFFERENCES: the exponential makes large values even larger
           relative to small values

    JACOBIAN (derivative is a MATRIX, not a vector):
        For softmax, each output depends on ALL inputs (not just one).
        So the derivative is a Jacobian matrix:

            dS_i/dx_j = S_i * (delta_ij - S_j)

        Where:
            S_i = softmax(x_i)
            delta_ij = 1 if i == j, else 0 (Kronecker delta)

        Expanded:
            If i == j:  dS_i/dx_i = S_i * (1 - S_i)    ← like sigmoid!
            If i != j:  dS_i/dx_j = -S_i * S_j          ← cross terms

        The Jacobian matrix is:
            J = diag(S) - S * S^T

    NUMERICAL STABILITY (THE MOST IMPORTANT TRICK):
        Problem:  exp(1000) = overflow (inf)
        Solution: Subtract the max before exponentiating

            softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

        WHY THIS WORKS (proof that it doesn't change the result):
            exp(x_i - c) / sum_j(exp(x_j - c))
            = exp(x_i) * exp(-c) / (sum_j(exp(x_j)) * exp(-c))
            = exp(x_i) / sum_j(exp(x_j))
            = softmax(x_i)

        The exp(-c) cancels in numerator and denominator!
        We choose c = max(x) to keep all exponents <= 0, preventing overflow.

    TEMPERATURE SCALING:
        softmax(x / T) where T is the "temperature":
            T = 1.0:  Standard softmax
            T > 1.0:  FLATTER distribution (more uniform, more "exploration")
            T < 1.0:  SHARPER distribution (more peaked, more "exploitation")
            T -> 0:   Approaches argmax (one-hot vector)
            T -> inf: Approaches uniform distribution (1/n, 1/n, ..., 1/n)

        CONNECTION TO PHYSICS:
            The name "temperature" comes from the Boltzmann distribution in
            statistical mechanics: P(state_i) = exp(-E_i / kT) / Z
            where E is energy, k is Boltzmann's constant, T is temperature,
            and Z is the partition function (normalizing constant).
            Higher temperature = more randomness = more entropy.

    USAGE IN NEURAL NETWORKS:
        - FINAL layer of classification networks (turn logits to probabilities)
        - ATTENTION mechanism (softmax of scores gives attention weights)
        - LANGUAGE MODELS (softmax over vocabulary gives next-token distribution)
        - Knowledge distillation (Hinton et al., 2015 — use high T to "soften"
          teacher's predictions)

    Args:
        axis: Axis along which to compute softmax. Default -1 (last axis).
        temperature: Temperature for scaling. Default 1.0.
    """

    def __init__(self, axis: int = -1, temperature: float = 1.0) -> None:
        self.axis = axis
        self.temperature = temperature

    @property
    def name(self) -> str:
        return f"Softmax(T={self.temperature})"

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability and temperature scaling.

        Args:
            x: Input logits of any shape.

        Returns:
            Probability distribution (same shape as x), sums to 1 along axis.
        """
        # Apply temperature scaling
        scaled = x / self.temperature

        # Numerical stability: subtract max to prevent exp() overflow
        # keepdims=True so we can broadcast the subtraction
        shifted = scaled - np.max(scaled, axis=self.axis, keepdims=True)

        # Compute exp and normalize
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=self.axis, keepdims=True)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Compute the Jacobian of softmax.

        For a 1D input of length n, returns an (n, n) Jacobian matrix.
        For batched inputs, returns the Jacobian for the last sample.

        The Jacobian is: J = diag(S) - S @ S^T
            Where S = softmax(x).

        Args:
            x: Input logits (1D vector or last axis is used).

        Returns:
            Jacobian matrix of shape (n, n) where n = x.shape[-1].
        """
        s = self.forward(x)
        # Handle multi-dimensional input: use the last vector
        if s.ndim > 1:
            s = s[-1]
        # Jacobian: diag(s) - outer(s, s)
        return np.diag(s) - np.outer(s, s)


# =============================================================================
# STANDALONE CONVENIENCE FUNCTIONS
# =============================================================================
# These return (value, gradient) tuples for quick one-off computations.
# Use the classes above when you need persistent configuration.
# =============================================================================

def relu(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute ReLU activation and its gradient.

    Args:
        x: Input array.

    Returns:
        Tuple of (activation_value, gradient).

    Example:
        >>> val, grad = relu(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        >>> print(val)   # [0. 0. 0. 1. 2.]
        >>> print(grad)  # [0. 0. 0. 1. 1.]
    """
    fn = ReLU()
    return fn.forward(x), fn.backward(x)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Compute Leaky ReLU activation and its gradient.

    Args:
        x: Input array.
        alpha: Slope for negative inputs (default 0.01).

    Returns:
        Tuple of (activation_value, gradient).

    Example:
        >>> val, grad = leaky_relu(np.array([-2.0, 0.0, 2.0]), alpha=0.1)
        >>> print(val)   # [-0.2  0.   2. ]
        >>> print(grad)  # [0.1 1.  1. ]
    """
    fn = LeakyReLU(alpha=alpha)
    return fn.forward(x), fn.backward(x)


def sigmoid(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Sigmoid activation and its gradient.

    Args:
        x: Input array.

    Returns:
        Tuple of (activation_value, gradient).

    Example:
        >>> val, grad = sigmoid(np.array([0.0]))
        >>> print(val)   # [0.5]
        >>> print(grad)  # [0.25]  <- maximum gradient of sigmoid!
    """
    fn = Sigmoid()
    return fn.forward(x), fn.backward(x)


def tanh(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Tanh activation and its gradient.

    Args:
        x: Input array.

    Returns:
        Tuple of (activation_value, gradient).

    Example:
        >>> val, grad = tanh(np.array([0.0]))
        >>> print(val)   # [0.]
        >>> print(grad)  # [1.]  <- maximum gradient of tanh
    """
    fn = Tanh()
    return fn.forward(x), fn.backward(x)


def gelu(x: np.ndarray, approximate: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute GELU activation and its gradient.

    Args:
        x: Input array.
        approximate: Use tanh approximation (default True, matches PyTorch).

    Returns:
        Tuple of (activation_value, gradient).

    Example:
        >>> val, grad = gelu(np.array([0.0]))
        >>> print(val)   # [0.]
        >>> print(grad)  # [0.5]
    """
    fn = GELU(approximate=approximate)
    return fn.forward(x), fn.backward(x)


def softmax(
    x: np.ndarray,
    axis: int = -1,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Softmax probabilities and Jacobian.

    Args:
        x: Input logits.
        axis: Axis along which to compute softmax.
        temperature: Temperature for scaling (higher = flatter).

    Returns:
        Tuple of (probabilities, jacobian_matrix).

    Example:
        >>> probs, jac = softmax(np.array([1.0, 2.0, 3.0]))
        >>> print(probs)          # [0.09, 0.24, 0.67]
        >>> print(probs.sum())    # 1.0
    """
    fn = Softmax(axis=axis, temperature=temperature)
    return fn.forward(x), fn.backward(x)


# =============================================================================
# NUMERICAL GRADIENT CHECKER
# =============================================================================

def numerical_gradient(
    func: ActivationFunction,
    x: np.ndarray,
    h: float = 1e-7,
) -> np.ndarray:
    """Compute numerical gradient using the central difference method.

    This is the GOLD STANDARD for verifying analytical gradients.

    FORMULA:
        f'(x) ≈ (f(x + h) - f(x - h)) / (2h)

    WHY CENTRAL DIFFERENCE (not forward difference)?
        Forward difference:  (f(x+h) - f(x)) / h          — O(h) error
        Central difference:  (f(x+h) - f(x-h)) / (2h)     — O(h^2) error

        Central difference is MORE ACCURATE for the same h because the
        linear error terms cancel out (Taylor expansion shows this).

    Args:
        func: An ActivationFunction instance.
        x: Point at which to compute the gradient.
        h: Step size (default 1e-7, balancing precision and floating-point error).

    Returns:
        Numerical gradient, same shape as x.
    """
    return (func.forward(x + h) - func.forward(x - h)) / (2.0 * h)


# =============================================================================
# DEMO / MAIN — Run this file to see activations in action
# =============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  ACTIVATION FUNCTIONS FROM SCRATCH")
    print("  Understanding Non-Linearity in Neural Networks")
    print("=" * 72)

    # =========================================================================
    # DEMO 1: Compare all activations at key input values
    # =========================================================================
    print("\n" + "=" * 72)
    print("  DEMO 1: Activation Values at Key Points")
    print("  Comparing f(x) for all activation functions")
    print("=" * 72)

    test_points = np.array([-3.0, -1.0, 0.0, 0.5, 1.0, 3.0])
    activations: list[ActivationFunction] = [
        ReLU(),
        LeakyReLU(alpha=0.01),
        Sigmoid(),
        Tanh(),
        GELU(approximate=True),
    ]

    # Print header
    header = f"{'x':>8s}"
    for act in activations:
        header += f" | {act.name:>16s}"
    print(f"\n{header}")
    print("-" * len(header))

    # Print values
    for x_val in test_points:
        x_arr = np.array([x_val])
        row = f"{x_val:8.1f}"
        for act in activations:
            val = act.forward(x_arr)[0]
            row += f" | {val:16.6f}"
        print(row)

    # =========================================================================
    # DEMO 2: Derivatives at key points
    # =========================================================================
    print("\n" + "=" * 72)
    print("  DEMO 2: Derivatives (Gradients) at Key Points")
    print("  These are the LOCAL gradients used in backpropagation")
    print("=" * 72)

    header = f"{'x':>8s}"
    for act in activations:
        header += f" | {act.name:>16s}"
    print(f"\n{header}")
    print("-" * len(header))

    for x_val in test_points:
        x_arr = np.array([x_val])
        row = f"{x_val:8.1f}"
        for act in activations:
            grad = act.backward(x_arr)[0]
            row += f" | {grad:16.6f}"
        print(row)

    # =========================================================================
    # DEMO 3: The Vanishing Gradient Problem
    # =========================================================================
    print("\n" + "=" * 72)
    print("  DEMO 3: The Vanishing Gradient Problem")
    print("  Sigmoid derivative through 10 layers of backpropagation")
    print("=" * 72)
    print("""
    In backpropagation, gradients are MULTIPLIED through layers (chain rule):
        dL/dx = dL/dy_10 * dy_10/dy_9 * dy_9/dy_8 * ... * dy_1/dx

    Each factor is the activation's derivative. If the derivative is small
    (sigmoid maxes at 0.25), the gradient shrinks EXPONENTIALLY.
    """)

    sig = Sigmoid()
    sig_relu = ReLU()
    sig_tanh = Tanh()

    # Best case for sigmoid: x = 0 gives maximum derivative of 0.25
    sig_grad = 1.0           # Starting gradient (from the loss)
    relu_grad = 1.0
    tanh_grad = 1.0

    print(f"  {'Layer':>6s} | {'Sigmoid Grad':>14s} | {'Tanh Grad':>14s} | {'ReLU Grad':>14s}")
    print(f"  {'-' * 6}-+-{'-' * 14}-+-{'-' * 14}-+-{'-' * 14}")

    for layer in range(1, 11):
        # Best case: x = 0 for sigmoid/tanh, x = 1 for ReLU
        sig_deriv = sig.backward(np.array([0.0]))[0]    # = 0.25
        tanh_deriv = sig_tanh.backward(np.array([0.0]))[0]  # = 1.0
        relu_deriv = sig_relu.backward(np.array([1.0]))[0]  # = 1.0

        sig_grad *= sig_deriv
        tanh_grad *= tanh_deriv
        relu_grad *= relu_deriv

        print(f"  {layer:6d} | {sig_grad:14.10f} | {tanh_grad:14.10f} | {relu_grad:14.10f}")

    print(f"""
    OBSERVATION:
      After 10 layers, sigmoid gradient = {sig_grad:.2e} (essentially ZERO)
      After 10 layers, tanh gradient    = {tanh_grad:.2e} (perfect at x=0, but degrades elsewhere)
      After 10 layers, ReLU gradient    = {relu_grad:.2e} (unchanged! No vanishing gradient)

    This is why ReLU replaced sigmoid/tanh for deep network hidden layers.
    Sigmoid's max derivative of 0.25 means: 0.25^10 = {0.25**10:.2e}
    """)

    # =========================================================================
    # DEMO 4: The Dying ReLU Problem
    # =========================================================================
    print("=" * 72)
    print("  DEMO 4: The Dying ReLU Problem")
    print("  What happens when a ReLU neuron sees only negative inputs")
    print("=" * 72)

    relu_fn = ReLU()
    leaky_fn = LeakyReLU(alpha=0.01)

    negative_inputs = np.array([-5.0, -3.0, -1.0, -0.5, -0.1])

    print(f"\n  {'Input':>8s} | {'ReLU Output':>12s} | {'ReLU Grad':>10s} | "
          f"{'LeakyReLU Out':>14s} | {'LeakyReLU Grad':>15s}")
    print(f"  {'-' * 8}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 14}-+-{'-' * 15}")

    for x_val in negative_inputs:
        x_arr = np.array([x_val])
        r_val = relu_fn.forward(x_arr)[0]
        r_grad = relu_fn.backward(x_arr)[0]
        l_val = leaky_fn.forward(x_arr)[0]
        l_grad = leaky_fn.backward(x_arr)[0]
        print(f"  {x_val:8.1f} | {r_val:12.4f} | {r_grad:10.4f} | "
              f"{l_val:14.4f} | {l_grad:15.4f}")

    print("""
    OBSERVATION:
      ReLU: ALL outputs are 0, ALL gradients are 0.
        -> The neuron is DEAD. No gradient flows. It can NEVER recover.
        -> No matter what the upstream gradient is, 0 * anything = 0.

      Leaky ReLU: Small negative outputs and gradients (alpha = 0.01).
        -> The neuron is NOT dead. A small gradient still flows.
        -> Given enough training, it can potentially recover.

    IN PRACTICE:
      - ~10-40% of neurons can die in a ReLU network
      - He initialization (He et al., 2015) helps prevent this
      - Leaky ReLU / PReLU / ELU are alternatives when dying is a problem
    """)

    # =========================================================================
    # DEMO 5: Softmax with Temperature Scaling
    # =========================================================================
    print("=" * 72)
    print("  DEMO 5: Softmax with Temperature Scaling")
    print("  How temperature controls the 'sharpness' of the distribution")
    print("=" * 72)

    logits = np.array([2.0, 1.0, 0.1])
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"\n  Input logits: {logits}")
    print(f"\n  {'Temperature':>12s} | {'P(class 0)':>11s} | {'P(class 1)':>11s} | "
          f"{'P(class 2)':>11s} | {'Entropy':>8s} | {'Interpretation':>20s}")
    print(f"  {'-' * 12}-+-{'-' * 11}-+-{'-' * 11}-+-{'-' * 11}-+-"
          f"{'-' * 8}-+-{'-' * 20}")

    for T in temperatures:
        sm = Softmax(temperature=T)
        probs = sm.forward(logits)

        # Shannon entropy: H = -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs + 1e-12))

        if T <= 0.1:
            interp = "Near argmax"
        elif T <= 0.5:
            interp = "Very confident"
        elif T <= 1.0:
            interp = "Standard"
        elif T <= 2.0:
            interp = "Softer"
        elif T <= 5.0:
            interp = "Exploratory"
        else:
            interp = "Near uniform"

        print(f"  {T:12.1f} | {probs[0]:11.6f} | {probs[1]:11.6f} | "
              f"{probs[2]:11.6f} | {entropy:8.4f} | {interp:>20s}")

    print("""
    OBSERVATION:
      T -> 0:   Distribution becomes one-hot (all mass on the largest logit)
      T = 1.0:  Standard softmax
      T -> inf: Distribution approaches uniform (1/3, 1/3, 1/3)

    APPLICATIONS OF TEMPERATURE:
      - Language model generation: T < 1 for factual text, T > 1 for creative
      - Knowledge distillation: High T to reveal "dark knowledge" in soft labels
      - Reinforcement learning: Temperature controls exploration vs exploitation
      - Contrastive learning: SimCLR uses T = 0.07 for sharp similarity matching

    CONNECTION TO PHYSICS (Boltzmann Distribution):
      P(state_i) = exp(-E_i / kT) / Z
      In physics, temperature controls how "spread out" particles are across
      energy states. Same idea in softmax: temperature controls how spread
      out probability mass is across classes.
    """)

    # =========================================================================
    # DEMO 6: Softmax Numerical Stability
    # =========================================================================
    print("=" * 72)
    print("  DEMO 6: Softmax Numerical Stability")
    print("  Why we subtract max(x) before exponentiating")
    print("=" * 72)

    # Large logits that would cause overflow without the max-subtraction trick
    large_logits = np.array([1000.0, 1001.0, 1002.0])

    print(f"\n  Logits: {large_logits}")
    print("  exp(1000) would be: overflow (inf)")
    print(f"  exp(1000 - 1002) = exp(-2) = {np.exp(-2.0):.6f}  (safe!)")

    sm = Softmax()
    probs = sm.forward(large_logits)
    print(f"\n  Softmax result (stable): {probs}")
    print(f"  Sum of probabilities:    {probs.sum():.10f}")
    print(f"  All positive?            {np.all(probs > 0)}")

    # Show that the result is the same as with small logits (just shifted)
    small_logits = large_logits - 1000.0  # [0, 1, 2]
    probs_small = sm.forward(small_logits)
    print(f"\n  Equivalent small logits: {small_logits}")
    print(f"  Softmax result:          {probs_small}")
    print(f"  Results match?           {np.allclose(probs, probs_small)}")

    print("""
    WHY THIS WORKS:
      softmax(x - c) = exp(x - c) / sum(exp(x - c))
                     = exp(x)*exp(-c) / (sum(exp(x))*exp(-c))
                     = exp(x) / sum(exp(x))
                     = softmax(x)

      The constant cancels! So subtracting max(x) gives the SAME result
      but prevents overflow. This is used in EVERY neural network library.
    """)

    # =========================================================================
    # DEMO 7: Verify Analytical Gradients Against Numerical Gradients
    # =========================================================================
    print("=" * 72)
    print("  DEMO 7: Gradient Verification")
    print("  Checking analytical derivatives against numerical approximation")
    print("  Using central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)")
    print("=" * 72)

    # NOTE: We avoid x = 0 for ReLU/LeakyReLU because the derivative is
    # technically undefined there (the function has a kink). The numerical
    # gradient averages the left and right derivatives, giving 0.5, while
    # our analytical convention returns 0. Both are valid — the network
    # works fine either way because hitting x = exactly 0 is measure-zero.
    test_x = np.array([-2.0, -1.0, -0.5, 0.3, 0.5, 1.0, 2.0])
    h = 1e-7

    print(f"\n  Test points: {test_x}")
    print(f"  Step size h = {h}\n")

    all_passed = True
    for act in activations:
        analytical = act.backward(test_x)
        numerical = numerical_gradient(act, test_x, h=h)

        # Relative error (with epsilon to avoid division by zero)
        abs_diff = np.abs(analytical - numerical)
        denom = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-12)
        rel_error = abs_diff / denom
        max_rel_error = np.max(rel_error)

        passed = max_rel_error < 1e-4
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  {act.name:>20s} | Max relative error: {max_rel_error:.2e} | [{status}]")

        # Show detailed comparison for the first activation
        if isinstance(act, ReLU):
            print(f"    {'x':>8s} | {'Analytical':>12s} | {'Numerical':>12s} | {'Abs Diff':>12s}")
            print(f"    {'-' * 8}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}")
            for i, xv in enumerate(test_x):
                print(f"    {xv:8.1f} | {analytical[i]:12.8f} | "
                      f"{numerical[i]:12.8f} | {abs_diff[i]:12.2e}")

    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    print("""
    WHY GRADIENT CHECKING MATTERS:
      Bugs in gradient computation are SILENT — the network trains but learns
      poorly. Gradient checking catches these bugs by comparing your analytical
      derivative against a numerical approximation.

      RULE OF THUMB:
        Relative error < 1e-5:  Correct implementation
        Relative error < 1e-3:  Might be OK (check edge cases)
        Relative error > 1e-3:  Almost certainly a bug

      ALWAYS gradient-check new activation functions before using them!
    """)

    # =========================================================================
    # DEMO 8: GELU vs ReLU — The Smooth Difference
    # =========================================================================
    print("=" * 72)
    print("  DEMO 8: GELU vs ReLU — Why Transformers Prefer GELU")
    print("=" * 72)

    gelu_fn = GELU(approximate=True)
    relu_fn = ReLU()

    fine_points = np.array([-2.0, -1.0, -0.5, -0.17, 0.0, 0.5, 1.0, 2.0])

    relu_d_label = "ReLU f'(x)"
    gelu_d_label = "GELU f'(x)"
    print(f"\n  {'x':>8s} | {'ReLU f(x)':>10s} | {'GELU f(x)':>10s} | "
          f"{relu_d_label:>11s} | {gelu_d_label:>11s}")
    print(f"  {'-' * 8}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 11}-+-{'-' * 11}")

    for x_val in fine_points:
        x_arr = np.array([x_val])
        r_v = relu_fn.forward(x_arr)[0]
        g_v = gelu_fn.forward(x_arr)[0]
        r_d = relu_fn.backward(x_arr)[0]
        g_d = gelu_fn.backward(x_arr)[0]
        print(f"  {x_val:8.2f} | {r_v:10.6f} | {g_v:10.6f} | "
              f"{r_d:11.6f} | {g_d:11.6f}")

    print("""
    KEY DIFFERENCES:
      1. At x = -0.17: ReLU = 0, GELU ≈ -0.08 (slightly negative!)
         GELU is NON-MONOTONIC near zero — this gives extra expressiveness.

      2. At x = 0: ReLU derivative JUMPS from 0 to 1 (discontinuous)
         GELU derivative is 0.5 (smooth, continuous everywhere)

      3. For x < 0: ReLU gradient is EXACTLY 0 (dead zone)
         GELU gradient smoothly approaches 0 (information still flows)

    This smoothness is why GELU works better in transformers, where the
    optimization landscape benefits from continuous, well-behaved gradients.
    """)

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 72)
    print("  SUMMARY: Which Activation Function to Use?")
    print("=" * 72)
    print("""
    HIDDEN LAYERS:
      Default choice:        ReLU (simple, fast, works well)
      If dying ReLU:         Leaky ReLU or PReLU
      Transformers/LLMs:     GELU (smooth, empirically better)

    OUTPUT LAYERS:
      Binary classification: Sigmoid (outputs probability in [0, 1])
      Multi-class:           Softmax (outputs probability distribution)
      Regression:            Linear (no activation, or tanh for bounded output)

    GATES (LSTM/GRU):
      Forget/input gates:    Sigmoid (controls "how much" to remember/forget)
      Candidate values:      Tanh (zero-centered, bounded)

    RULES OF THUMB:
      - Start with ReLU. Switch only if you have a reason.
      - For transformers, use GELU.
      - Never use sigmoid/tanh in hidden layers of deep networks.
      - Always use numerically stable softmax (subtract max).
      - Gradient-check any custom activation function.
    """)

    logger.info(
        "activation_functions_demo_complete",
        activations_covered=["ReLU", "LeakyReLU", "Sigmoid", "Tanh", "GELU", "Softmax"],
        demos_run=8,
    )
