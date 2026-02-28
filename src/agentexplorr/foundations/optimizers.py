"""
Gradient Descent Optimizers — From Scratch
============================================

This module implements the most important gradient descent optimizers used in
modern deep learning, built entirely from NumPy so you can see every step of
the math. No magic, no hidden abstractions.

WHAT IS AN OPTIMIZER?
    Training a neural network means finding weights (parameters) that minimize
    a loss function. An optimizer is the algorithm that iteratively adjusts
    those weights to reduce the loss.

    Imagine you are blindfolded on a hilly landscape. Your goal is to reach the
    lowest valley. You can only feel the slope beneath your feet (the gradient).
    An optimizer is your strategy for deciding which direction to step and how
    far to go.

WHAT IS GRADIENT DESCENT?
    The gradient of a function at a point tells you the direction of STEEPEST
    ASCENT. So the NEGATIVE gradient points downhill — toward lower loss.

    The simplest strategy: take a small step in the negative gradient direction.

        w_new = w_old - learning_rate * gradient

    This is vanilla gradient descent. Everything else in this file is a
    refinement of this one idea.

    Geometrically:
    - The loss function defines a surface in weight space
    - The gradient is a vector pointing "uphill" on that surface
    - We step in the opposite direction (downhill)
    - The learning rate controls step size

THE LEARNING RATE (lr):
    The most important hyperparameter in all of deep learning.

    - Too large:  Steps overshoot the minimum, loss oscillates or diverges
                  Imagine taking huge leaps on a mountain — you jump over
                  the valley and land on the other side!

    - Too small:  Steps are tiny, training takes forever, and you might get
                  stuck in a local minimum or saddle point

    - Just right: Smooth convergence to a good minimum

    Typical values: 1e-4 to 1e-2 (depends heavily on the optimizer)

WHY "STOCHASTIC" GRADIENT DESCENT?
    True gradient descent computes the gradient using ALL training examples.
    That is expensive! Instead, we estimate the gradient using a random
    mini-batch (e.g., 32 or 64 examples). This estimate is noisy but:
      1. Much faster per step (process 32 examples vs 50,000)
      2. The noise actually helps escape local minima (acts as regularization)
      3. Enables training on datasets that don't fit in memory

    "Stochastic" = random. The randomness comes from the mini-batch sampling.

THE OPTIMIZER ZOO — A BRIEF HISTORY:
    1986: SGD              — The original (Rumelhart, Hinton, Williams)
    1999: SGD + Momentum   — Add "inertia" to smooth updates (Qian, 1999)
    2011: Adagrad          — Per-parameter learning rates (Duchi et al.)
    2012: RMSprop          — Fix Adagrad's dying learning rates (Hinton, unpublished)
    2014: Adam             — Combine momentum + RMSprop (Kingma & Ba)
    2017: AdamW            — Fix Adam's weight decay (Loshchilov & Hutter)

    Today, AdamW is the default for most transformer / LLM training.

LEARNING RESOURCES:
    - PAPER: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
      https://arxiv.org/abs/1412.6980
    - PAPER: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)
      https://arxiv.org/abs/1711.05101
    - PAPER: "On the importance of initialization and momentum in deep learning"
      (Sutskever et al., 2013)
      https://proceedings.mlr.press/v28/sutskever13.html
    - BLOG: Sebastian Ruder's "An overview of gradient descent optimization"
      https://ruder.io/optimizing-gradient-descent/
    - VIDEO: "How Optimization for Machine Learning Works" — 3Blue1Brown
      https://www.youtube.com/watch?v=IHZwWFHWa-w
    - VIDEO: "Why Adam Works" — Alfredo Canziani (NYU Deep Learning)
      https://www.youtube.com/watch?v=JXQT_vxqwIs
    - DOCS: PyTorch optim module
      https://pytorch.org/docs/stable/optim.html
    - BOOK: Goodfellow, Bengio, Courville — "Deep Learning", Chapter 8
      https://www.deeplearningbook.org/contents/optimization.html
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agentexplorr.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Base Optimizer — abstract interface that all optimizers share
# =============================================================================

class Optimizer:
    """Base class for all gradient descent optimizers.

    Every optimizer follows the same contract:
      1. __init__(): Store hyperparameters (learning rate, etc.) and initialize
         any internal state (momentum buffers, running averages, etc.)
      2. step(params, grads): Given current parameters and their gradients,
         compute and apply one update step.
      3. reset(): Clear all internal state (useful when starting a new
         optimization run on a different problem).

    WHY A BASE CLASS?
        All optimizers share the same interface. Having a base class lets us
        swap optimizers without changing the training loop code — a key
        principle in software design (Liskov Substitution Principle).

    Attributes:
        lr: Learning rate — controls the step size of each update.
        t: Step counter — tracks how many update steps have been taken.
            Some optimizers (like Adam) use this for bias correction.
        state: Dictionary holding any internal optimizer state (momentum
            buffers, running averages, etc.). Keyed by parameter index.
    """

    def __init__(self, lr: float = 0.01) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        self.lr: float = lr
        self.t: int = 0  # Step counter (used by Adam for bias correction)
        self.state: dict[int, dict[str, Any]] = {}

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        """Perform one optimization step.

        This method is called once per mini-batch during training. It takes
        the current parameter values and their gradients, and returns the
        updated parameter values.

        Args:
            params: List of parameter arrays (e.g., weights and biases).
                    Each element is a numpy array of any shape.
            grads:  List of gradient arrays, same shapes as params.
                    grads[i] is d(loss)/d(params[i]).

        Returns:
            List of updated parameter arrays (same shapes as input).

        Raises:
            ValueError: If params and grads have different lengths.
        """
        raise NotImplementedError("Subclasses must implement step()")

    def reset(self) -> None:
        """Clear all internal state.

        Call this when you want to restart optimization from scratch
        (e.g., on a new problem or a new set of parameters). This resets
        momentum buffers, running averages, and the step counter.
        """
        self.t = 0
        self.state = {}

    def _validate_inputs(
        self, params: list[np.ndarray], grads: list[np.ndarray]
    ) -> None:
        """Validate that params and grads are compatible.

        Args:
            params: List of parameter arrays.
            grads:  List of gradient arrays.

        Raises:
            ValueError: If lengths or shapes don't match.
        """
        if len(params) != len(grads):
            raise ValueError(
                f"params and grads must have the same length, "
                f"got {len(params)} and {len(grads)}"
            )
        for i, (p, g) in enumerate(zip(params, grads, strict=False)):
            if p.shape != g.shape:
                raise ValueError(
                    f"Shape mismatch at index {i}: "
                    f"param shape {p.shape} != grad shape {g.shape}"
                )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self.lr})"


# =============================================================================
# SGD — Stochastic Gradient Descent (the simplest optimizer)
# =============================================================================

class SGD(Optimizer):
    """Stochastic Gradient Descent — the simplest and most fundamental optimizer.

    UPDATE RULE:
        w = w - lr * gradient

    That's it. Just take a step in the negative gradient direction.

    GEOMETRIC INTUITION:
        Picture the loss function as a bowl-shaped surface. The gradient at any
        point is a vector pointing "uphill." We step in the opposite direction
        (downhill). The learning rate controls how big each step is.

        In 2D, if the loss is f(w1, w2), then:
            gradient = [df/dw1, df/dw2]   (points uphill)
            step     = -lr * gradient      (points downhill)

    WHY "STOCHASTIC"?
        In true gradient descent, you'd compute the gradient using ALL training
        examples (the entire dataset). Stochastic GD uses a random mini-batch
        instead. This is noisy but:
          - Much faster (process 32 examples instead of 50,000)
          - The noise helps escape shallow local minima
          - Enables training on data that doesn't fit in memory

    PROBLEMS WITH VANILLA SGD:
        1. OSCILLATION IN NARROW VALLEYS:
           If the loss surface is shaped like a narrow ravine (common in
           practice), SGD oscillates back and forth across the valley walls
           while making slow progress along the valley floor.

           Imagine a marble rolling in a long, narrow trough — it bounces
           side to side while slowly moving forward.

        2. SAME LEARNING RATE FOR ALL PARAMETERS:
           Some parameters might need bigger steps (they're in a flat region)
           while others need smaller steps (they're in a steep region). SGD
           uses the same learning rate for everything.

        3. SENSITIVE TO LEARNING RATE:
           Too large and training diverges. Too small and it crawls.
           No good way to automatically adjust.

        These problems motivated all the subsequent optimizers in this file.

    WHEN TO USE SGD:
        - Simple problems or when you want to understand the basics
        - SGD with momentum is still competitive for CNNs on image tasks
        - Research: SGD often generalizes better than Adam (but trains slower)

    LEARNING RESOURCES:
        - PAPER: "Learning representations by back-propagating errors"
          (Rumelhart, Hinton, Williams, 1986)
          https://www.nature.com/articles/323533a0
        - BLOG: "An overview of gradient descent optimization algorithms"
          https://ruder.io/optimizing-gradient-descent/

    Example:
        >>> import numpy as np
        >>> optimizer = SGD(lr=0.01)
        >>> params = [np.array([3.0, -2.0])]
        >>> grads = [np.array([1.0, -0.5])]
        >>> updated = optimizer.step(params, grads)
        >>> print(updated[0])  # [2.99, -1.995]

    Args:
        lr: Learning rate (step size). Typical values: 0.001 to 0.1.
    """

    def __init__(self, lr: float = 0.01) -> None:
        super().__init__(lr=lr)
        logger.info("optimizer_created", optimizer="SGD", lr=lr)

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        """Perform one SGD update step.

        THE MATH:
            For each parameter w with gradient g:
                w_new = w - lr * g

            This is the gradient descent update rule. Nothing more.

        Args:
            params: Current parameter values.
            grads:  Gradients of the loss w.r.t. each parameter.

        Returns:
            Updated parameter values.
        """
        self._validate_inputs(params, grads)
        self.t += 1

        updated = []
        for p, g in zip(params, grads, strict=False):
            # The core update: step downhill
            p_new = p - self.lr * g
            updated.append(p_new)

        return updated


# =============================================================================
# Momentum — SGD with a "memory" of past gradients
# =============================================================================

class Momentum(Optimizer):
    r"""SGD with Momentum — accelerates convergence by accumulating velocity.

    THE KEY IDEA:
        Instead of using only the current gradient, maintain a running average
        of past gradients (the "velocity"). This smooths out oscillations and
        accelerates movement in directions with consistent gradients.

    UPDATE RULE:
        v = beta * v + g           (update velocity: blend old velocity with new gradient)
        w = w - lr * v              (update weights using velocity)

        Where:
          - v is the velocity (accumulated gradient history)
          - beta is the momentum coefficient (how much history to keep)
          - g is the current gradient
          - lr is the learning rate

    PHYSICAL ANALOGY:
        Think of a ball rolling downhill on the loss surface:
          - Without momentum (SGD): the ball has no mass. It moves exactly
            where the slope pushes it. It oscillates in ravines.
          - With momentum: the ball has mass and inertia. It builds up speed
            in consistent directions and smooths through oscillations.

        The momentum coefficient beta controls how "heavy" the ball is:
          - beta = 0: no momentum, same as SGD
          - beta = 0.9: keep 90% of previous velocity (typical value)
          - beta = 0.99: very heavy ball, very smooth but slow to change direction

    WHY MOMENTUM HELPS:
        1. SMOOTHS OSCILLATIONS:
           In narrow valleys, gradients point side-to-side. Momentum averages
           these out (they cancel), while gradients along the valley floor
           accumulate, leading to faster progress.

           Without momentum:  \/\/\/\/\/  (zigzag across the valley)
           With momentum:     ---------->  (smooth path along the valley)

        2. ACCELERATES THROUGH CONSISTENT GRADIENTS:
           When the gradient points in the same direction repeatedly, velocity
           builds up. This means momentum takes bigger steps in flat regions
           and smaller steps in curved regions — exactly what you want.

        3. ESCAPES LOCAL MINIMA:
           The accumulated velocity can carry the ball through small bumps
           in the loss surface, helping escape shallow local minima.

    NESTEROV MOMENTUM (look-ahead variant):
        Standard momentum: compute gradient at current position, then step.
        Nesterov:         step to where momentum would take you, THEN compute
                          gradient at that look-ahead position.

        Nesterov update:
            v = beta * v + gradient(w - lr * beta * v)    (gradient at look-ahead)
            w = w - lr * v

        Why is Nesterov better? It's like looking where you're about to go
        before deciding how to adjust. If the look-ahead position is past the
        minimum, the gradient there will push you back — a corrective signal
        that standard momentum doesn't have.

        In this implementation, we approximate Nesterov momentum using the
        equivalent reformulation:
            v = beta * v + g
            w = w - lr * (g + beta * v)

        This is algebraically equivalent but easier to implement (no need to
        evaluate the gradient at a different position).

    LEARNING RESOURCES:
        - PAPER: "On the importance of initialization and momentum in deep
          learning" (Sutskever et al., 2013)
          https://proceedings.mlr.press/v28/sutskever13.html
        - PAPER: "A method for solving a convex programming problem with
          convergence rate O(1/k^2)" (Nesterov, 1983)
        - VIDEO: "Momentum in Gradient Descent" — deeplearning.ai
          https://www.youtube.com/watch?v=k8fTYJPd3_I

    Example:
        >>> optimizer = Momentum(lr=0.01, beta=0.9)
        >>> params = [np.array([3.0, -2.0])]
        >>> grads = [np.array([1.0, -0.5])]
        >>> updated = optimizer.step(params, grads)

    Args:
        lr:       Learning rate (step size). Typical: 0.01.
        beta:     Momentum coefficient (how much history to retain).
                  Typical: 0.9. Must be in [0, 1).
        nesterov: If True, use Nesterov (look-ahead) momentum.
    """

    def __init__(
        self,
        lr: float = 0.01,
        beta: float = 0.9,
        nesterov: bool = False,
    ) -> None:
        super().__init__(lr=lr)
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        self.beta: float = beta
        self.nesterov: bool = nesterov
        logger.info(
            "optimizer_created",
            optimizer="Momentum",
            lr=lr,
            beta=beta,
            nesterov=nesterov,
        )

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        """Perform one momentum update step.

        THE MATH:
            For each parameter w_i with gradient g_i:

            Standard momentum:
                v_i = beta * v_i + g_i
                w_i = w_i - lr * v_i

            Nesterov momentum (equivalent reformulation):
                v_i = beta * v_i + g_i
                w_i = w_i - lr * (g_i + beta * v_i)

        WHY THIS WORKS:
            The velocity v is an exponential moving average of gradients.
            When gradients consistently point the same way, v grows large
            (acceleration). When gradients alternate direction, v stays
            small (damping oscillations).

        Args:
            params: Current parameter values.
            grads:  Gradients of the loss w.r.t. each parameter.

        Returns:
            Updated parameter values.
        """
        self._validate_inputs(params, grads)
        self.t += 1

        updated = []
        for i, (p, g) in enumerate(zip(params, grads, strict=False)):
            # Initialize velocity to zeros on first step
            if i not in self.state:
                self.state[i] = {"v": np.zeros_like(p)}

            v = self.state[i]["v"]

            # Update velocity: blend old velocity with new gradient
            #   v = beta * v + g
            # This is an exponential moving average. With beta=0.9, the velocity
            # "remembers" roughly the last 10 gradients (1/(1-0.9) = 10).
            v = self.beta * v + g
            self.state[i]["v"] = v

            # Nesterov: w = w - lr * (g + beta * v) — look-ahead correction
            # Standard: w = w - lr * v — velocity directly
            p_new = p - self.lr * (g + self.beta * v) if self.nesterov else p - self.lr * v

            updated.append(p_new)

        return updated

    def __repr__(self) -> str:
        nesterov_str = ", nesterov=True" if self.nesterov else ""
        return f"Momentum(lr={self.lr}, beta={self.beta}{nesterov_str})"


# =============================================================================
# Adam — Adaptive Moment Estimation (the modern default)
# =============================================================================

class Adam(Optimizer):
    """Adam — Adaptive Moment Estimation.

    Adam combines TWO ideas:
      1. Momentum (1st moment): track the mean of gradients over time
      2. Adaptive learning rates (2nd moment): track the mean of SQUARED
         gradients to give each parameter its own effective learning rate

    THE TWO RUNNING AVERAGES:
        m = exponential moving average of gradients      (1st moment, "mean")
        v = exponential moving average of squared grads  (2nd moment, "variance")

        Think of it this way:
        - m tells you "which direction have gradients been pointing recently?"
          (like momentum — smooths the update direction)
        - v tells you "how large have gradients been recently?"
          (used to normalize — big gradients get smaller steps)

    UPDATE RULES:
        1. Update biased 1st moment estimate (momentum):
            m = beta1 * m + (1 - beta1) * g

        2. Update biased 2nd moment estimate (adaptive learning rate):
            v = beta2 * v + (1 - beta2) * g^2     (element-wise squaring)

        3. Bias correction (crucial for early steps!):
            m_hat = m / (1 - beta1^t)
            v_hat = v / (1 - beta2^t)

        4. Update parameters:
            w = w - lr * m_hat / (sqrt(v_hat) + epsilon)

    WHY BIAS CORRECTION?
        m and v are initialized to zero. In the first few steps, they are
        biased toward zero because they haven't accumulated enough history.

        Example with beta1=0.9:
          Step 1: m = 0.9*0 + 0.1*g1 = 0.1*g1    (biased toward 0!)
          Step 2: m = 0.9*(0.1*g1) + 0.1*g2       (still mostly zeros)

        The true mean is closer to g, but m is only 0.1*g. Dividing by
        (1 - 0.9^1) = 0.1 corrects this: m_hat = 0.1*g / 0.1 = g.

        After many steps, beta1^t -> 0, so (1 - beta1^t) -> 1, and the
        correction vanishes. It only matters for the first ~10-20 steps.

    WHY THE sqrt(v) NORMALIZATION?
        Each parameter's update is divided by sqrt(v_hat), which is roughly
        the RMS (root mean square) of recent gradients for that parameter.

        - Parameters with large gradients (steep loss surface) get SMALLER
          effective learning rates: lr / sqrt(large_v) = small step
        - Parameters with small gradients (flat loss surface) get LARGER
          effective learning rates: lr / sqrt(small_v) = big step

        This is why Adam is "adaptive" — it automatically adjusts the
        learning rate per parameter.

    DEFAULT HYPERPARAMETERS (from the original paper):
        - lr:     0.001    (much smaller than SGD's typical 0.01-0.1)
        - beta1:  0.9      (momentum coefficient for 1st moment)
        - beta2:  0.999    (coefficient for 2nd moment; higher = longer memory)
        - epsilon: 1e-8    (numerical stability; prevents division by zero)

    WHY ADAM IS POPULAR:
        1. Works well with default hyperparameters (less tuning needed)
        2. Handles sparse gradients (NLP, embeddings)
        3. Adapts to each parameter's gradient magnitude
        4. Combines benefits of momentum and RMSprop

    LIMITATIONS OF ADAM:
        1. May generalize worse than SGD+momentum on some tasks (especially
           image classification with CNNs)
        2. Weight decay interacts poorly with adaptive learning rates
           (this is why AdamW was invented — see below)
        3. Can converge to sharp minima (which generalize less well)

    LEARNING RESOURCES:
        - PAPER: "Adam: A Method for Stochastic Optimization"
          (Kingma & Ba, 2014)
          https://arxiv.org/abs/1412.6980
        - VIDEO: "Adam Optimizer" — Alfredo Canziani (NYU)
          https://www.youtube.com/watch?v=JXQT_vxqwIs
        - BLOG: "Adam — latest trends in deep learning optimization"
          https://ruder.io/optimizing-gradient-descent/index.html#adam

    Example:
        >>> optimizer = Adam(lr=0.001)
        >>> params = [np.array([3.0, -2.0])]
        >>> grads = [np.array([1.0, -0.5])]
        >>> updated = optimizer.step(params, grads)

    Args:
        lr:      Learning rate. Default: 0.001.
        beta1:   Exponential decay rate for 1st moment (momentum).
                 Default: 0.9.
        beta2:   Exponential decay rate for 2nd moment (adaptive LR).
                 Default: 0.999.
        epsilon: Small constant for numerical stability. Default: 1e-8.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(lr=lr)
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon
        logger.info(
            "optimizer_created",
            optimizer="Adam",
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        """Perform one Adam update step.

        THE MATH (for each parameter w with gradient g):

            Step 1 — Update biased first moment estimate (momentum-like):
                m = beta1 * m + (1 - beta1) * g

                This is an exponential moving average (EMA) of gradients.
                With beta1=0.9, it roughly averages the last ~10 gradients.
                Similar to momentum, it smooths the update direction.

            Step 2 — Update biased second moment estimate (adaptive LR):
                v = beta2 * v + (1 - beta2) * g^2

                EMA of squared gradients. With beta2=0.999, it averages the
                last ~1000 squared gradients. This tracks the "scale" of
                gradients for each parameter individually.

            Step 3 — Bias correction:
                m_hat = m / (1 - beta1^t)
                v_hat = v / (1 - beta2^t)

                Without this, m and v underestimate the true moments in
                early training because they're initialized to zero.

            Step 4 — Parameter update:
                w = w - lr * m_hat / (sqrt(v_hat) + epsilon)

                The update direction comes from m_hat (smoothed gradient).
                The step size is ADAPTED per-parameter by dividing by
                sqrt(v_hat) — parameters with historically large gradients
                get smaller updates, and vice versa.

        Args:
            params: Current parameter values.
            grads:  Gradients of the loss w.r.t. each parameter.

        Returns:
            Updated parameter values.
        """
        self._validate_inputs(params, grads)
        self.t += 1

        updated = []
        for i, (p, g) in enumerate(zip(params, grads, strict=False)):
            # Initialize state on first step
            if i not in self.state:
                self.state[i] = {
                    "m": np.zeros_like(p),  # 1st moment vector (mean of gradients)
                    "v": np.zeros_like(p),  # 2nd moment vector (mean of squared gradients)
                }

            m = self.state[i]["m"]
            v = self.state[i]["v"]

            # --- Step 1: Update biased first moment estimate ---
            # m = beta1 * m + (1 - beta1) * g
            # This is momentum: a weighted average of old direction + new gradient
            m = self.beta1 * m + (1.0 - self.beta1) * g

            # --- Step 2: Update biased second moment estimate ---
            # v = beta2 * v + (1 - beta2) * g^2
            # This tracks how "large" gradients typically are for each parameter.
            # Note: g**2 is element-wise squaring, NOT a dot product.
            v = self.beta2 * v + (1.0 - self.beta2) * (g ** 2)

            # Save updated state
            self.state[i]["m"] = m
            self.state[i]["v"] = v

            # --- Step 3: Bias correction ---
            # m_hat = m / (1 - beta1^t)
            # v_hat = v / (1 - beta2^t)
            # This compensates for the zero initialization of m and v.
            # In early steps, t is small so beta^t is large, and the correction
            # is significant. As t grows, beta^t -> 0 and correction vanishes.
            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)

            # --- Step 4: Update parameters ---
            # w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
            #
            # The numerator (m_hat) provides the direction (like momentum).
            # The denominator (sqrt(v_hat) + eps) provides per-parameter scaling:
            #   - Large past gradients -> large v_hat -> smaller step
            #   - Small past gradients -> small v_hat -> larger step
            p_new = p - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            updated.append(p_new)

        return updated

    def __repr__(self) -> str:
        return (
            f"Adam(lr={self.lr}, beta1={self.beta1}, "
            f"beta2={self.beta2}, epsilon={self.epsilon})"
        )


# =============================================================================
# AdamW — Adam with Decoupled Weight Decay (the LLM standard)
# =============================================================================

class AdamW(Optimizer):
    """AdamW — Adam with Decoupled Weight Decay Regularization.

    AdamW is a critical fix to how Adam handles weight decay (L2 regularization).
    It is the standard optimizer for training transformers and large language
    models (GPT, BERT, LLaMA, etc.).

    THE PROBLEM WITH L2 REGULARIZATION IN ADAM:
        Weight decay / L2 regularization adds a penalty for large weights:
            L_reg = L + (lambda/2) * ||w||^2

        The gradient of this regularized loss is:
            g_reg = g + lambda * w

        In vanilla Adam, this regularized gradient goes through the adaptive
        scaling (division by sqrt(v)):
            w = w - lr * (g + lambda * w) / (sqrt(v) + eps)

        But this means the weight decay term (lambda * w) is ALSO scaled by
        the adaptive learning rate! This is NOT the same as true weight decay.

        Why? Because:
        - For parameters with large past gradients, v is large, so the weight
          decay effect is weakened (divided by a large number).
        - For parameters with small past gradients, v is small, so the weight
          decay effect is amplified.

        This coupling between weight decay and adaptive learning rates
        is undesirable and was shown to hurt generalization.

    THE ADAMW FIX:
        AdamW DECOUPLES weight decay from the gradient-based update.
        The weight decay is applied DIRECTLY to the weights, AFTER the
        Adam update, without going through the adaptive scaling:

            # Standard Adam update (same as before, on RAW gradient g):
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g^2
            m_hat = m / (1 - beta1^t)
            v_hat = v / (1 - beta2^t)

            # Apply Adam update AND weight decay SEPARATELY:
            w = w - lr * m_hat / (sqrt(v_hat) + eps) - lr * lambda * w
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^
                    gradient-based update                weight decay
                    (adaptive scaling applied)           (NO adaptive scaling)

        Now weight decay shrinks ALL weights uniformly by a fixed fraction
        per step, regardless of their gradient history. This is true weight
        decay (as originally proposed by Hanson & Pratt, 1988).

    WHY THIS MATTERS FOR TRANSFORMERS:
        Transformers have many parameters with very different gradient
        magnitudes (attention weights vs FFN weights vs embeddings). With
        vanilla Adam + L2, some parameters effectively have much stronger
        or weaker regularization than intended. AdamW ensures consistent
        regularization across all parameters.

        In practice, AdamW with weight_decay=0.01-0.1 significantly improves
        generalization of large transformer models.

    TYPICAL HYPERPARAMETERS FOR TRANSFORMER TRAINING:
        - lr:           1e-4 to 3e-4 (often with warmup + cosine decay)
        - beta1:        0.9
        - beta2:        0.95 to 0.999
        - epsilon:      1e-8
        - weight_decay: 0.01 to 0.1
        - Warmup:       first 1-5% of training steps

    LEARNING RESOURCES:
        - PAPER: "Decoupled Weight Decay Regularization"
          (Loshchilov & Hutter, 2017)
          https://arxiv.org/abs/1711.05101
        - PAPER: "Fixing Weight Decay Regularization in Adam"
          (Loshchilov & Hutter, 2018 — ICLR workshop version)
          https://openreview.net/forum?id=rk6qdGgCZ
        - VIDEO: "AdamW and Super-Convergence" — fast.ai
          https://www.youtube.com/watch?v=QN4Q8dS1E5c
        - BLOG: "AdamW vs Adam" — fast.ai
          https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html

    Example:
        >>> optimizer = AdamW(lr=0.001, weight_decay=0.01)
        >>> params = [np.array([3.0, -2.0])]
        >>> grads = [np.array([1.0, -0.5])]
        >>> updated = optimizer.step(params, grads)

    Args:
        lr:           Learning rate. Default: 0.001.
        beta1:        Decay rate for 1st moment. Default: 0.9.
        beta2:        Decay rate for 2nd moment. Default: 0.999.
        epsilon:      Numerical stability constant. Default: 1e-8.
        weight_decay: Weight decay coefficient (lambda). Default: 0.01.
                      Set to 0.0 to recover standard Adam.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(lr=lr)
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon
        self.weight_decay: float = weight_decay
        logger.info(
            "optimizer_created",
            optimizer="AdamW",
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
        )

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        """Perform one AdamW update step.

        THE MATH (for each parameter w with gradient g):

            Steps 1-3 are IDENTICAL to Adam (using the raw gradient g,
            NOT the regularized gradient g + lambda * w):

            Step 1:  m = beta1 * m + (1 - beta1) * g
            Step 2:  v = beta2 * v + (1 - beta2) * g^2
            Step 3:  m_hat = m / (1 - beta1^t)
                     v_hat = v / (1 - beta2^t)

            Step 4 — Update with DECOUPLED weight decay:
                w = w - lr * m_hat / (sqrt(v_hat) + eps)    [Adam step]
                    - lr * weight_decay * w                  [weight decay]

            COMPARE WITH ADAM + L2:
                In Adam + L2, the gradient is g + lambda * w, and this goes
                through adaptive scaling. In AdamW, g and lambda * w are
                applied separately. This seemingly small difference has a
                big impact on generalization for large models.

        Args:
            params: Current parameter values.
            grads:  Gradients of the loss w.r.t. each parameter (WITHOUT
                    any regularization term — AdamW handles that internally).

        Returns:
            Updated parameter values.
        """
        self._validate_inputs(params, grads)
        self.t += 1

        updated = []
        for i, (p, g) in enumerate(zip(params, grads, strict=False)):
            # Initialize state on first step
            if i not in self.state:
                self.state[i] = {
                    "m": np.zeros_like(p),  # 1st moment (mean of gradients)
                    "v": np.zeros_like(p),  # 2nd moment (mean of squared grads)
                }

            m = self.state[i]["m"]
            v = self.state[i]["v"]

            # --- Steps 1-2: Same as Adam (using RAW gradient, not regularized) ---
            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * (g ** 2)

            self.state[i]["m"] = m
            self.state[i]["v"] = v

            # --- Step 3: Bias correction (same as Adam) ---
            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)

            # --- Step 4: DECOUPLED update ---
            # First: the Adam gradient step (adaptive)
            adam_step = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Second: the weight decay step (NOT adaptive — applied directly)
            decay_step = self.lr * self.weight_decay * p

            # Combined update
            p_new = p - adam_step - decay_step

            updated.append(p_new)

        return updated

    def __repr__(self) -> str:
        return (
            f"AdamW(lr={self.lr}, beta1={self.beta1}, "
            f"beta2={self.beta2}, weight_decay={self.weight_decay})"
        )


# =============================================================================
# Learning Rate Schedules — Brief Overview
# =============================================================================
#
# Learning rate schedules adjust the learning rate DURING training. This is
# almost always beneficial — training with a fixed learning rate is suboptimal.
#
# The optimizers above use a fixed lr. In practice, you wrap them with a
# schedule that modifies lr at each step or epoch. Here's a brief overview
# of the most common schedules:
#
# 1. WARMUP
#    --------
#    Start with a very small learning rate and linearly increase it over
#    the first N steps (typically 1-5% of total training steps).
#
#    WHY? In the first few steps, the model's parameters are random. Large
#    updates can push them into bad regions of the loss surface. Starting
#    slow lets the model "orient" itself before taking big steps.
#
#    Warmup is ESSENTIAL for transformer training. Without it, training
#    often diverges immediately.
#
#    lr_warmup(step) = lr_max * (step / warmup_steps)    for step < warmup_steps
#
# 2. COSINE ANNEALING
#    ------------------
#    After warmup, smoothly decrease the learning rate following a cosine curve
#    from lr_max down to lr_min (typically 0 or 0.1 * lr_max).
#
#    WHY COSINE? It decays quickly at first (when we're far from the optimum),
#    then slowly near the end (when we're fine-tuning near the minimum). This
#    matches the intuition that early training needs exploration (big steps)
#    while late training needs exploitation (small steps).
#
#    lr_cosine(step) = lr_min + 0.5 * (lr_max - lr_min) *
#                      (1 + cos(pi * step / total_steps))
#
#    The combination of warmup + cosine annealing is the most popular schedule
#    for training LLMs (used in GPT-3, LLaMA, etc.).
#
# 3. STEP DECAY
#    -----------
#    Drop the learning rate by a fixed factor (e.g., 0.1) every N epochs.
#
#    lr_step(epoch) = lr_initial * gamma^(epoch // step_size)
#
#    Simple and effective for CNNs. Common recipe: drop by 10x at epochs
#    30, 60, 90 in a 100-epoch training run.
#
# 4. LINEAR DECAY
#    -------------
#    Decrease the learning rate linearly from lr_max to 0 (or lr_min).
#
#    lr_linear(step) = lr_max * (1 - step / total_steps)
#
#    Simple, used in some BERT fine-tuning setups.
#
# 5. REDUCE ON PLATEAU
#    -------------------
#    Monitor validation loss. If it hasn't improved for `patience` epochs,
#    reduce the learning rate by a factor. This is adaptive — the schedule
#    responds to the model's actual performance rather than following a
#    predetermined curve.
#
# IMPLEMENTATION NOTE:
#    Full implementations of these schedules are beyond the scope of this
#    file. In practice, you'd use PyTorch's built-in schedulers:
#      - torch.optim.lr_scheduler.CosineAnnealingLR
#      - torch.optim.lr_scheduler.StepLR
#      - torch.optim.lr_scheduler.OneCycleLR (warmup + cosine)
#      - torch.optim.lr_scheduler.ReduceLROnPlateau
#
#    See: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
#
# LEARNING RESOURCES:
#    - PAPER: "SGDR: Stochastic Gradient Descent with Warm Restarts"
#      (Loshchilov & Hutter, 2016) — introduced cosine annealing
#      https://arxiv.org/abs/1608.03983
#    - VIDEO: "Learning Rate Schedules" — deeplearning.ai
#      https://www.youtube.com/watch?v=kk2BQRhmGRY
#    - BLOG: "The 1cycle Learning Rate Policy" — fast.ai
#      https://sgugger.github.io/the-1cycle-policy.html
# =============================================================================


# =============================================================================
# Demo — Runnable optimization examples
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # DEMO: Comparing Optimizers on Test Functions
    # =========================================================================
    #
    # We'll optimize two classic test functions to see how each optimizer
    # behaves. These functions have known minima, so we can verify correctness
    # and compare convergence speed.
    #
    # Test Function 1: Quadratic Bowl (simple)
    #     f(x, y) = x^2 + 10*y^2
    #     Gradient: [2x, 20y]
    #     Minimum: (0, 0) with f = 0
    #
    #     This function has a narrow valley along the x-axis (the y-direction
    #     is 10x steeper than the x-direction). SGD oscillates across the
    #     valley; momentum and Adam handle it better.
    #
    # Test Function 2: Rosenbrock Function (harder)
    #     f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
    #     Gradient: [-2(1-x) - 400*x*(y-x^2), 200*(y-x^2)]
    #     Minimum: (1, 1) with f = 0
    #
    #     A classic optimization test: a narrow curved valley. Easy to find
    #     the valley, hard to converge to the minimum at (1, 1).
    # =========================================================================

    np.set_printoptions(precision=6, suppress=True)

    print("=" * 75)
    print("OPTIMIZER DEMO — Gradient Descent Optimizers From Scratch")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Define test functions and their gradients
    # -------------------------------------------------------------------------

    def quadratic_bowl(params: list[np.ndarray]) -> tuple[float, list[np.ndarray]]:
        """Quadratic bowl: f(x,y) = x^2 + 10*y^2.

        This is an elliptical bowl. The y-direction is 10x steeper than
        the x-direction, creating a narrow valley along the x-axis.

        SGD struggles with this because:
        - The gradient in y is large (steep) -> SGD takes big steps in y
        - The gradient in x is small (gentle) -> SGD takes tiny steps in x
        - Result: oscillation in y, slow progress in x

        Returns:
            Tuple of (loss_value, list_of_gradients).
        """
        w = params[0]
        x, y = w[0], w[1]
        loss = x ** 2 + 10.0 * y ** 2
        grad = np.array([2.0 * x, 20.0 * y])
        return float(loss), [grad]

    def rosenbrock(params: list[np.ndarray]) -> tuple[float, list[np.ndarray]]:
        """Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2.

        The global minimum is at (1, 1) where f = 0.

        This function has a narrow, banana-shaped valley. Finding the valley
        is easy, but converging along the flat bottom to (1,1) is hard.
        It's a classic test for optimization algorithms.

        Returns:
            Tuple of (loss_value, list_of_gradients).
        """
        w = params[0]
        x, y = w[0], w[1]
        loss = (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2
        # Gradient derivation:
        #   df/dx = -2(1-x) + 100 * 2*(y-x^2) * (-2x) = -2(1-x) - 400*x*(y-x^2)
        #   df/dy = 100 * 2*(y-x^2) = 200*(y-x^2)
        grad_x = -2.0 * (1.0 - x) - 400.0 * x * (y - x ** 2)
        grad_y = 200.0 * (y - x ** 2)
        grad = np.array([grad_x, grad_y])
        return float(loss), [grad]

    # -------------------------------------------------------------------------
    # Helper to run an optimizer on a test function
    # -------------------------------------------------------------------------

    def run_optimizer(
        optimizer: Optimizer,
        objective_fn: Any,
        initial_params: list[np.ndarray],
        n_steps: int = 200,
        print_every: int = 50,
    ) -> tuple[list[float], list[np.ndarray]]:
        """Run an optimizer on an objective function and track the trajectory.

        Args:
            optimizer: The optimizer to use.
            objective_fn: Function that takes params and returns (loss, grads).
            initial_params: Starting point as list of numpy arrays.
            n_steps: Number of optimization steps.
            print_every: Print progress every N steps.

        Returns:
            Tuple of (loss_history, parameter_trajectory).
        """
        # Deep copy initial params so we don't mutate the originals
        params = [p.copy() for p in initial_params]
        optimizer.reset()

        loss_history: list[float] = []
        trajectory: list[np.ndarray] = [params[0].copy()]

        for step in range(1, n_steps + 1):
            loss, grads = objective_fn(params)
            loss_history.append(loss)
            params = optimizer.step(params, grads)
            trajectory.append(params[0].copy())

            if step % print_every == 0 or step == 1:
                print(
                    f"    Step {step:4d}  |  "
                    f"Loss: {loss:12.6f}  |  "
                    f"Params: [{params[0][0]:9.6f}, {params[0][1]:9.6f}]"
                )

        final_loss, _ = objective_fn(params)
        print(
            f"    FINAL     |  "
            f"Loss: {final_loss:12.6f}  |  "
            f"Params: [{params[0][0]:9.6f}, {params[0][1]:9.6f}]"
        )

        return loss_history, trajectory

    # =========================================================================
    # EXPERIMENT 1: Quadratic Bowl — Comparing all optimizers
    # =========================================================================

    print("\n" + "=" * 75)
    print("EXPERIMENT 1: Quadratic Bowl  f(x,y) = x^2 + 10*y^2")
    print("Starting point: (3.0, 4.0)   |   Minimum: (0.0, 0.0)")
    print("=" * 75)
    print()
    print("This function has a narrow valley (y is 10x steeper than x).")
    print("Watch how different optimizers handle the anisotropy:")
    print()

    start_point = [np.array([3.0, 4.0])]
    n_steps = 200
    results_bowl: dict[str, tuple[list[float], list[np.ndarray]]] = {}

    # --- SGD ---
    print("-" * 50)
    print("SGD (lr=0.01)")
    print("-" * 50)
    sgd = SGD(lr=0.01)
    results_bowl["SGD"] = run_optimizer(sgd, quadratic_bowl, start_point, n_steps)

    # --- SGD with higher lr (to show oscillation) ---
    print()
    print("-" * 50)
    print("SGD (lr=0.09) — Near the stability limit!")
    print("Note: lr > 0.1 would diverge (1/max_eigenvalue = 1/20 = 0.05)")
    print("-" * 50)
    sgd_fast = SGD(lr=0.09)
    results_bowl["SGD (lr=0.09)"] = run_optimizer(
        sgd_fast, quadratic_bowl, start_point, n_steps
    )

    # --- Momentum ---
    print()
    print("-" * 50)
    print("Momentum (lr=0.01, beta=0.9)")
    print("-" * 50)
    mom = Momentum(lr=0.01, beta=0.9)
    results_bowl["Momentum"] = run_optimizer(mom, quadratic_bowl, start_point, n_steps)

    # --- Nesterov Momentum ---
    print()
    print("-" * 50)
    print("Nesterov Momentum (lr=0.01, beta=0.9)")
    print("-" * 50)
    nesterov = Momentum(lr=0.01, beta=0.9, nesterov=True)
    results_bowl["Nesterov"] = run_optimizer(
        nesterov, quadratic_bowl, start_point, n_steps
    )

    # --- Adam ---
    print()
    print("-" * 50)
    print("Adam (lr=0.1)")
    print("-" * 50)
    adam = Adam(lr=0.1)
    results_bowl["Adam"] = run_optimizer(adam, quadratic_bowl, start_point, n_steps)

    # --- AdamW ---
    print()
    print("-" * 50)
    print("AdamW (lr=0.1, weight_decay=0.01)")
    print("-" * 50)
    adamw = AdamW(lr=0.1, weight_decay=0.01)
    results_bowl["AdamW"] = run_optimizer(adamw, quadratic_bowl, start_point, n_steps)

    # --- Summary ---
    print()
    print("=" * 75)
    print("EXPERIMENT 1 SUMMARY — Final loss after 200 steps")
    print("=" * 75)
    for name, (losses, _) in results_bowl.items():
        final = losses[-1]
        converged = "YES" if final < 1e-4 else "no"
        print(f"  {name:25s}  |  Final loss: {final:12.8f}  |  Converged: {converged}")

    print()
    print("KEY OBSERVATIONS:")
    print("  - SGD (lr=0.01) converges slowly — the low lr is safe but sluggish")
    print("  - SGD (lr=0.09) oscillates — high lr causes instability in the steep y-direction")
    print("  - Momentum smooths the oscillations and converges faster")
    print("  - Nesterov momentum is slightly faster than standard momentum")
    print("  - Adam adapts per-parameter learning rates, handling the anisotropy well")
    print("  - AdamW adds a slight weight decay, pulling params toward zero (helpful here!)")

    # =========================================================================
    # EXPERIMENT 2: Rosenbrock Function — A harder challenge
    # =========================================================================

    print("\n" + "=" * 75)
    print("EXPERIMENT 2: Rosenbrock  f(x,y) = (1-x)^2 + 100*(y-x^2)^2")
    print("Starting point: (-1.0, 1.0)   |   Minimum: (1.0, 1.0)")
    print("=" * 75)
    print()
    print("The Rosenbrock function has a narrow banana-shaped valley.")
    print("Finding the valley is easy; navigating to the minimum at (1,1) is hard.")
    print()

    start_point_rosen = [np.array([-1.0, 1.0])]
    n_steps_rosen = 500
    results_rosen: dict[str, tuple[list[float], list[np.ndarray]]] = {}

    # --- SGD ---
    print("-" * 50)
    print("SGD (lr=0.001)")
    print("-" * 50)
    sgd_r = SGD(lr=0.001)
    results_rosen["SGD"] = run_optimizer(
        sgd_r, rosenbrock, start_point_rosen, n_steps_rosen, print_every=100
    )

    # --- Momentum ---
    print()
    print("-" * 50)
    print("Momentum (lr=0.001, beta=0.9)")
    print("-" * 50)
    mom_r = Momentum(lr=0.001, beta=0.9)
    results_rosen["Momentum"] = run_optimizer(
        mom_r, rosenbrock, start_point_rosen, n_steps_rosen, print_every=100
    )

    # --- Adam ---
    print()
    print("-" * 50)
    print("Adam (lr=0.005)")
    print("-" * 50)
    adam_r = Adam(lr=0.005)
    results_rosen["Adam"] = run_optimizer(
        adam_r, rosenbrock, start_point_rosen, n_steps_rosen, print_every=100
    )

    # --- AdamW ---
    print()
    print("-" * 50)
    print("AdamW (lr=0.005, weight_decay=0.001)")
    print("-" * 50)
    adamw_r = AdamW(lr=0.005, weight_decay=0.001)
    results_rosen["AdamW"] = run_optimizer(
        adamw_r, rosenbrock, start_point_rosen, n_steps_rosen, print_every=100
    )

    # --- Summary ---
    print()
    print("=" * 75)
    print("EXPERIMENT 2 SUMMARY — Final loss after 500 steps on Rosenbrock")
    print("=" * 75)
    for name, (losses, _) in results_rosen.items():
        final = losses[-1]
        print(f"  {name:25s}  |  Final loss: {final:12.6f}")

    print()
    print("KEY OBSERVATIONS:")
    print("  - SGD struggles — the narrow valley causes slow progress")
    print("  - Momentum helps but can overshoot in the curved valley")
    print("  - Adam handles the varying curvature well with adaptive rates")
    print("  - Rosenbrock is genuinely hard — even Adam needs many more steps")
    print("    to fully converge to (1, 1)")

    # =========================================================================
    # EXPERIMENT 3: Effect of Learning Rate on SGD
    # =========================================================================

    print("\n" + "=" * 75)
    print("EXPERIMENT 3: How Learning Rate Affects SGD Convergence")
    print("Function: f(x,y) = x^2 + 10*y^2  |  Start: (3.0, 4.0)")
    print("=" * 75)
    print()
    print("We'll run SGD with different learning rates to see:")
    print("  - Too small: converges but very slowly")
    print("  - Just right: smooth convergence")
    print("  - Too large: oscillates or diverges")
    print()

    learning_rates = [0.001, 0.01, 0.05, 0.09, 0.105]
    start = [np.array([3.0, 4.0])]

    for lr_val in learning_rates:
        opt = SGD(lr=lr_val)
        opt.reset()
        params_lr = [start[0].copy()]
        diverged = False

        for step in range(100):
            loss_val, grad_val = quadratic_bowl(params_lr)
            if loss_val > 1e10 or np.isnan(loss_val):
                print(
                    f"  lr={lr_val:<8.4f}  |  "
                    f"DIVERGED at step {step}!  Loss exploded to {loss_val:.2e}"
                )
                diverged = True
                break
            params_lr = opt.step(params_lr, grad_val)

        if not diverged:
            final_loss_val, _ = quadratic_bowl(params_lr)
            status = "converged" if final_loss_val < 0.01 else "slow"
            print(
                f"  lr={lr_val:<8.4f}  |  "
                f"Final loss after 100 steps: {final_loss_val:12.8f}  ({status})"
            )

    print()
    print("TAKEAWAY:")
    print("  The maximum stable learning rate for SGD on f(x,y) = x^2 + 10*y^2")
    print("  is lr < 2/L where L = max eigenvalue of the Hessian = 20.")
    print("  So lr < 0.1. Above that, SGD diverges.")
    print("  This is why adaptive methods like Adam are so valuable —")
    print("  they effectively find different learning rates for each direction.")

    # =========================================================================
    # EXPERIMENT 4: Effect of Momentum Coefficient (beta)
    # =========================================================================

    print("\n" + "=" * 75)
    print("EXPERIMENT 4: How Momentum Coefficient (beta) Affects Convergence")
    print("Function: f(x,y) = x^2 + 10*y^2  |  Start: (3.0, 4.0)  |  lr=0.01")
    print("=" * 75)
    print()
    print("beta controls how much 'memory' the optimizer has:")
    print("  beta=0.0: no memory (same as SGD)")
    print("  beta=0.5: moderate memory")
    print("  beta=0.9: typical, remembers ~10 past gradients")
    print("  beta=0.99: very heavy ball, slow to change direction")
    print()

    betas = [0.0, 0.5, 0.9, 0.95, 0.99]
    start_mom = [np.array([3.0, 4.0])]

    for beta_val in betas:
        opt_m = Momentum(lr=0.01, beta=beta_val)
        opt_m.reset()
        params_m = [start_mom[0].copy()]

        for _step in range(200):
            loss_m, grad_m = quadratic_bowl(params_m)
            if loss_m > 1e10 or np.isnan(loss_m):
                break
            params_m = opt_m.step(params_m, grad_m)

        final_loss_m, _ = quadratic_bowl(params_m)
        if final_loss_m > 1e10 or np.isnan(final_loss_m):
            print(f"  beta={beta_val:<5.2f}  |  DIVERGED!")
        else:
            print(
                f"  beta={beta_val:<5.2f}  |  "
                f"Final loss: {final_loss_m:12.8f}  |  "
                f"Final params: [{params_m[0][0]:9.6f}, {params_m[0][1]:9.6f}]"
            )

    print()
    print("TAKEAWAY:")
    print("  beta=0.9 is the sweet spot for most problems.")
    print("  Too low (0.0-0.5): not enough smoothing, similar to vanilla SGD.")
    print("  Too high (0.99): the 'ball' is too heavy and overshoots the minimum,")
    print("  or takes too long to change direction when the gradient shifts.")

    # =========================================================================
    # Final Summary
    # =========================================================================

    print("\n" + "=" * 75)
    print("SUMMARY: When to Use Each Optimizer")
    print("=" * 75)
    print("""
    +----------+-----------------------------------------------------+
    | Optimizer | When to use                                         |
    +----------+-----------------------------------------------------+
    | SGD      | Learning / teaching. Simple problems.               |
    |          | Research (sometimes generalizes better than Adam).   |
    +----------+-----------------------------------------------------+
    | Momentum | CNNs on images (SGD + momentum is competitive).     |
    |          | When you have time to tune the learning rate.        |
    +----------+-----------------------------------------------------+
    | Adam     | General-purpose default. NLP tasks. GANs.           |
    |          | When you want fast convergence without much tuning.  |
    +----------+-----------------------------------------------------+
    | AdamW    | Transformers and LLMs. Any model with weight decay.  |
    |          | The STANDARD for modern large-scale training.        |
    +----------+-----------------------------------------------------+

    TYPICAL RECIPES:
      - ResNet on ImageNet:    SGD + Momentum (lr=0.1, beta=0.9, step decay)
      - BERT fine-tuning:      AdamW (lr=2e-5, wd=0.01, warmup + linear decay)
      - GPT / LLM pretraining: AdamW (lr=3e-4, wd=0.1, warmup + cosine decay)
      - Quick experiment:      Adam (lr=0.001, defaults) — just works
    """)
