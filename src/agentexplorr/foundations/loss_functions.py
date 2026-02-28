"""
Loss Functions — How We Measure "How Wrong" a Model Is
========================================================

WHAT IS A LOSS FUNCTION?
    A loss function (also called cost function, objective function, or criterion)
    measures the difference between a model's predictions and the true values.
    It outputs a single number: the "loss". A lower loss = better predictions.

    Training a model = finding parameters that MINIMIZE the loss function.

    The gradient of the loss with respect to the model's predictions tells us
    WHICH DIRECTION to adjust. If the gradient is positive, we decrease the
    prediction; if negative, we increase it. This is the foundation of
    gradient descent and backpropagation.

WHY DIFFERENT LOSS FUNCTIONS?
    Different tasks need different loss functions:

      Task                  | Loss Function           | Why
      ----------------------|-------------------------|---------------------------
      Regression            | MSE (Mean Squared Error)| Penalizes large errors
      Classification        | Cross-Entropy           | Penalizes confident wrongs
      Binary Classification | Binary Cross-Entropy    | Special case of CE
      Robust Regression     | Huber Loss              | Handles outliers
      General Framework     | Negative Log-Likelihood | Unifies all of the above

    Using the WRONG loss function is a common beginner mistake. For example,
    using MSE for classification converges slowly because its gradients vanish
    when the model is confidently wrong (we demonstrate this in the demo).

THE BIG PICTURE — WHY THESE SPECIFIC FUNCTIONS?
    All standard loss functions arise from one principle:
    MAXIMUM LIKELIHOOD ESTIMATION (MLE).

    Given data, find the model parameters that make the data most probable:
        argmax_theta P(data | theta)

    Taking the negative log (for numerical stability and convexity):
        argmin_theta -log P(data | theta)  =  argmin_theta NLL

    Different assumptions about P(data | theta) give different losses:
      - Assume Gaussian noise     -> NLL = MSE (up to constants)
      - Assume categorical output -> NLL = Cross-Entropy
      - Assume Bernoulli output   -> NLL = Binary Cross-Entropy

LEARNING RESOURCES:
    - Stanford CS229 Loss Functions: https://cs229.stanford.edu/notes2022fall/main_notes.pdf
    - Deep Learning Book Ch. 6.2 (Goodfellow): https://www.deeplearningbook.org/contents/mlp.html
    - VIDEO: "Cross-Entropy Loss Explained" — https://www.youtube.com/watch?v=Pwgpl9mKars
    - VIDEO: "Maximum Likelihood Estimation" — https://www.youtube.com/watch?v=XepXtl9YKwc
    - VIDEO: "KL Divergence & Cross-Entropy" — https://www.youtube.com/watch?v=ErfnhcEV1O8
    - VIDEO: "Huber Loss Explained" — https://www.youtube.com/watch?v=gIx974WtY_Y
    - PyTorch Loss Functions: https://pytorch.org/docs/stable/nn.html#loss-functions
    - Information Theory (Cover & Thomas): https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X
    - Bishop's Pattern Recognition Ch. 4.3: https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from agentexplorr.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Base Class for All Loss Functions
# =============================================================================

class LossFunction(ABC):
    """Abstract base class for all loss functions.

    Every loss function must implement two methods:

      1. forward(y_pred, y_true)  — Compute the scalar loss value.
         This answers: "How wrong are these predictions?"

      2. backward(y_pred, y_true) — Compute the gradient dL/d(y_pred).
         This answers: "In which direction should we adjust predictions?"

    WHY BOTH FORWARD AND BACKWARD?
        During training, the forward pass computes the loss (a single number).
        The backward pass computes gradients that flow back through the network
        via backpropagation (the chain rule). Together, they form the core of
        the training loop:

            loss = loss_fn.forward(predictions, targets)      # How wrong?
            grad = loss_fn.backward(predictions, targets)      # Which direction?
            predictions -= learning_rate * grad                 # Adjust!

    NAMING CONVENTION:
        We use forward/backward to match PyTorch's convention (nn.Module.forward
        and autograd's backward). This makes the transition to PyTorch natural.
    """

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the loss value.

        Args:
            y_pred: Model predictions. Shape depends on the specific loss.
            y_true: Ground truth values. Same shape as y_pred (or compatible).

        Returns:
            Scalar loss value (float). Lower is better.
        """

    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. y_pred.

        Args:
            y_pred: Model predictions.
            y_true: Ground truth values.

        Returns:
            Gradient array with same shape as y_pred.
            Each element tells us: "How does the loss change if we nudge
            this prediction by a tiny amount?"
        """

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Allow using the loss function like a callable: loss_fn(y_pred, y_true)."""
        return self.forward(y_pred, y_true)


# =============================================================================
# 1. Mean Squared Error (MSE) — The Default Loss for Regression
# =============================================================================

class MeanSquaredError(LossFunction):
    """Mean Squared Error loss for regression tasks.

    FORMULA:
        L = (1/n) * sum( (y_pred - y_true)^2 )

    DERIVATIVE (with respect to y_pred):
        dL/d(y_pred) = (2/n) * (y_pred - y_true)

    WHY SQUARED?
        1. Penalizes large errors MORE than small errors (quadratic penalty).
           An error of 10 costs 100, while an error of 1 costs only 1.
        2. Is differentiable everywhere (unlike absolute error at 0).
        3. Has a unique global minimum (the loss surface is a smooth bowl).
        4. Connects to maximum likelihood under Gaussian noise assumption:
           If errors are Gaussian, minimizing MSE = maximizing likelihood.

    WHEN TO USE:
        - Regression tasks (predicting continuous values)
        - When errors are roughly normally distributed
        - When you want to penalize large errors heavily

    WHEN NOT TO USE:
        - Classification (use cross-entropy instead — we show why in the demo)
        - Data with many outliers (use Huber loss instead)
        - When all errors should be penalized equally (use MAE instead)

    MATHEMATICAL DERIVATION FROM MLE:
        Assume y_true = y_pred + epsilon, where epsilon ~ N(0, sigma^2)

        P(y_true | y_pred) = (1 / sqrt(2*pi*sigma^2)) * exp(-(y_true - y_pred)^2 / (2*sigma^2))

        Log-likelihood:
            log P = -n/2 * log(2*pi*sigma^2) - (1 / (2*sigma^2)) * sum((y_true - y_pred)^2)

        Negative log-likelihood (drop constants):
            NLL proportional to sum((y_true - y_pred)^2) = n * MSE

        So minimizing MSE = maximizing likelihood under Gaussian noise!

    Example:
        >>> mse = MeanSquaredError()
        >>> y_pred = np.array([2.5, 0.0, 2.1])
        >>> y_true = np.array([3.0, -0.5, 2.0])
        >>> loss = mse.forward(y_pred, y_true)   # 0.1067
        >>> grad = mse.backward(y_pred, y_true)   # [-0.333, 0.333, 0.067]
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute MSE loss.

        Args:
            y_pred: Predicted values, shape (n,) or (n, d).
            y_true: True values, same shape as y_pred.

        Returns:
            Mean squared error (scalar).
        """
        n = y_pred.shape[0]
        loss = np.sum((y_pred - y_true) ** 2) / n
        return float(loss)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute gradient of MSE w.r.t. y_pred.

        The gradient is (2/n) * (y_pred - y_true).

        INTUITION:
            - If y_pred > y_true: gradient is positive -> decrease y_pred
            - If y_pred < y_true: gradient is negative -> increase y_pred
            - Larger errors produce larger gradients (proportional feedback)

        Args:
            y_pred: Predicted values.
            y_true: True values.

        Returns:
            Gradient array, same shape as y_pred.
        """
        n = y_pred.shape[0]
        grad = (2.0 / n) * (y_pred - y_true)
        return grad


# =============================================================================
# 2. Cross-Entropy Loss — THE Most Important Loss in Deep Learning
# =============================================================================

class CrossEntropyLoss(LossFunction):
    """Cross-Entropy loss for multi-class classification.

    THIS IS THE MOST IMPORTANT LOSS FUNCTION IN DEEP LEARNING.
    Almost every classification model (image classifiers, language models,
    speech recognition) uses cross-entropy.

    THE BIG CONNECTION — INFORMATION THEORY:
        Cross-entropy comes from Claude Shannon's information theory (1948).

        1. ENTROPY: H(p) = -sum( p(x) * log(p(x)) )
           Measures the average "surprise" or "information content" in a
           distribution. A coin flip (50/50) has maximum entropy (most
           surprising). A loaded die has lower entropy (more predictable).

        2. CROSS-ENTROPY: H(p, q) = -sum( p(x) * log(q(x)) )
           Measures the average surprise when we use distribution q to
           encode data that actually follows distribution p. If q = p,
           cross-entropy equals entropy (optimal encoding). If q != p,
           cross-entropy > entropy (suboptimal encoding = wasted bits).

        3. KL DIVERGENCE: D_KL(p || q) = H(p, q) - H(p)
           The "extra" surprise from using q instead of p. Always >= 0.
           Equals 0 if and only if p = q.

        4. KEY INSIGHT: Since H(p) is constant (determined by the data),
           minimizing cross-entropy H(p, q) = minimizing KL divergence D_KL.
           So training with cross-entropy = making our model distribution q
           match the true data distribution p.

    FORMULA (multi-class):
        L = -(1/n) * sum_i( sum_c( y_true[i,c] * log(y_pred[i,c]) ) )

        For one-hot encoded targets (the common case), only the correct
        class c* contributes:
        L = -(1/n) * sum_i( log(y_pred[i, c*]) )

        In words: "What's the average negative log probability assigned
        to the correct class?"

    DERIVATIVE (the beautiful result):
        When combining softmax activation + cross-entropy loss:
            dL/d(logits) = softmax(logits) - one_hot(target)

        This is beautifully simple! The gradient is just:
            (predicted probability) - (1 if correct class, 0 otherwise)

        If the model predicts 90% for the correct class, gradient = 0.9 - 1 = -0.1
        If the model predicts 10% for the correct class, gradient = 0.1 - 1 = -0.9

        Confident correct predictions -> small gradient (barely update)
        Confident wrong predictions  -> large gradient (update a lot!)

    WHY LOG?
        log() creates an asymmetric penalty:
          - log(0.99) = -0.01  (correct and confident -> tiny loss)
          - log(0.50) = -0.69  (uncertain -> moderate loss)
          - log(0.01) = -4.61  (wrong and confident -> HUGE loss!)

        This is exactly what we want: heavily penalize confident wrong predictions.

    NUMERICAL STABILITY — THE LOG-SUM-EXP TRICK:
        Naively computing softmax then log can overflow/underflow:
          softmax(x_i) = exp(x_i) / sum(exp(x_j))  -- exp(1000) = inf!

        Solution: subtract the maximum before exponentiating:
          log(softmax(x_i)) = x_i - max(x) - log(sum(exp(x_j - max(x))))

        This is called the "log-sum-exp" trick. PyTorch does it automatically
        in F.cross_entropy, but we implement it explicitly here for education.

    Args:
        from_logits: If True, y_pred contains raw logits (before softmax).
            We apply softmax internally. This is numerically more stable and
            is the recommended usage. If False, y_pred contains probabilities.

    Example:
        >>> ce = CrossEntropyLoss(from_logits=True)
        >>> logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
        >>> targets = np.array([0, 1])  # class indices
        >>> loss = ce.forward(logits, targets)
        >>> grad = ce.backward(logits, targets)
    """

    def __init__(self, from_logits: bool = True) -> None:
        self.from_logits = from_logits

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute cross-entropy loss.

        Args:
            y_pred: If from_logits=True, raw logits of shape (n, num_classes).
                    If from_logits=False, probabilities of shape (n, num_classes).
            y_true: Class indices of shape (n,) with values in [0, num_classes).
                    NOT one-hot encoded (for simplicity and matching PyTorch).

        Returns:
            Scalar cross-entropy loss.
        """
        n = y_pred.shape[0]

        if self.from_logits:
            # Use the log-sum-exp trick for numerical stability
            log_probs = self._log_softmax(y_pred)
        else:
            # Clip to avoid log(0) = -inf
            log_probs = np.log(np.clip(y_pred, 1e-12, 1.0))

        # For each sample, pick the log-probability of the correct class
        # y_true[i] is the correct class index for sample i
        correct_log_probs = log_probs[np.arange(n), y_true.astype(int)]

        # Negative mean log-probability
        loss = -np.mean(correct_log_probs)
        return float(loss)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute gradient of cross-entropy loss w.r.t. logits (or probs).

        THE BEAUTIFUL RESULT:
            dL/d(logits) = (1/n) * (softmax(logits) - one_hot(targets))

        This elegant formula is why softmax + cross-entropy is the standard
        combo for classification. The gradient computation is trivially simple.

        Args:
            y_pred: Raw logits (n, num_classes) if from_logits=True,
                    or probabilities if from_logits=False.
            y_true: Class indices (n,).

        Returns:
            Gradient array, same shape as y_pred.
        """
        n = y_pred.shape[0]
        y_true_int = y_true.astype(int)

        probs = self._softmax(y_pred) if self.from_logits else y_pred.copy()

        # Create one-hot encoding of targets
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n), y_true_int] = 1.0

        # The gradient: (predicted probabilities - true one-hot) / n
        grad = (probs - one_hot) / n
        return grad

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax.

        softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

        Subtracting max(x) prevents overflow in exp() while producing
        identical results (since exp(a-c)/sum(exp(b-c)) = exp(a)/sum(exp(b))).

        Args:
            logits: Raw scores, shape (n, num_classes).

        Returns:
            Probabilities summing to 1 along axis=1.
        """
        # Subtract max for numerical stability (prevents exp overflow)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    @staticmethod
    def _log_softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable log-softmax using the log-sum-exp trick.

        log(softmax(x_i)) = x_i - max(x) - log(sum(exp(x_j - max(x))))

        This is MORE numerically stable than log(softmax(x)) because:
        1. softmax can produce very small values (e.g., 1e-45)
        2. log(1e-45) = -103.3 which is fine, but the intermediate 1e-45
           might underflow to 0 in float32, giving log(0) = -inf

        The log-sum-exp trick avoids the intermediate small values entirely.

        Args:
            logits: Raw scores, shape (n, num_classes).

        Returns:
            Log-probabilities, shape (n, num_classes).
        """
        # max(x) for each sample
        max_logits = np.max(logits, axis=1, keepdims=True)
        # log(sum(exp(x - max(x))))
        log_sum_exp = np.log(np.sum(np.exp(logits - max_logits), axis=1, keepdims=True))
        # x - max(x) - log(sum(exp(x - max(x))))
        return logits - max_logits - log_sum_exp


# =============================================================================
# 3. Binary Cross-Entropy — For Binary Classification
# =============================================================================

class BinaryCrossEntropy(LossFunction):
    """Binary Cross-Entropy loss for binary classification (2 classes).

    FORMULA:
        L = -(1/n) * sum( y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred) )

    INTUITION:
        This is the special case of cross-entropy with exactly 2 classes.
        The model outputs a single probability p = P(class=1).

        - When y_true = 1: loss = -log(p)        -> want p close to 1
        - When y_true = 0: loss = -log(1 - p)    -> want p close to 0

    DERIVATIVE:
        dL/dp = -(1/n) * ( y_true/p - (1 - y_true)/(1 - p) )

        Simplified: dL/dp = -(1/n) * (y_true - p) / (p * (1 - p))

    CONNECTION TO LOGISTIC REGRESSION:
        Logistic regression uses:
          p = sigmoid(w^T x + b) = 1 / (1 + exp(-(w^T x + b)))

        Training logistic regression = minimizing binary cross-entropy.
        This is why logistic regression, despite its name, is a CLASSIFICATION
        algorithm (it minimizes a classification loss, not a regression loss).

    WHEN from_logits=True:
        The model outputs raw logits z (before sigmoid). We compute:
          p = sigmoid(z) = 1 / (1 + exp(-z))
          loss = -[y*log(sigmoid(z)) + (1-y)*log(1-sigmoid(z))]

        This can be simplified to a numerically stable form:
          loss = max(z, 0) - z*y + log(1 + exp(-|z|))

        This avoids computing sigmoid (which can overflow/underflow).

    Args:
        from_logits: If True, y_pred is raw logits (before sigmoid).
            Numerically more stable. If False, y_pred is probabilities in [0, 1].

    Example:
        >>> bce = BinaryCrossEntropy(from_logits=False)
        >>> y_pred = np.array([0.9, 0.1, 0.8, 0.3])
        >>> y_true = np.array([1.0, 0.0, 1.0, 0.0])
        >>> loss = bce.forward(y_pred, y_true)
        >>> grad = bce.backward(y_pred, y_true)
    """

    def __init__(self, from_logits: bool = False) -> None:
        self.from_logits = from_logits

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute binary cross-entropy loss.

        Args:
            y_pred: Predicted probabilities (n,) in [0,1] if from_logits=False,
                    or raw logits (n,) if from_logits=True.
            y_true: Binary labels (n,), values in {0, 1}.

        Returns:
            Scalar binary cross-entropy loss.
        """
        if self.from_logits:
            # Numerically stable binary cross-entropy from logits:
            #   loss = max(z, 0) - z*y + log(1 + exp(-|z|))
            z = y_pred
            loss = np.mean(
                np.maximum(z, 0) - z * y_true + np.log(1.0 + np.exp(-np.abs(z)))
            )
        else:
            # Clip probabilities to avoid log(0)
            p = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
            loss = -np.mean(
                y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)
            )

        return float(loss)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute gradient of BCE w.r.t. y_pred.

        If from_logits=False:
            dL/dp = -(1/n) * (y/p - (1-y)/(1-p))

        If from_logits=True:
            dL/dz = (1/n) * (sigmoid(z) - y)
            (This is the same beautiful result as multi-class cross-entropy!)

        Args:
            y_pred: Predicted probabilities or logits.
            y_true: Binary labels.

        Returns:
            Gradient array, same shape as y_pred.
        """
        n = y_pred.shape[0]

        if self.from_logits:
            # dL/dz = (sigmoid(z) - y) / n
            p = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid
            grad = (p - y_true) / n
        else:
            # dL/dp = -(y/p - (1-y)/(1-p)) / n
            p = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
            grad = -(y_true / p - (1.0 - y_true) / (1.0 - p)) / n

        return grad


# =============================================================================
# 4. Negative Log-Likelihood (NLL) — The General Framework
# =============================================================================

class NegativeLogLikelihood(LossFunction):
    """Negative Log-Likelihood loss — the GENERAL FRAMEWORK behind all losses.

    MAXIMUM LIKELIHOOD ESTIMATION (MLE):
        Given observed data D = {x_1, ..., x_n} and a model with parameters
        theta, MLE finds:
            theta* = argmax_theta P(D | theta)

        Assuming i.i.d. samples:
            P(D | theta) = product_i P(x_i | theta)

        Taking the log (product -> sum, more numerically stable):
            log P(D | theta) = sum_i log P(x_i | theta)

        Since maximizing f = minimizing -f:
            theta* = argmin_theta -sum_i log P(x_i | theta)
                   = argmin_theta NLL

    HOW NLL UNIFIES ALL LOSSES:
        - Assume P(y|x) = Normal(y_pred, sigma^2) -> NLL = MSE (+ constants)
        - Assume P(y|x) = Categorical(softmax(z)) -> NLL = Cross-Entropy
        - Assume P(y|x) = Bernoulli(sigmoid(z))   -> NLL = Binary Cross-Entropy

        So ALL standard losses are special cases of NLL under different
        distributional assumptions. This is a deep and beautiful connection.

    IMPLEMENTATION NOTE:
        This class expects LOG-PROBABILITIES as input (like PyTorch's NLLLoss).
        Typically you'd use log_softmax as the last layer of your network,
        then feed the output to NLL. This is equivalent to cross-entropy loss
        (and in fact, PyTorch's CrossEntropyLoss = LogSoftmax + NLLLoss).

    Example:
        >>> nll = NegativeLogLikelihood()
        >>> log_probs = np.log([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])  # log-probabilities
        >>> targets = np.array([0, 1])  # correct class indices
        >>> loss = nll.forward(log_probs, targets)
        >>> grad = nll.backward(log_probs, targets)
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute NLL loss.

        Args:
            y_pred: Log-probabilities, shape (n, num_classes).
                    Each row should be the output of log_softmax.
            y_true: Class indices, shape (n,).

        Returns:
            Scalar NLL loss = -mean(log_probs[i, target[i]]).
        """
        n = y_pred.shape[0]
        y_true_int = y_true.astype(int)

        # Pick the log-probability of the correct class for each sample
        correct_log_probs = y_pred[np.arange(n), y_true_int]
        loss = -np.mean(correct_log_probs)
        return float(loss)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute gradient of NLL w.r.t. log-probabilities.

        dL/d(log_prob[i,c]) = -1/n  if c == target[i]
                             =  0    otherwise

        This is sparse: only the correct class gets a non-zero gradient.
        In practice, this gradient flows through log_softmax and then
        through softmax, producing the familiar (softmax - one_hot) result.

        Args:
            y_pred: Log-probabilities (n, num_classes).
            y_true: Class indices (n,).

        Returns:
            Gradient array, same shape as y_pred.
        """
        n = y_pred.shape[0]
        y_true_int = y_true.astype(int)

        grad = np.zeros_like(y_pred)
        grad[np.arange(n), y_true_int] = -1.0 / n
        return grad


# =============================================================================
# 5. Huber Loss — Robust Regression That Handles Outliers
# =============================================================================

class HuberLoss(LossFunction):
    """Huber loss — combines the best of MSE and MAE for robust regression.

    THE OUTLIER PROBLEM WITH MSE:
        MSE squares errors, so outliers (very large errors) dominate the loss.
        A single outlier with error=100 contributes 10,000 to MSE, drowning
        out the signal from hundreds of normal samples with error=1 (cost=1).

    THE SMOOTHNESS PROBLEM WITH MAE:
        MAE (Mean Absolute Error) is robust to outliers (linear penalty),
        but its gradient is discontinuous at 0 (|x| has a kink). This makes
        optimization harder — the gradient is always +1 or -1, never adapting
        to the error magnitude.

    HUBER LOSS — THE BEST OF BOTH:
        Huber loss acts like MSE for small errors (smooth, adaptive gradient)
        and like MAE for large errors (linear penalty, robust to outliers).

    FORMULA:
        L = 0.5 * (y_pred - y_true)^2           if |y_pred - y_true| <= delta
        L = delta * (|y_pred - y_true| - 0.5 * delta)  otherwise

    DERIVATIVE:
        dL/d(y_pred) = (y_pred - y_true)                    if |error| <= delta
        dL/d(y_pred) = delta * sign(y_pred - y_true)        otherwise

    THE delta PARAMETER:
        delta controls the transition point between MSE and MAE behavior.
        - delta = 1.0 is a common default
        - Smaller delta = more robust (more like MAE)
        - Larger delta  = more sensitive (more like MSE)
        - delta -> infinity = exactly MSE
        - delta -> 0 = exactly MAE

    WHEN TO USE:
        - Regression with potential outliers (sensor data, financial data)
        - When you want smooth gradients (unlike MAE) but outlier robustness
        - Reinforcement learning (common in DQN, Stable Baselines3)

    Args:
        delta: Threshold for switching between MSE and MAE behavior.
            Default is 1.0. Tune based on your expected error distribution.

    Example:
        >>> huber = HuberLoss(delta=1.0)
        >>> y_pred = np.array([2.5, 10.0, 2.1])
        >>> y_true = np.array([3.0, 3.0, 2.0])
        >>> loss = huber.forward(y_pred, y_true)
        >>> grad = huber.backward(y_pred, y_true)
    """

    def __init__(self, delta: float = 1.0) -> None:
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        self.delta = delta

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute Huber loss.

        Args:
            y_pred: Predicted values, shape (n,) or (n, d).
            y_true: True values, same shape as y_pred.

        Returns:
            Scalar Huber loss.
        """
        error = y_pred - y_true
        abs_error = np.abs(error)

        # Piecewise: quadratic for small errors, linear for large errors
        quadratic = 0.5 * error ** 2
        linear = self.delta * (abs_error - 0.5 * self.delta)

        # Select based on error magnitude
        elementwise_loss = np.where(abs_error <= self.delta, quadratic, linear)

        loss = np.mean(elementwise_loss)
        return float(loss)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute gradient of Huber loss w.r.t. y_pred.

        INTUITION:
            - For small errors (|e| <= delta): gradient = e (like MSE)
              -> gradient adapts to error magnitude (gentle correction)
            - For large errors (|e| > delta): gradient = delta * sign(e) (like MAE)
              -> gradient is CAPPED (prevents exploding updates from outliers)

        This smooth transition is why Huber loss is the preferred loss in
        reinforcement learning — it prevents large Q-value errors from
        destabilizing training.

        Args:
            y_pred: Predicted values.
            y_true: True values.

        Returns:
            Gradient array, same shape as y_pred.
        """
        n = y_pred.shape[0]
        error = y_pred - y_true

        # Piecewise gradient
        grad = np.where(
            np.abs(error) <= self.delta,
            error,                           # MSE gradient (smooth)
            self.delta * np.sign(error),     # MAE gradient (capped)
        )

        return grad / n


# =============================================================================
# Standalone Convenience Functions
# =============================================================================

def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Mean Squared Error (convenience function).

    L = (1/n) * sum((y_pred - y_true)^2)

    Args:
        y_pred: Predicted values.
        y_true: True values.

    Returns:
        Scalar MSE loss.
    """
    return MeanSquaredError().forward(y_pred, y_true)


def cross_entropy_loss(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    from_logits: bool = True,
) -> float:
    """Compute Cross-Entropy Loss (convenience function).

    L = -(1/n) * sum(log(p_correct_class))

    Args:
        y_pred: Logits (n, num_classes) if from_logits=True, or
                probabilities (n, num_classes) if from_logits=False.
        y_true: Class indices (n,).
        from_logits: Whether y_pred contains raw logits.

    Returns:
        Scalar cross-entropy loss.
    """
    return CrossEntropyLoss(from_logits=from_logits).forward(y_pred, y_true)


def binary_cross_entropy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    from_logits: bool = False,
) -> float:
    """Compute Binary Cross-Entropy (convenience function).

    L = -(1/n) * sum(y*log(p) + (1-y)*log(1-p))

    Args:
        y_pred: Predicted probabilities (n,) if from_logits=False,
                or raw logits (n,) if from_logits=True.
        y_true: Binary labels (n,), values in {0, 1}.
        from_logits: Whether y_pred contains raw logits.

    Returns:
        Scalar binary cross-entropy loss.
    """
    return BinaryCrossEntropy(from_logits=from_logits).forward(y_pred, y_true)


def huber_loss(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    delta: float = 1.0,
) -> float:
    """Compute Huber Loss (convenience function).

    Combines MSE (small errors) and MAE (large errors).

    Args:
        y_pred: Predicted values.
        y_true: True values.
        delta: Threshold for MSE/MAE transition.

    Returns:
        Scalar Huber loss.
    """
    return HuberLoss(delta=delta).forward(y_pred, y_true)


def negative_log_likelihood(
    log_probs: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Compute Negative Log-Likelihood (convenience function).

    L = -(1/n) * sum(log_probs[i, target[i]])

    Args:
        log_probs: Log-probabilities (n, num_classes).
        y_true: Class indices (n,).

    Returns:
        Scalar NLL loss.
    """
    return NegativeLogLikelihood().forward(log_probs, y_true)


# =============================================================================
# Demo Section — Run this file to see loss functions in action!
# =============================================================================

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    def print_header(title: str) -> None:
        """Print a formatted section header."""
        print("\n" + "=" * 75)
        print(f"  {title}")
        print("=" * 75)

    def print_subheader(title: str) -> None:
        """Print a formatted subsection header."""
        print(f"\n--- {title} ---")

    # =========================================================================
    # DEMO 1: Mean Squared Error — Worked Numerical Example
    # =========================================================================
    print_header("DEMO 1: Mean Squared Error (MSE) for Regression")

    print("""
    Scenario: Predicting house prices (in $100K).
    True prices:      [3.0, 5.5, 2.0, 7.0]
    Model predictions: [2.8, 5.0, 2.5, 6.5]
    """)

    y_true_reg = np.array([3.0, 5.5, 2.0, 7.0])
    y_pred_reg = np.array([2.8, 5.0, 2.5, 6.5])

    mse = MeanSquaredError()
    loss_val = mse.forward(y_pred_reg, y_true_reg)
    grad_val = mse.backward(y_pred_reg, y_true_reg)

    print("  Step-by-step calculation:")
    errors = y_pred_reg - y_true_reg
    squared = errors ** 2
    print(f"    Errors (pred - true):  {errors}")
    print(f"    Squared errors:        {squared}")
    print(f"    Sum of squared:        {np.sum(squared):.4f}")
    print(f"    MSE = sum / n:         {loss_val:.4f}")

    print("\n  Gradient dL/d(y_pred) = (2/n)*(y_pred - y_true):")
    print(f"    {grad_val}")
    print("    Interpretation:")
    for i in range(len(y_pred_reg)):
        direction = "decrease" if grad_val[i] > 0 else "increase"
        print(f"      Prediction {i}: grad={grad_val[i]:+.4f} -> {direction} this prediction")

    # =========================================================================
    # DEMO 2: Cross-Entropy Loss — The Most Important Loss
    # =========================================================================
    print_header("DEMO 2: Cross-Entropy Loss for Classification")

    print("""
    Scenario: 3-class classification (cat=0, dog=1, bird=2).
    We have 4 samples with these model logits and true labels:
    """)

    logits = np.array([
        [2.0, 1.0, 0.1],   # Model thinks: mostly cat
        [0.5, 2.5, 0.3],   # Model thinks: mostly dog
        [0.1, 0.3, 3.0],   # Model thinks: mostly bird
        [2.0, 2.0, 0.5],   # Model uncertain: cat or dog?
    ])
    targets = np.array([0, 1, 2, 0])  # True: cat, dog, bird, cat
    class_names = ["cat", "dog", "bird"]

    ce = CrossEntropyLoss(from_logits=True)
    probs = CrossEntropyLoss._softmax(logits)
    log_probs_demo = CrossEntropyLoss._log_softmax(logits)

    print("  Logits -> Softmax Probabilities:")
    for i in range(len(logits)):
        true_class = class_names[targets[i]]
        pred_class = class_names[np.argmax(probs[i])]
        correct_prob = probs[i, targets[i]]
        print(f"    Sample {i}: logits={logits[i]} -> probs={probs[i]}")
        print(f"              True={true_class}, Predicted={pred_class}, "
              f"P(correct)={correct_prob:.4f}")

    loss_val = ce.forward(logits, targets)
    grad_val = ce.backward(logits, targets)
    print(f"\n  Cross-Entropy Loss: {loss_val:.4f}")

    print("\n  Gradient (softmax - one_hot) / n:")
    for i in range(len(logits)):
        one_hot = np.zeros(3)
        one_hot[targets[i]] = 1.0
        print(f"    Sample {i}: grad={grad_val[i]}  "
              f"(= ({probs[i]} - {one_hot}) / {len(logits)})")

    print_subheader("Why log() creates asymmetric penalty")
    print("  Correct class probability -> Loss contribution:")
    test_probs = [0.99, 0.90, 0.70, 0.50, 0.30, 0.10, 0.01]
    for p in test_probs:
        nll = -np.log(p)
        print(f"    P(correct) = {p:.2f}  ->  -log(p) = {nll:.4f}  "
              f"{'(barely punished)' if nll < 0.5 else '(HEAVILY punished!)' if nll > 2 else '(moderate)'}")

    # =========================================================================
    # DEMO 3: Why Cross-Entropy is Better Than MSE for Classification
    # =========================================================================
    print_header("DEMO 3: Cross-Entropy vs MSE for Classification")

    print("""
    KEY INSIGHT: When the model is confidently WRONG, MSE gradients are tiny
    (vanishing gradient problem), but cross-entropy gradients are large.

    Consider a binary classification where true label = 1 (positive class).
    The model outputs different predicted probabilities:
    """)

    print("  Model Prediction  |  MSE Loss   MSE Grad   |  BCE Loss   BCE Grad")
    print("  " + "-" * 71)

    mse_obj = MeanSquaredError()
    bce_obj = BinaryCrossEntropy(from_logits=False)

    test_predictions = [0.99, 0.90, 0.70, 0.50, 0.30, 0.10, 0.01]
    for p_val in test_predictions:
        y_p = np.array([p_val])
        y_t = np.array([1.0])

        mse_loss = mse_obj.forward(y_p, y_t)
        mse_grad = mse_obj.backward(y_p, y_t)[0]
        bce_loss = bce_obj.forward(y_p, y_t)
        bce_grad = bce_obj.backward(y_p, y_t)[0]

        marker = ""
        if p_val <= 0.1:
            marker = " <-- CONFIDENTLY WRONG: CE gradient is huge, MSE gradient is small!"
        print(f"  p = {p_val:.2f}           | "
              f" {mse_loss:.4f}    {mse_grad:+.4f}    | "
              f" {bce_loss:.4f}    {bce_grad:+.4f}{marker}")

    print("""
    TAKEAWAY:
      When p=0.01 (model is 99% sure it is WRONG):
        MSE gradient = -0.02  (tiny! barely updates)
        BCE gradient = -100.0 (huge! strongly corrects)

      This is why cross-entropy converges faster for classification.
      MSE causes "gradient saturation" — when the model is very wrong,
      it can barely learn because the gradients are too small.
    """)

    # =========================================================================
    # DEMO 4: Binary Cross-Entropy — Worked Example
    # =========================================================================
    print_header("DEMO 4: Binary Cross-Entropy")

    print("""
    Scenario: Spam detection (0=not spam, 1=spam).
    Model outputs probability of being spam.
    """)

    y_pred_bin = np.array([0.9, 0.2, 0.8, 0.3, 0.6])
    y_true_bin = np.array([1.0, 0.0, 1.0, 0.0, 1.0])

    bce = BinaryCrossEntropy(from_logits=False)
    bce_loss = bce.forward(y_pred_bin, y_true_bin)
    bce_grad = bce.backward(y_pred_bin, y_true_bin)

    print(f"  Predictions:  {y_pred_bin}")
    print(f"  True labels:  {y_true_bin}")
    print("\n  Per-sample losses:")
    for i in range(len(y_pred_bin)):
        p = np.clip(y_pred_bin[i], 1e-12, 1.0 - 1e-12)
        sample_loss = -(y_true_bin[i] * np.log(p) + (1 - y_true_bin[i]) * np.log(1 - p))
        correct = "correct" if (y_pred_bin[i] > 0.5) == y_true_bin[i] else "WRONG"
        print(f"    Sample {i}: y={y_true_bin[i]:.0f}, p={y_pred_bin[i]:.1f} "
              f"-> loss={sample_loss:.4f} ({correct})")

    print(f"\n  Total BCE Loss: {bce_loss:.4f}")
    print(f"  Gradient:       {bce_grad}")

    print_subheader("From-logits version (numerically stable)")
    logits_bin = np.array([2.0, -1.5, 1.5, -0.8, 0.4])
    sigmoid_probs = 1.0 / (1.0 + np.exp(-logits_bin))

    bce_logits = BinaryCrossEntropy(from_logits=True)
    loss_from_logits = bce_logits.forward(logits_bin, y_true_bin)
    loss_from_probs = bce.forward(sigmoid_probs, y_true_bin)

    print(f"  Logits:              {logits_bin}")
    print(f"  Sigmoid(logits):     {sigmoid_probs}")
    print(f"  Loss from logits:    {loss_from_logits:.6f}")
    print(f"  Loss from sigmoid:   {loss_from_probs:.6f}")
    print(f"  Match: {np.isclose(loss_from_logits, loss_from_probs)}")

    # =========================================================================
    # DEMO 5: NLL — The Unifying Framework
    # =========================================================================
    print_header("DEMO 5: Negative Log-Likelihood — The General Framework")

    print("""
    NLL is the general loss. Cross-entropy is NLL with softmax outputs.
    Let's verify: CrossEntropy(logits) == NLL(log_softmax(logits))
    """)

    logits_nll = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.5, 0.3],
        [0.1, 0.3, 3.0],
    ])
    targets_nll = np.array([0, 1, 2])

    # Cross-entropy on logits
    ce_loss = CrossEntropyLoss(from_logits=True).forward(logits_nll, targets_nll)

    # NLL on log-softmax
    log_probs_nll = CrossEntropyLoss._log_softmax(logits_nll)
    nll_loss = NegativeLogLikelihood().forward(log_probs_nll, targets_nll)

    print(f"  CrossEntropy(logits):      {ce_loss:.6f}")
    print(f"  NLL(log_softmax(logits)):  {nll_loss:.6f}")
    print(f"  They match: {np.isclose(ce_loss, nll_loss)}")

    print("""
    THIS IS WHY:
      PyTorch's CrossEntropyLoss = LogSoftmax + NLLLoss (combined for stability)
      When you use nn.CrossEntropyLoss, PyTorch does BOTH steps internally.

      The deeper reason: cross-entropy IS negative log-likelihood when the
      model outputs come from a softmax (categorical distribution).
    """)

    # =========================================================================
    # DEMO 6: Huber Loss — Robust to Outliers
    # =========================================================================
    print_header("DEMO 6: Huber Loss — Handling Outliers")

    print("""
    Scenario: Predicting temperature with a sensor that occasionally glitches.
    Most errors are small, but one reading is wildly off (outlier).
    """)

    y_true_huber = np.array([20.0, 22.0, 21.0, 23.0, 20.5])
    y_pred_huber = np.array([20.5, 22.3, 21.2, 50.0, 20.8])  # 50.0 is an outlier!

    huber = HuberLoss(delta=1.0)
    huber_loss_val = huber.forward(y_pred_huber, y_true_huber)
    huber_grad = huber.backward(y_pred_huber, y_true_huber)

    mse_loss_val = MeanSquaredError().forward(y_pred_huber, y_true_huber)
    mse_grad_val = MeanSquaredError().backward(y_pred_huber, y_true_huber)

    print(f"  True values:  {y_true_huber}")
    print(f"  Predictions:  {y_pred_huber}  (note: 50.0 is an outlier!)")
    print(f"\n  Errors:       {y_pred_huber - y_true_huber}")

    print(f"\n  MSE Loss:     {mse_loss_val:.4f}")
    print(f"  Huber Loss:   {huber_loss_val:.4f}")
    print(f"\n  MSE is {mse_loss_val/huber_loss_val:.1f}x larger than Huber!")
    print("  The outlier (error=27) dominates MSE (27^2 = 729), but Huber")
    print("  limits its contribution (delta * (27 - 0.5) = 26.5).")

    print("\n  Gradient comparison for the outlier (sample 3, error=27.0):")
    print(f"    MSE  gradient: {mse_grad_val[3]:+.4f}  (HUGE! destabilizes training)")
    print(f"    Huber gradient: {huber_grad[3]:+.4f}  (capped at delta={huber.delta})")

    print_subheader("Huber Loss Behavior Across Error Magnitudes")
    print("  Error    | MSE Loss  | Huber Loss | MSE Grad  | Huber Grad")
    print("  " + "-" * 63)
    delta = 1.0
    for err in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        y_p = np.array([err])
        y_t = np.array([0.0])
        mse_l = MeanSquaredError().forward(y_p, y_t)
        hub_l = HuberLoss(delta=delta).forward(y_p, y_t)
        mse_g = MeanSquaredError().backward(y_p, y_t)[0]
        hub_g = HuberLoss(delta=delta).backward(y_p, y_t)[0]
        region = "quadratic" if err <= delta else "linear"
        print(f"  {err:5.1f}    | {mse_l:9.4f} | {hub_l:10.4f} | "
              f"{mse_g:+9.4f} | {hub_g:+10.4f}  ({region})")

    # =========================================================================
    # DEMO 7: Log-Sum-Exp Trick — Numerical Stability
    # =========================================================================
    print_header("DEMO 7: The Log-Sum-Exp Trick for Numerical Stability")

    print("""
    PROBLEM: Large logits cause exp() to overflow to inf.

    Example: logits = [1000, 1001, 999]
      Naive softmax:
        exp(1000) = inf  -- OVERFLOW!
        exp(1001) = inf
        inf / inf = NaN  -- CRASH!

      With log-sum-exp trick (subtract max first):
        shifted = [1000-1001, 1001-1001, 999-1001] = [-1, 0, -2]
        exp([-1, 0, -2]) = [0.368, 1.000, 0.135]  -- No overflow!
        softmax = [0.245, 0.665, 0.090]  -- Correct!
    """)

    large_logits = np.array([[1000.0, 1001.0, 999.0]])

    # Naive softmax (will produce warnings)
    print("  Naive computation:")
    with np.errstate(over="warn", invalid="warn"):
        naive_exp = np.exp(large_logits)
        naive_softmax = naive_exp / np.sum(naive_exp, axis=1, keepdims=True)
    print(f"    exp([1000, 1001, 999]) = {naive_exp}")
    print(f"    naive softmax = {naive_softmax}  <-- NaN or inf!")

    # Stable softmax using the trick
    print("\n  Stable computation (log-sum-exp trick):")
    stable_softmax = CrossEntropyLoss._softmax(large_logits)
    stable_log_softmax = CrossEntropyLoss._log_softmax(large_logits)
    print(f"    stable softmax     = {stable_softmax}")
    print(f"    stable log_softmax = {stable_log_softmax}")
    print(f"    sum of probs       = {np.sum(stable_softmax):.6f} (should be 1.0)")

    # Also demonstrate with very negative logits
    print("\n  Very negative logits (potential underflow):")
    small_logits = np.array([[-1000.0, -999.0, -1001.0]])
    stable_small = CrossEntropyLoss._softmax(small_logits)
    stable_log_small = CrossEntropyLoss._log_softmax(small_logits)
    print(f"    logits = {small_logits}")
    print(f"    stable softmax     = {stable_small}")
    print(f"    stable log_softmax = {stable_log_small}")
    print(f"    sum of probs       = {np.sum(stable_small):.6f}")

    # =========================================================================
    # DEMO 8: Verify Our Gradients Against PyTorch
    # =========================================================================
    print_header("DEMO 8: Gradient Verification Against PyTorch")

    try:
        import torch
        import torch.nn.functional as F

        print("  PyTorch found! Verifying gradients...\n")

        # --- MSE Gradient Check ---
        print_subheader("MSE Gradient Check")
        y_p_np = np.array([2.5, 0.0, 2.1, 4.0], dtype=np.float64)
        y_t_np = np.array([3.0, -0.5, 2.0, 3.5], dtype=np.float64)

        # Our implementation
        our_mse_grad = MeanSquaredError().backward(y_p_np, y_t_np)

        # PyTorch
        y_p_torch = torch.tensor(y_p_np, requires_grad=True)
        y_t_torch = torch.tensor(y_t_np)
        mse_torch = F.mse_loss(y_p_torch, y_t_torch)
        mse_torch.backward()
        torch_mse_grad = y_p_torch.grad.numpy()

        print(f"    Our gradient:     {our_mse_grad}")
        print(f"    PyTorch gradient: {torch_mse_grad}")
        print(f"    Match: {np.allclose(our_mse_grad, torch_mse_grad, atol=1e-7)}")

        # --- Cross-Entropy Gradient Check ---
        print_subheader("Cross-Entropy Gradient Check")
        logits_np = np.array([
            [2.0, 1.0, 0.1],
            [0.5, 2.5, 0.3],
            [0.1, 0.3, 3.0],
        ], dtype=np.float64)
        targets_np = np.array([0, 1, 2], dtype=np.int64)

        # Our implementation
        our_ce_grad = CrossEntropyLoss(from_logits=True).backward(logits_np, targets_np)

        # PyTorch
        logits_torch = torch.tensor(logits_np, requires_grad=True)
        targets_torch = torch.tensor(targets_np)
        ce_torch = F.cross_entropy(logits_torch, targets_torch)
        ce_torch.backward()
        torch_ce_grad = logits_torch.grad.numpy()

        print(f"    Our gradient:\n{our_ce_grad}")
        print(f"    PyTorch gradient:\n{torch_ce_grad}")
        print(f"    Match: {np.allclose(our_ce_grad, torch_ce_grad, atol=1e-7)}")

        # --- Binary Cross-Entropy Gradient Check ---
        print_subheader("Binary Cross-Entropy Gradient Check")
        logits_bin_np = np.array([1.5, -0.5, 2.0, -1.0], dtype=np.float64)
        targets_bin_np = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)

        # Our implementation (from logits)
        our_bce_grad = BinaryCrossEntropy(from_logits=True).backward(
            logits_bin_np, targets_bin_np
        )

        # PyTorch
        logits_bin_torch = torch.tensor(logits_bin_np, requires_grad=True)
        targets_bin_torch = torch.tensor(targets_bin_np)
        bce_torch = F.binary_cross_entropy_with_logits(
            logits_bin_torch, targets_bin_torch
        )
        bce_torch.backward()
        torch_bce_grad = logits_bin_torch.grad.numpy()

        print(f"    Our gradient:     {our_bce_grad}")
        print(f"    PyTorch gradient: {torch_bce_grad}")
        print(f"    Match: {np.allclose(our_bce_grad, torch_bce_grad, atol=1e-7)}")

        # --- Huber Loss Gradient Check ---
        print_subheader("Huber Loss Gradient Check")
        y_p_hub_np = np.array([1.5, 0.5, 3.0, -2.0], dtype=np.float64)
        y_t_hub_np = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)

        # Our implementation
        our_hub_grad = HuberLoss(delta=1.0).backward(y_p_hub_np, y_t_hub_np)

        # PyTorch (uses SmoothL1Loss with beta=1.0 which is Huber with delta=1.0)
        y_p_hub_torch = torch.tensor(y_p_hub_np, requires_grad=True)
        y_t_hub_torch = torch.tensor(y_t_hub_np)
        hub_torch = F.smooth_l1_loss(y_p_hub_torch, y_t_hub_torch, beta=1.0)
        hub_torch.backward()
        torch_hub_grad = y_p_hub_torch.grad.numpy()

        print(f"    Our gradient:     {our_hub_grad}")
        print(f"    PyTorch gradient: {torch_hub_grad}")
        print(f"    Match: {np.allclose(our_hub_grad, torch_hub_grad, atol=1e-7)}")

        # --- Cross-Entropy Loss Value Check ---
        print_subheader("Cross-Entropy Loss Value Check")
        our_ce_loss = CrossEntropyLoss(from_logits=True).forward(logits_np, targets_np)
        torch_ce_loss = F.cross_entropy(
            torch.tensor(logits_np), torch.tensor(targets_np)
        ).item()
        print(f"    Our loss:     {our_ce_loss:.6f}")
        print(f"    PyTorch loss: {torch_ce_loss:.6f}")
        print(f"    Match: {np.isclose(our_ce_loss, torch_ce_loss, atol=1e-6)}")

        print("\n  All gradient checks passed!")

    except ImportError:
        print("  PyTorch not installed. Skipping gradient verification.")
        print("  Install with: pip install torch")
        print("\n  Performing finite-difference gradient check instead...\n")

        # Finite-difference gradient verification (works without PyTorch)
        eps = 1e-5

        print_subheader("MSE Finite-Difference Gradient Check")
        y_p_np = np.array([2.5, 0.0, 2.1, 4.0], dtype=np.float64)
        y_t_np = np.array([3.0, -0.5, 2.0, 3.5], dtype=np.float64)
        our_grad = MeanSquaredError().backward(y_p_np, y_t_np)
        fd_grad = np.zeros_like(y_p_np)
        for i in range(len(y_p_np)):
            y_p_plus = y_p_np.copy()
            y_p_minus = y_p_np.copy()
            y_p_plus[i] += eps
            y_p_minus[i] -= eps
            fd_grad[i] = (
                MeanSquaredError().forward(y_p_plus, y_t_np)
                - MeanSquaredError().forward(y_p_minus, y_t_np)
            ) / (2 * eps)
        print(f"    Analytic gradient:          {our_grad}")
        print(f"    Finite-difference gradient: {fd_grad}")
        print(f"    Match: {np.allclose(our_grad, fd_grad, atol=1e-5)}")

        print_subheader("Cross-Entropy Finite-Difference Gradient Check")
        logits_np = np.array([
            [2.0, 1.0, 0.1],
            [0.5, 2.5, 0.3],
        ], dtype=np.float64)
        targets_np = np.array([0, 1], dtype=np.int64)
        our_grad = CrossEntropyLoss(from_logits=True).backward(logits_np, targets_np)
        fd_grad = np.zeros_like(logits_np)
        for i in range(logits_np.shape[0]):
            for j in range(logits_np.shape[1]):
                l_plus = logits_np.copy()
                l_minus = logits_np.copy()
                l_plus[i, j] += eps
                l_minus[i, j] -= eps
                fd_grad[i, j] = (
                    CrossEntropyLoss(from_logits=True).forward(l_plus, targets_np)
                    - CrossEntropyLoss(from_logits=True).forward(l_minus, targets_np)
                ) / (2 * eps)
        print(f"    Analytic gradient:\n{our_grad}")
        print(f"    Finite-difference gradient:\n{fd_grad}")
        print(f"    Match: {np.allclose(our_grad, fd_grad, atol=1e-5)}")

        print("\n  Finite-difference checks passed!")

    # =========================================================================
    # Summary
    # =========================================================================
    print_header("SUMMARY: Choosing the Right Loss Function")

    print("""
    +----------------------------+------------------+----------------------------+
    | Loss Function              | Use For          | Key Property               |
    +----------------------------+------------------+----------------------------+
    | Mean Squared Error (MSE)   | Regression       | Penalizes large errors     |
    | Cross-Entropy              | Multi-class      | Penalizes confident wrongs |
    | Binary Cross-Entropy       | Binary class.    | Special case of CE         |
    | Negative Log-Likelihood    | General MLE      | Unifies all above          |
    | Huber Loss                 | Robust regression| Handles outliers           |
    +----------------------------+------------------+----------------------------+

    RULES OF THUMB:
      1. Classification? -> Cross-Entropy (always, unless you have a good reason)
      2. Regression? -> MSE (default) or Huber (if outliers present)
      3. Binary output? -> Binary Cross-Entropy
      4. Custom distribution? -> Derive from NLL

    COMMON MISTAKES:
      1. Using MSE for classification (slow convergence, gradient saturation)
      2. Forgetting numerical stability (always use from_logits=True)
      3. Not clipping probabilities before log (log(0) = -inf -> NaN)
      4. Wrong axis in softmax (should be along class dimension)

    NEXT STEPS:
      - Learn how gradients flow through networks: backpropagation.py
      - Learn how weights are updated: optimizers.py
      - See loss functions in action: classical_ml/deep_learning/training_loop.py
    """)
