"""
BaseModel — extends nn.Module with educational tooling.

Teaches:
  • Parameters vs Buffers — what gets gradients, what gets saved
  • Weight initialisation — why Kaiming / Xavier matter
  • Parameter counting — how big is my model?
  • Hook API — attach callbacks to any layer's forward/backward

Every model in this framework inherits from BaseModel.
"""

import torch
import torch.nn as nn
from typing import Callable


class BaseModel(nn.Module):
    """
    Adds four capabilities on top of nn.Module:
      1. parameter_count()     — how many trainable weights
      2. weight_init()         — Kaiming He initialisation for conv/linear
      3. register_activation_hook(name) — capture any layer's output
      4. register_gradient_hook(name)   — capture any layer's gradient
    """

    def __init__(self):
        super().__init__()
        self._activation_hooks: dict = {}
        self._gradient_hooks: dict = {}
        self._captured_activations: dict = {}
        self._captured_gradients: dict = {}

    # ── Parameter counting ────────────────────────────────────────────────────

    def parameter_count(self, trainable_only: bool = True) -> int:
        """
        Count parameters.

        Parameters are tensors registered with nn.Parameter (or inside
        sub-modules). They are returned by model.parameters() and saved
        in model.state_dict().

        Buffers (e.g. BatchNorm running_mean) are NOT parameters — they
        have no gradient and are not updated by the optimizer. They ARE
        saved in state_dict() because they're needed at inference time.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def parameter_summary(self) -> str:
        trainable = self.parameter_count(trainable_only=True)
        total     = self.parameter_count(trainable_only=False)
        frozen    = total - trainable
        lines = [
            f"  Trainable parameters : {trainable:>12,}",
            f"  Frozen    parameters : {frozen:>12,}",
            f"  Total     parameters : {total:>12,}",
        ]
        return "\n".join(lines)

    # ── Weight initialisation ─────────────────────────────────────────────────

    def init_weights(self) -> None:
        """
        Apply Kaiming He initialisation to conv and linear layers.

        Why Kaiming (He) init?
          • With ReLU, roughly half the neurons are zeroed each forward pass.
          • Naive initialisation (e.g. N(0,1)) causes variance to shrink
            exponentially with depth → vanishing gradients.
          • Kaiming compensates: std = sqrt(2 / fan_in), keeping variance
            stable across layers even with ReLU non-linearities.

        For BatchNorm: weight=1, bias=0 is the standard identity start.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)    # scale starts at 1 (identity)
                nn.init.zeros_(m.bias)     # shift starts at 0 (identity)

    # ── Forward activation hooks ──────────────────────────────────────────────

    def register_activation_hook(self, layer_name: str) -> None:
        """
        Capture the output tensor of a named layer during forward().

        A forward hook is a function attached to a module that fires
        AFTER that module's forward() returns, receiving (module, input, output).

        Usage:
            model.register_activation_hook("layer2")
            out = model(x)
            act = model.get_activation("layer2")   # tensor

        Common use cases:
          • Visualise feature maps
          • Build a classifier on top of a frozen backbone (linear probe)
          • Detect dying ReLUs (activations stuck at 0)
        """
        layer = dict(self.named_modules()).get(layer_name)
        if layer is None:
            raise ValueError(f"No layer named {layer_name!r}. "
                             f"Available: {list(dict(self.named_modules()).keys())}")

        def hook_fn(module, input, output):
            self._captured_activations[layer_name] = output.detach()

        handle = layer.register_forward_hook(hook_fn)
        self._activation_hooks[layer_name] = handle

    def get_activation(self, layer_name: str) -> torch.Tensor:
        if layer_name not in self._captured_activations:
            raise RuntimeError(f"No activation for {layer_name!r}. "
                               f"Did you call register_activation_hook() and run a forward pass?")
        return self._captured_activations[layer_name]

    # ── Backward gradient hooks ───────────────────────────────────────────────

    def register_gradient_hook(self, layer_name: str) -> None:
        """
        Capture the gradient flowing INTO a named layer during backward().

        A backward hook fires during backprop, receiving (module, grad_input, grad_output).
        grad_output is the gradient arriving from the layer ABOVE (upstream).

        Common use cases:
          • Debug vanishing/exploding gradients
          • Implement gradient penalty terms
          • Monitor which layers receive the strongest gradient signal
        """
        layer = dict(self.named_modules()).get(layer_name)
        if layer is None:
            raise ValueError(f"No layer named {layer_name!r}.")

        def hook_fn(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self._captured_gradients[layer_name] = grad_output[0].detach()

        handle = layer.register_full_backward_hook(hook_fn)
        self._gradient_hooks[layer_name] = handle

    def get_gradient(self, layer_name: str) -> torch.Tensor:
        if layer_name not in self._captured_gradients:
            raise RuntimeError(f"No gradient for {layer_name!r}. "
                               f"Did you call register_gradient_hook() and run backward()?")
        return self._captured_gradients[layer_name]

    def remove_hooks(self) -> None:
        """Remove all registered forward and backward hooks."""
        for h in self._activation_hooks.values():
            h.remove()
        for h in self._gradient_hooks.values():
            h.remove()
        self._activation_hooks.clear()
        self._gradient_hooks.clear()
        self._captured_activations.clear()
        self._captured_gradients.clear()
