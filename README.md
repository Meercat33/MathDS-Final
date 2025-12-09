
# MathDS-Final

GitHub Pages: https://Meercat33.github.io/MathDS-Final

About

This is a small, client-side Neural Network Visualizer (vanilla HTML/CSS/JS). Build a tiny fully-connected network, set inputs/targets and hyperparameters, and watch weights, activations, and loss update live while training with basic gradient descent.

Features

- Build network: set input size, hidden layer sizes (comma-separated), and output size.
- Choose activation: `None`, `Sigmoid`, or `ReLU` (applies to hidden layers; output behavior depends on chosen activation).
- Random initialization (integer-based, scaled) and reset.
- Training controls: learning rate, iterations-per-step, viz delay, Start/Pause/Step.
- Live SVG network diagram: nodes (show activations/inputs/outputs), edges with weight values, and a loss plot (MSE).
- Click an edge to inspect its weights and bias in the inspector panel.

Recommended parameter boundaries (for stable, visible training)

- Initial weights/biases: scaled integers in approximately [-1, 1] (the app uses integer init × 0.1). Smaller magnitudes reduce numerical overflow.
- Learning rate: try `0.01` or `0.001` for regression with linear outputs; `0.1` can work for small networks with bounded activations (sigmoid), but may cause instability for deep nets or large targets.
- Iterations per step: `1–10` — larger values update the model faster but make the viz jumpy.
- Viz delay (ms): `0` for smooth RAF updates, `100–500` ms to slow and observe individual steps.
- Targets: when using `Sigmoid` outputs, keep targets in `[0,1]`. For targets outside that range use `None` (linear) or `ReLU` (non-negative) outputs — but be cautious: linear outputs are unbounded and may require smaller learning rates or gradient clipping.
- Input `x0`: the first input is treated as a constant `1` (bias input) for alignment with bias-based models.

Stability tips

- If loss explodes or becomes `NaN`: reduce the learning rate, lower initial weight scale, or set `Iterations per step` to `1` and add a small `Viz delay` to watch updates.
- Consider gradient clipping or L2 regularization for more robust training (not enabled by default).

Files

- `index.html` — main UI and controls
- `styles.css` — app styles
- `app.js` — neural network code, training loop, and visualization

If you want, I can add per-layer activation controls, gradient clipping, alternative optimizers (Adam/momentum), or support for multiple training examples/batches.
# MathDS-Final