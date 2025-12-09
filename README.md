# MathDS-Final

Neural Network Visualizer — a vanilla HTML/CSS/JS frontend app that builds a small fully-connected neural network, lets you set inputs/targets and training hyperparameters, and visualizes weights and loss while training with gradient descent.

Quick start

- Open `index.html` in a modern browser (Chrome/Firefox). For best results, serve the folder with a simple static server:

```bash
# from project root
python3 -m http.server 8000
# then open http://localhost:8000 in your browser
```

Features

- Configure input/hidden/output sizes and activation (None, Sigmoid, ReLU).
- Random initialization of weights/biases.
- Set learning rate and iterations per step, then Start/Pause/Step training.
- Live SVG visualization of nodes and weighted edges (color/width reflect sign/magnitude).
- Loss plot (MSE) updated live.

Files

- `index.html` — UI and SVG canvas.
- `styles.css` — styles.
- `app.js` — NN implementation, training loop, and visualization.

Want enhancements? I can add batching, multiple training examples, adjustable weight initializers, improved backprop for other loss types, or export/import network weights.
# MathDS-Final