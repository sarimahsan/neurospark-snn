# ⚡ NeuroSpark: Spiking Neural Network Simulator & Biological Neurodynamics Studio

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Vanilla JS](https://img.shields.io/badge/Stack-Vanilla%20JS%20%7C%20HTML5%20Canvas-yellow.svg)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero-brightgreen.svg)](package.json)
[![Neuromorphic Computing](https://img.shields.io/badge/Topic-Neuromorphic%20Computing-purple.svg)](#)

An interactive, zero-dependency browser-based **Spiking Neural Network (SNN)** simulator and biological neurodynamics playground. NeuroSpark allows you to train spiking networks using **Surrogate Gradient Backpropagation Through Time (BPTT)**, visualize real-time action potentials on neuroscience raster plots and oscilloscopes, sketch patterns on an interactive drawing pad, benchmark against standard ANNs, and explore the **Izhikevich biological neuron model**.

---

## 🎬 Video Demonstration

<div align="center">

https://github.com/neurospark-snn/assets/demo.mp4

[![Watch Demo Video](https://img.shields.io/badge/▶_Watch_Full_Demo_Video-assets%2Fdemo.mp4-00f2fe?style=for-the-badge)](https://github.com/sarimahsan/neurospark-snn/raw/main/assets/demo.mp4)

<p><em>Interactive Walkthrough: SNN Classification Lab, Pattern Drawing Studio, Biological Playground & SNN vs ANN Benchmark</em></p>

</div>

---

## 🌟 Interactive Studios & Features

### 1. 🔬 SNN Classification Lab
* **Leaky Integrate-and-Fire (LIF) Neurons**:
  $$V_i[t] = \beta \cdot V_i[t-1] \cdot (1 - S_i[t-1]) + \sum_j W_{ij} S_j[t] + b_i$$
  $$S_i[t] = \Theta(V_i[t] - V_{th})$$
* **Surrogate Gradient BPTT**: Overcomes the non-differentiable Heaviside step function using a Fast-Sigmoid surrogate derivative $\sigma'(v) = \frac{1}{(1 + \gamma |v - V_{th}|)^2}$ paired with an Adam optimizer.
* **4-Quadrant Live Visualizer**:
  1. **Network Topology Graph**: Node-link diagram showing membrane potential soma glow and animated traveling synaptic pulses.
  2. **2D Decision Boundary Heatmap**: Live spatial classification landscape for **Non-Linear XOR**, **Concentric Circles**, **Two Moons**, and **Two Spirals**.
  3. **Spike Raster Plot**: Neuroscience raster plot tracking spike ticks across Input, Hidden, and Output layers over temporal time steps $T$.
  4. **Membrane Potential Oscilloscope**: Real-time $V(t)$ trace with dashed threshold line $V_{th}$ and action potential indicators.

---

### 2. ✏️ Pattern Drawing Studio
* **8×8 Interactive Drawing Pad**: Freehand mouse or touch sketching of digits and patterns.
* **Pixel-to-Spike Rate Encoding**: Converts 2D pixel intensities into temporal Poisson/rate spike trains across $T = 16$ time steps.
* **Live In-Browser Training**: One-click training button using surrogate gradient BPTT on augmented digit templates.
* **Prediction Telemetry**: Dynamic firing frequency bars for digits $0$ through $4$, alongside a synchronized spike raster.

---

### 3. ⚡ vs 🤖 SNN vs ANN Live Benchmark
* **Direct Side-by-Side Comparison**: Trains a Spiking Neural Network (LIF) and an equivalent Standard ANN (ReLU MLP) simultaneously on the exact same dataset samples.
* **Dual Decision Boundaries**: Compares discrete spike rate-coded boundaries with continuous float ReLU boundaries.
* **Hardware & Energy Telemetry**:
  * **FLOPs (Multiply-Accumulate)** vs **SOPs (Sparse Additions)**
  * Dynamic energy estimation showing up to **$18\times$ lower energy consumption** in spiking neuromorphic mode!

---

### 4. 🧠 Izhikevich Biological Dynamics Playground
* **2D Coupled Dynamical System**:
  $$\frac{dv}{dt} = 0.04 v^2 + 5v + 140 - u + I(t)$$
  $$\frac{du}{dt} = a(bv - u)$$
  $$\text{if } v \ge +30\text{ mV} \implies v \leftarrow c,\; u \leftarrow u + d$$
* **7 Cortical Firing Presets**:
  - **Regular Spiking (RS)**: Neocortical pyramidal neurons with spike-frequency adaptation.
  - **Intrinsically Bursting (IB)**: Cortical layer 5 pyramidal cells.
  - **Chattering (CH)**: Fast rhythmic bursting interneurons.
  - **Fast Spiking (FS)**: Inhibitory basket and chandelier cells.
  - **Low-Threshold Spiking (LTS)**: Rebound burst firing.
  - **Thalamo-Cortical (TC)**: Thalamic relay neurons.
  - **Resonator (RZ)**: Subthreshold frequency oscillations.
* **Dual Real-time Oscilloscope & Phase Plane**: Plots $v(t)$ voltage waveforms and $(v \text{ vs } u)$ phase space with live $v$-nullcline and $u$-nullcline curves.

---

## 🚀 Quick Start (Zero Installation Required)

Because NeuroSpark is built with standard Vanilla ES6 JavaScript and HTML5 Canvas, it requires **zero external build tools or package installations**.

### Option A: One-Click Launch (Windows)
Double-click [`start.bat`](start.bat) to launch the local web server and open the browser automatically.

### Option B: Using Node.js
```bash
npm start
# or
node start.bat
```
Then visit **[http://localhost:8080](http://localhost:8080)**.

### Option C: Using Any Static Web Server
You can serve the repository with Python or any static file server:
```bash
python -m http.server 8080
```

---

## 📚 In-Depth Technical Documentation

Comprehensive architectural and theoretical guides are available in the [`docs/`](docs/) directory:

| Document | Description |
| :--- | :--- |
| **[docs/ANN_TO_SNN_BRIDGE.md](docs/ANN_TO_SNN_BRIDGE.md)** | Rosetta Stone guide mapping ANN, RNN, and Transformer concepts directly to SNN mechanics. |
| **[docs/SNN_LAB_EXPLAINED.md](docs/SNN_LAB_EXPLAINED.md)** | Step-by-step breakdown of the SNN Classification Lab panels, LIF equations, and BPTT math. |
| **[docs/PATTERN_STUDIO_EXPLAINED.md](docs/PATTERN_STUDIO_EXPLAINED.md)** | Complete guide to the 8×8 Drawing Studio, pixel-to-spike encoding, and raster plots. |
| **[docs/SNN_VS_ANN_BENCHMARK.md](docs/SNN_VS_ANN_BENCHMARK.md)** | Side-by-side benchmark comparison, FLOPs vs SOPs, and neuromorphic energy analysis. |
| **[docs/IZHIKEVICH_EXPLAINED.md](docs/IZHIKEVICH_EXPLAINED.md)** | Biological neuron modeling guide, ODE equations, 4 parameters ($a, b, c, d$), and phase plane nullclines. |

---

## 📁 Repository Structure

```
neurosimulator/
├── index.html               # Main application layout, viewports, and tabs
├── style.css                # Dark cyber-neurology aesthetic design system
├── package.json             # NPM metadata and launch scripts (zero dependencies)
├── LICENSE                  # Open-source MIT License
├── start.bat                # Windows one-click local server launcher
├── .gitignore               # Standard Git ignore rules
│
├── assets/                  # Media and video assets
│   └── demo.mp4             # Full walkthrough video demonstration
│
├── js/
│   ├── snn.js               # Leaky Integrate-and-Fire layers, Surrogate Gradient BPTT, Adam
│   ├── ann.js               # Equivalent standard Float32 ReLU MLP for direct benchmarking
│   ├── izhikevich.js        # Izhikevich 2D biological dynamical system & cortical presets
│   ├── datasets.js          # Non-linear 2D datasets (Circles, XOR, Moons, Spirals) & 8x8 digits
│   ├── visualizer.js        # High-DPI canvas visualizers (Topology, Raster, Oscilloscope, Phase Plane)
│   └── app.js               # Main application coordinator, training loops, and telemetry
│
└── docs/                    # In-depth theoretical and architectural guides
    ├── ANN_TO_SNN_BRIDGE.md
    ├── SNN_LAB_EXPLAINED.md
    ├── PATTERN_STUDIO_EXPLAINED.md
    ├── SNN_VS_ANN_BENCHMARK.md
    └── IZHIKEVICH_EXPLAINED.md
```

---

## 📖 Key References & Theoretical Background

1. **Izhikevich, Eugene M.** (2003). *"Simple Model of Spiking Neurons"*. IEEE Transactions on Neural Networks, 14(6), 1569-1572.
2. **Neftci, E. O., Mostafa, H., & Zenke, F.** (2019). *"Surrogate Gradient Learning in Spiking Neural Networks"*. IEEE Signal Processing Magazine, 36(6), 51-63.
3. **Gerstner, W., & Kistler, W. M.** (2002). *"Spiking Neuron Models: Single Neurons, Populations, Plasticity"*. Cambridge University Press.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — free for academic, personal, and commercial open-source use.
