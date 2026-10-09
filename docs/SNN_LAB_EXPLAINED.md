# 🔬 SNN Classification Lab — Detailed Guide & Architecture

This guide explains **how the Spiking Neural Network (SNN) Classification Lab works**, the underlying mathematics, and **what each visual element displays one by one**.

---

## 1. Core Concept: Why Spiking Neurons?

In conventional Artificial Neural Networks (ANNs), neurons output a continuous floating-point number (e.g., `0.742`) via continuous activation functions like ReLU or Sigmoid:
$$y = \text{ReLU}(W \cdot x + b)$$

In contrast, **Spiking Neural Networks (SNNs)** emulate biological brain physiology:
1. Neurons do **not** transmit continuous values. They communicate strictly using **discrete binary events called action potentials or spikes** ($S \in \{0, 1\}$).
2. Information is processed **across time** through temporal dynamics ($t = 0, 1, 2, \dots, T-1$).
3. Neurons have a **membrane potential** ($V$) that accumulates incoming charge, leaks over time, fires when reaching a threshold, and resets.

---

## 2. Mathematical Dynamics: Step-by-Step

Every Leaky Integrate-and-Fire (LIF) neuron follows these equations at every discrete time step $t$:

### Step 2.1: Synaptic Integration
Incoming spikes from pre-synaptic neurons are multiplied by synaptic weights $W$ and summed:
$$I_i[t] = \sum_{j} W_{ij} \cdot S_j[t] + b_i$$

### Step 2.2: Leaky Membrane Potential Update
If the neuron did **not** spike in the previous time step ($S[t-1] = 0$), its membrane potential retains a decayed fraction $\beta$ of its previous charge plus the new incoming current $I[t]$:
$$V_i[t] = \beta \cdot V_i[t-1] \cdot (1 - S_i[t-1]) + I_i[t]$$
* **$\beta$ (Decay parameter)**: Models membrane resistance and capacitance ($\tau_m$). Typically $0.80 - 0.95$. A smaller $\beta$ leaks charge faster.

### Step 2.3: Thresholding & Spike Generation
The neuron compares its membrane potential against threshold $V_{th}$:
$$S_i[t] = \Theta(V_i[t] - V_{th}) = \begin{cases} 1 & \text{if } V_i[t] \ge V_{th} \\ 0 & \text{if } V_i[t] < V_{th} \end{cases}$$
* When $S_i[t] = 1$, the neuron **discharges a spike** to all connected downstream neurons.
* The membrane potential immediately resets back to $V_{reset} = 0.0\text{V}$ on the next step.

---

## 3. How Training Works: Surrogate Gradient BPTT

Training an SNN using backpropagation is historically difficult because the derivative of the Heaviside step function $\Theta(x)$ is the Dirac delta function $\delta(x)$:
$$\frac{d\Theta}{dx} = 0 \quad (\forall x \neq 0), \quad \infty \quad (x = 0)$$

Because this derivative is zero almost everywhere, standard gradient descent cannot update the weights.

### The Solution: Fast-Sigmoid Surrogate Derivative
During the backward pass (Backpropagation Through Time, or BPTT), we replace the non-differentiable step with a smooth **Fast-Sigmoid surrogate gradient**:
$$\sigma'(v) = \frac{1}{(1 + \gamma \cdot |v - V_{th}|)^2}$$

This provides a continuous gradient window around the firing threshold $V_{th}$, allowing error gradients to propagate smoothly backwards through all time steps $T$ to update synaptic weights via the **Adam optimizer**.

---

## 4. Visual Elements Breakdown: What Everything Shows One by One

### Panel 1: SNN Architecture & Synaptic Pulses (Top-Left Canvas)
* **Neuron Nodes (Circles)**:
  * Arranged in 3 physical layers: **Input Layer** (2 neurons for coordinates $x, y$), **Hidden LIF Layer** (8–24 neurons), and **Output LIF Layer** (2 neurons for Class 0 and Class 1).
  * **Soma Color & Glow**: Brightness represents the current instantaneous membrane potential $V(t)$.
  * **Spike Bloom**: When a neuron fires ($S=1$), it flashes pure white-cyan with a radiant halo.
  * **Interactivity**: You can **click on any neuron** in this canvas to instantly switch the Oscilloscope to monitor that neuron!
* **Synaptic Connections (Lines)**:
  * **Cyan Lines**: Positive excitatory weights ($W > 0$).
  * **Pink/Rose Lines**: Negative inhibitory weights ($W < 0$).
  * **Line Thickness**: Proportional to weight magnitude $|W|$.
* **Animated Traveling Pulses (Dots)**:
  * When a pre-synaptic neuron spikes, small glowing energy packets visibly travel along synapses toward post-synaptic neurons, simulating axonal action potential propagation.

---

### Panel 2: 2D Decision Boundary Heatmap (Top-Right Canvas)
* **Background Grid**: Evaluates the SNN over a grid across the 2D coordinate space $[0, 1] \times [0, 1]$.
* **Color Coding**:
  * **Electric Cyan Region**: Network predicts **Class 0** (Class 0 neuron fired more spikes).
  * **Amber/Orange Region**: Network predicts **Class 1** (Class 1 neuron fired more spikes).
  * **Color Saturation / Intensity**: Represents network confidence (the difference in spike frequency between the two output neurons).
* **Scatter Dots**: The dataset training samples.
  * Cyan dots = Class 0 ground truth.
  * Orange dots = Class 1 ground truth.
* **Live Evolution**: As you click **"Train Network"**, you can watch the decision boundary dynamically warp from a random split into non-linear circular or XOR boundaries!

---

### Panel 3: Spike Raster Plot (Bottom-Left Canvas)
The spike raster is the **gold standard visualization in computational neuroscience**:
* **Horizontal Axis (X-axis)**: Time steps $t = 0, 1, 2, \dots, T-1$.
* **Vertical Axis (Y-axis)**: Individual neurons, separated by layer:
  * **Input Channels (`In0`, `In1`)**: Blue labels.
  * **Hidden Channels (`H0` – `H7`)**: Purple labels.
  * **Output Channels (`Out0`, `Out1`)**: Amber labels.
* **Vertical Ticks**: Each vertical bar marks the exact discrete time step where that neuron emitted an action potential.
* **Red Dashed Vertical Cursor**: Shows the simulation playhead scrubbing through time.

---

### Panel 4: Membrane Potential Oscilloscope (Bottom-Right Canvas)
Visualizes the voltage dynamics of the **currently selected neuron**:
* **Horizontal Axis**: Time $t = 0 \dots T-1$.
* **Vertical Axis**: Membrane potential in Volts ($V$).
* **Solid Cyan Curve**: The continuous trace $V(t)$ showing sub-threshold integration and exponential leakage.
* **Red Dashed Line ($V_{th}$)**: Spiking threshold (default $1.0\text{V}$). When the cyan curve touches this line, a spike occurs.
* **Amber Dots & Vertical Rays**: Mark the exact instant an action potential discharged and triggered a membrane reset back to $0\text{V}$.

---

### Panel 5: Control Sidebar & Real-time Telemetry
* **Task Dataset**:
  * `Non-Linear XOR`: 4 clusters requiring hidden non-linear combinations.
  * `Concentric Circles`: Radial inner ring vs outer ring.
  * `Two Moons`: Interleaving crescent manifolds.
  * `Two Spirals`: Complex temporal winding patterns.
* **Hyperparameter Sliders**:
  * `Time Steps (T)`: Temporal resolution of each inference pass (8 to 32 steps).
  * `Membrane Decay (β)`: How much charge is retained between steps ($0.50$ to $0.98$).
  * `Threshold (Vth)`: Voltage required to trigger an action potential.
  * `Learning Rate`: Adam optimizer step size.
  * `Hidden Neurons`: Capacity of the intermediate spiking layer.
* **Telemetry Readouts**:
  * **Spike Rate (Hz)**: Global firing frequency across all active neurons.
  * **Synaptic Ops (SOPs)**: Count of sparse synaptic accumulations. Unlike ANNs which perform dense multiply-accumulate (MAC) operations for every connection, SNNs only consume energy when a spike occurs!
  * **Loss (CE)**: Categorical Cross-Entropy on output rate-coded softmax.
  * **Accuracy (%)**: Percentage of samples correctly classified.
