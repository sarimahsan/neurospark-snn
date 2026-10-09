# ✏️ Pattern Drawing Studio — Guide for ML Engineers

This guide explains **how the Pattern Drawing Studio works**, how a static 2D image is converted into temporal spike trains, and **what each panel displays**.

---

## 1. The High-Level Architecture (In ML Terms)

In a conventional Convolutional Neural Network (CNN) or Vision Transformer (ViT), an image is a static tensor of shape `(1, 64)`. The network does a single matrix multiplication pass:
$$\mathbf{y} = \text{Softmax}(W_2 \cdot \text{ReLU}(W_1 \cdot \mathbf{x} + b_1) + b_2)$$

In the **Pattern Drawing Studio**, the network is an SNN with architecture **$64 \to 32 \to 5$**:
* **64 Input Neurons**: One for each pixel of the $8 \times 8$ grid.
* **32 Hidden LIF Neurons**: Recurrent spiking feature detectors.
* **5 Output Class Neurons**: One neuron for each digit ($0, 1, 2, 3, 4$).
* **Time Dimension ($T = 16$)**: The image is streamed over a sequence of 16 time steps.

```
[8x8 Drawing Pad] 
       │ (Pixel intensity = firing rate)
       ▼
 [64 Input Spikes across 16 steps] 
       │ (W1 weights)
       ▼
[32 Hidden LIF Neurons with leaky potential V(t)] 
       │ (W2 weights)
       ▼
 [5 Output LIF Neurons (Digits 0 to 4)]
       │ (Count spikes over 16 steps)
       ▼
[Highest Spikes = Predicted Class]
```

---

## 2. Step-by-Step: From Mouse Stroke to Prediction

### Step 1: Pixel-to-Spike Encoding (Poisson / Rate Coding)
When you sketch a digit with the mouse or touch:
1. The canvas downsamples your drawing into an **$8 \times 8$ array of floats** between `0.0` (black/empty) and `1.0` (white/drawn).
2. For each time step $t \in [0, 15]$:
   * A bright pixel ($1.0$) fires a spike ($1$) almost every step.
   * A medium pixel ($0.5$) fires on roughly half the steps.
   * A dark pixel ($0.0$) stays silent ($0$).
3. The static image is now transformed into a **binary tensor of shape `(16, 64)`**.

### Step 2: Hidden Layer Feature Extraction ($64 \to 32$)
* At each step $t$, pre-synaptic spikes inject current $I = W_1 \cdot S_{in}[t]$.
* The 32 hidden neurons integrate this current into their membrane voltages $V_h[t]$ with decay $\beta = 0.88$.
* Whenever $V_h[t] \ge 1.0\text{V}$, that hidden neuron discharges a spike $S_h[t] = 1$ and resets to $0\text{V}$.

### Step 3: Class Readout ($32 \to 5$)
* Spikes from the hidden layer stimulate the 5 output neurons.
* At the end of $T = 16$ steps, we count how many times each output neuron spiked:
  $$\text{SpikeCount}_c = \sum_{t=0}^{15} S_{out}[t, c] \quad (c \in \{0, 1, 2, 3, 4\})$$
* **The digit whose output neuron spiked the most is the network's prediction!**
* Confidence probabilities are calculated via Softmax on normalized spike rates:
  $$P(\text{class } c) = \frac{\exp\left(4 \cdot \frac{\text{SpikeCount}_c}{16}\right)}{\sum_{k} \exp\left(4 \cdot \frac{\text{SpikeCount}_k}{16}\right)}$$

---

## 3. What Each Panel Shows One by One

### 🖌️ Panel 1: 8×8 Pixel Drawing Pad (Left Side)
* **The Grid**: An interactive $8 \times 8$ canvas.
* **Mouse / Touch Drawing**: Click and drag to draw lines. A soft brush falloff anti-aliases neighbor cells so strokes feel smooth.
* **Clear Pad Button**: Wipes the grid back to all zeros.
* **Preset Archetypes (Pills `Digit 0` to `Digit 4`)**: Instant pre-defined bitmap templates. Clicking one injects the archetype into the pad and immediately runs an inference pass.
* **"Train Digits" Button**: Triggers online surrogate gradient BPTT training on 175 augmented digit variations (shifted by $\pm 1$ pixel with random noise) right in your browser.

---

### 📊 Panel 2: Spiking Predictions & Firing Rates (Top-Right)
* **Prediction Badge**: Displays the winning class (e.g., `Predicted: Digit 3`) or `No Input` if the pad is blank.
* **5 Horizontal Progress Bars**:
  * Each bar corresponds to one digit class ($0, 1, 2, 3, 4$).
  * **Bar Width**: Shows the Softmax probability percentage (e.g., `87%`).
  * **Numeric Readout**: Displays both percentage and raw spike count, e.g., `87% (14)`. This means Output Neuron #3 fired 14 spikes out of 16 time steps.
  * **Glowing Amber/Rose Highlight**: The top predicted digit lights up with an energetic glow.

---

### 📈 Panel 3: Digit Processing Spike Raster (Bottom-Right)
This plot reveals the internal temporal computation happening across the network for your drawn digit:
* **Horizontal Axis (X-axis)**: The 16 time steps ($t = 0 \dots 15$).
* **Vertical Axis (Y-axis)**: Shows stacked neuron channels:
  * **Input Channels (`In18`, `In26`...)**: The pixels you drew on. You can see their rapid firing ticks in **Cyan**.
  * **Hidden Channels (`H0` – `H5`)**: Sampled hidden LIF neurons firing in **Purple**.
  * **Output Channels (`C0` – `C4`)**: The 5 output classification neurons in **Amber**. Notice how the winning digit channel has a dense train of spikes, while competing digit channels stay mostly quiet!
