# 🧠 The Izhikevich Biological Neuron Model — Complete Guide

The **Izhikevich Model** (published in 2003 by computational neuroscientist Eugene M. Izhikevich) is one of the most celebrated neuron models in neuroscience because it solved a classic dilemma:

> **The Dilemma:**
> * **Hodgkin-Huxley (1952):** 4 non-linear differential equations with ion conductances ($Na^+, K^+, Cl^-$). Biologically realistic, but needs massive supercomputing power.
> * **Leaky Integrate-and-Fire (LIF):** 1 simple linear equation. Very fast, but biologically crude (cannot simulate bursting, chattering, or adaptation).
>
> **The Izhikevich Solution:**
> Reduced the 4D Hodgkin-Huxley system to a **2D system of ordinary differential equations (ODEs)** using bifurcation theory. It is almost as fast as LIF, yet reproduces **all 20+ known cortical firing dynamics of the mammalian brain**!

---

## 1. The Two Core Equations

The model tracks two coupled dynamical variables over time:
* **$v$ (Membrane Potential in mV):** The electrical charge across the neuron cell wall.
* **$u$ (Membrane Recovery Variable):** Represents the slower activation of outward $K^+$ currents and inactivation of $Na^+$ ion channels.

$$\frac{dv}{dt} = 0.04 v^2 + 5v + 140 - u + I(t)$$

$$\frac{du}{dt} = a(bv - u)$$

### The Auxiliary Spike Reset:
When $v$ reaches the peak of an action potential ($+30\text{ mV}$):
$$\text{if } v \ge +30\text{ mV} \implies \begin{cases} v \leftarrow c \\ u \leftarrow u + d \end{cases}$$

---

## 2. The 4 Parameters ($a, b, c, d$) in Plain English

| Parameter | Meaning | Typical Values | What changing it does |
| :---: | :--- | :---: | :--- |
| **$a$** | **Recovery timescale** | $0.02 - 0.10$ | Smaller values mean recovery variable $u$ responds slowly, producing bursts of spikes. |
| **$b$** | **Subthreshold sensitivity** | $0.20 - 0.25$ | Controls how strongly fluctuations in voltage $v$ couple to recovery variable $u$. |
| **$c$** | **After-spike voltage reset** | $-65\text{ mV}$ to $-50\text{ mV}$ | The voltage $v$ to which the neuron drops immediately after firing an action potential. |
| **$d$** | **After-spike recovery jump** | $2.0 - 8.0$ | How much extra recovery $u$ is added after firing. High $d$ slows down subsequent firing (**spike frequency adaptation**). |

---

## 3. The 7 Biological Cortical Presets in NeuroSpark

In Tab 3 (**Izhikevich Biological Dynamics**), you can click any of these presets:

### 1. Regular Spiking (RS) — Pyramidal Neurons
* **Parameters:** $a=0.02, b=0.2, c=-65, d=8$
* **Biology:** Found throughout the mammalian neocortex (the thinking layers).
* **Behavior:** When given a constant current $I$, it fires a rapid burst initially, then slows down to a steady, rhythmic pacing (**spike frequency adaptation**).

### 2. Intrinsically Bursting (IB) — Cortical Layer 5 Pyramidal Neurons
* **Parameters:** $a=0.02, b=0.2, c=-55, d=4$
* **Biology:** Deep output neurons in layer 5 of the cerebral cortex.
* **Behavior:** Discharges a burst of 3 to 5 rapid action potentials, followed by regular single spikes.

### 3. Chattering (CH) — Fast Bursting Interneurons
* **Parameters:** $a=0.02, b=0.2, c=-50, d=2$
* **Biology:** Cortical inhibitory neurons involved in 40 Hz gamma oscillations (linked to conscious attention).
* **Behavior:** Fires repetitive clusters of very high-frequency spikes with quiet gaps between bursts.

### 4. Fast Spiking (FS) — Chandelier & Basket Cells
* **Parameters:** $a=0.1, b=0.2, c=-65, d=2$
* **Biology:** Inhibitory interneurons that regulate cortical circuits.
* **Behavior:** Fires a relentless, high-frequency stream of spikes (up to 300+ Hz) with **zero adaptation**.

### 5. Low-Threshold Spiking (LTS)
* **Parameters:** $a=0.02, b=0.25, c=-65, d=2$
* **Behavior:** Interneurons capable of firing high-frequency bursts when recovering from an inhibitory/hyperpolarized state.

### 6. Thalamo-Cortical (TC) — Relay Neurons
* **Parameters:** $a=0.02, b=0.25, c=-65, d=0.05$
* **Biology:** Gateway neurons in the thalamus routing sensory data to the cortex.
* **Behavior:** Displays bistable tonic firing or post-inhibitory rebound bursts (essential for sleep rhythms).

### 7. Resonator (RZ) — Subthreshold Oscillations
* **Parameters:** $a=0.1, b=0.26, c=-65, d=2$
* **Behavior:** Exhibits damped subthreshold oscillations and selective resonance to specific frequency stimuli.

---

## 4. What the Tab 3 Canvases Show

### Left Canvas: Oscilloscope $v(t)$
* Displays the membrane potential waveform in millivolts (mV).
* Spikes cleanly peak at $+30\text{ mV}$ (dashed red line) before dropping to reset value $c$.

### Right Canvas: Phase Plane Portrait ($v$ vs $u$)
In dynamical systems theory, plotting $v$ against $u$ visualizes the system's attractor landscape:
* **Yellow Parabola ($v$-nullcline):** The curve where $\frac{dv}{dt} = 0$, defined by $u = 0.04 v^2 + 5v + 140 + I$.
* **Pink Line ($u$-nullcline):** The line where $\frac{du}{dt} = 0$, defined by $u = bv$.
* **Intersection Point:** The resting fixed point of the neuron.
* **Moving Cyan Loop:** The **Limit Cycle Trajectory**. As the neuron fires, the point orbits along the parabola and resets across the phase space.
