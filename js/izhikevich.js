/**
 * Izhikevich Simple Model of Spiking Neurons
 * References: Eugene M. Izhikevich (2003) "Simple Model of Spiking Neurons"
 * IEEE Transactions on Neural Networks, 14(6), 1569-1572.
 */

export const IZHIKEVICH_PRESETS = {
  regularSpiking: {
    name: "Regular Spiking (RS)",
    desc: "Neocortical pyramidal neurons. Exhibits spike frequency adaptation: high initial firing rate that slows down to steady state.",
    a: 0.02,
    b: 0.2,
    c: -65.0,
    d: 8.0,
    defaultCurrent: 10.0
  },
  intrinsicallyBursting: {
    name: "Intrinsically Bursting (IB)",
    desc: "Cortical layer 5 pyramidal neurons. Fires an initial burst of action potentials followed by repetitive single spikes.",
    a: 0.02,
    b: 0.2,
    c: -55.0,
    d: 4.0,
    defaultCurrent: 10.0
  },
  chattering: {
    name: "Chattering (CH)",
    desc: "Cortical inhibitory interneurons. Discharges stereotypical high-frequency bursts of spikes separated by brief quiescent periods.",
    a: 0.02,
    b: 0.2,
    c: -50.0,
    d: 2.0,
    defaultCurrent: 10.0
  },
  fastSpiking: {
    name: "Fast Spiking (FS)",
    desc: "Inhibitory chandelier and basket cells. Fires periodic trains of high-frequency action potentials with zero spike adaptation.",
    a: 0.1,
    b: 0.2,
    c: -65.0,
    d: 2.0,
    defaultCurrent: 10.0
  },
  lowThresholdSpiking: {
    name: "Low-Threshold Spiking (LTS)",
    desc: "Cortical interneurons firing high-frequency bursts when stimulated from hyperpolarized states, with frequency adaptation.",
    a: 0.02,
    b: 0.25,
    c: -65.0,
    d: 2.0,
    defaultCurrent: 10.0
  },
  thalamoCortical: {
    name: "Thalamo-Cortical (TC)",
    desc: "Thalamocortical relay cells showing bistable tonic firing or post-inhibitory rebound burst spiking.",
    a: 0.02,
    b: 0.25,
    c: -65.0,
    d: 0.05,
    defaultCurrent: 8.0
  },
  resonator: {
    name: "Resonator (RZ)",
    desc: "Subthreshold oscillatory dynamics. Selectively fires action potentials in response to specific resonant frequency stimuli.",
    a: 0.1,
    b: 0.26,
    c: -65.0,
    d: 2.0,
    defaultCurrent: 3.5
  }
};

export class IzhikevichNeuron {
  constructor(presetKey = "regularSpiking") {
    this.setPreset(presetKey);
    this.reset();
  }

  setPreset(key) {
    const p = IZHIKEVICH_PRESETS[key] || IZHIKEVICH_PRESETS.regularSpiking;
    this.presetKey = key;
    this.name = p.name;
    this.desc = p.desc;
    this.a = p.a;
    this.b = p.b;
    this.c = p.c;
    this.d = p.d;
    this.current = p.defaultCurrent;
  }

  reset() {
    this.v = -65.0; // Resting membrane potential (mV)
    this.u = this.b * this.v; // Recovery variable
    this.history = [];
    this.maxHistory = 1000;
    this.time = 0;
    this.spikeCount = 0;
  }

  // Numerical integration step with two 0.5 ms sub-steps for numerical stability
  step(injectedCurrent = this.current, dt = 0.5) {
    let spiked = false;
    const subSteps = 2;
    const subDt = dt / subSteps;

    for (let s = 0; s < subSteps; s++) {
      // dv/dt = 0.04*v^2 + 5*v + 140 - u + I
      const dv = 0.04 * this.v * this.v + 5.0 * this.v + 140.0 - this.u + injectedCurrent;
      // du/dt = a*(b*v - u)
      const du = this.a * (this.b * this.v - this.u);

      this.v += dv * subDt;
      this.u += du * subDt;

      // Spike detection threshold is +30 mV
      if (this.v >= 30.0) {
        this.v = this.c;
        this.u += this.d;
        spiked = true;
      }
    }

    this.time += dt;
    if (spiked) this.spikeCount++;

    const record = {
      t: this.time,
      v: spiked ? 30.0 : this.v, // Record peak for visual fidelity
      u: this.u,
      i: injectedCurrent,
      spiked
    };

    this.history.push(record);
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    }

    return record;
  }

  // Returns nullclines for Phase Plane analysis
  // v-nullcline: u = 0.04*v^2 + 5*v + 140 + I
  // u-nullcline: u = b*v
  getNullclines(current = this.current, vMin = -80, vMax = 40, steps = 100) {
    const vNull = [];
    const uNull = [];
    const dv = (vMax - vMin) / steps;

    for (let i = 0; i <= steps; i++) {
      const v = vMin + i * dv;
      const uV = 0.04 * v * v + 5.0 * v + 140.0 + current;
      const uU = this.b * v;
      vNull.push({ v, u: uV });
      uNull.push({ v, u: uU });
    }

    return { vNull, uNull };
  }
}
