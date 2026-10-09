/**
 * Spiking Neural Network (SNN) Core Engine
 * Implements Leaky Integrate-and-Fire (LIF) neurons with Surrogate Gradient BPTT
 */

export class LIFLayer {
  constructor(nIn, nOut, options = {}) {
    this.nIn = nIn;
    this.nOut = nOut;
    this.beta = options.beta ?? 0.85; // Membrane decay factor (tau)
    this.vThresh = options.vThresh ?? 1.0; // Spiking threshold
    this.vReset = options.vReset ?? 0.0; // Reset voltage
    this.surrogateScale = options.surrogateScale ?? 2.0; // Slope parameter for surrogate derivative
    this.resetMode = options.resetMode ?? 'zero'; // 'zero' or 'subtract'

    // Xavier/Kaiming initialization
    const scale = Math.sqrt(2.0 / (nIn + nOut));
    this.weights = new Float32Array(nOut * nIn);
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = (Math.random() * 2 - 1) * scale * 1.5;
    }
    this.biases = new Float32Array(nOut);

    // Adam optimizer buffers
    this.mW = new Float32Array(nOut * nIn);
    this.vW = new Float32Array(nOut * nIn);
    this.mB = new Float32Array(nOut);
    this.vB = new Float32Array(nOut);

    // Gradients
    this.gradW = new Float32Array(nOut * nIn);
    this.gradB = new Float32Array(nOut);

    // Forward cache for BPTT
    this.cache = null;
  }

  // Fast Sigmoid surrogate derivative: dS/dV ≈ 1 / (1 + γ * |V - Vth|)^2
  surrogateGrad(v) {
    const diff = Math.abs(v - this.vThresh);
    const denom = 1.0 + this.surrogateScale * diff;
    return 1.0 / (denom * denom);
  }

  // Forward pass through T time steps
  // xSeq: Float32Array of shape [T, nIn] (or [T, batch, nIn] if batch handled)
  forward(xSeq, T) {
    const nIn = this.nIn;
    const nOut = this.nOut;
    const beta = this.beta;
    const vThresh = this.vThresh;
    const vReset = this.vReset;

    const sHist = new Float32Array(T * nOut); // Spikes
    const vHist = new Float32Array(T * nOut); // Membrane voltages
    const iHist = new Float32Array(T * nOut); // Synaptic currents

    let vPrev = new Float32Array(nOut);
    let sPrev = new Float32Array(nOut);

    for (let t = 0; t < T; t++) {
      const tInOffset = t * nIn;
      const tOutOffset = t * nOut;

      for (let j = 0; j < nOut; j++) {
        // Compute synaptic input I[t] = W * x[t] + b
        let sum = this.biases[j];
        const wOffset = j * nIn;
        for (let i = 0; i < nIn; i++) {
          sum += this.weights[wOffset + i] * xSeq[tInOffset + i];
        }
        iHist[tOutOffset + j] = sum;

        // Leaky integrate: decay membrane potential if not spiked
        let v;
        if (this.resetMode === 'zero') {
          v = beta * vPrev[j] * (1.0 - sPrev[j]) + sum;
        } else {
          // Reset by subtraction
          v = beta * (vPrev[j] - sPrev[j] * vThresh) + sum;
        }

        // Fire spike
        const spike = v >= vThresh ? 1.0 : 0.0;

        vHist[tOutOffset + j] = v;
        sHist[tOutOffset + j] = spike;

        vPrev[j] = v;
        sPrev[j] = spike;
      }
    }

    this.cache = { xSeq, sHist, vHist, iHist, T };
    return { sHist, vHist, iHist };
  }

  // Zero out gradients before backward pass
  zeroGrad() {
    this.gradW.fill(0);
    this.gradB.fill(0);
  }

  // Backward pass through time (BPTT) with surrogate gradient
  // gradS_Seq: gradient wrt output spikes [T, nOut]
  // Returns gradient wrt input sequence [T, nIn]
  backward(gradS_Seq) {
    const { xSeq, sHist, vHist, T } = this.cache;
    const nIn = this.nIn;
    const nOut = this.nOut;
    const beta = this.beta;

    const gradX_Seq = new Float32Array(T * nIn);
    const deltaV = new Float32Array(nOut); // Backpropagated voltage delta across time

    for (let t = T - 1; t >= 0; t--) {
      const tOutOffset = t * nOut;
      const tInOffset = t * nIn;

      for (let j = 0; j < nOut; j++) {
        const v = vHist[tOutOffset + j];
        const s = sHist[tOutOffset + j];
        const dSurr = this.surrogateGrad(v);

        // dL/dV[t] = dL/dS[t] * surrogateGrad(V[t]) + deltaV[t+1] * beta * (1 - S[t])
        const dL_dV = gradS_Seq[tOutOffset + j] * dSurr + deltaV[j] * beta * (1.0 - s);

        // Update deltaV for t-1
        deltaV[j] = dL_dV;

        // Accumulate weight & bias gradients
        const wOffset = j * nIn;
        this.gradB[j] += dL_dV;
        for (let i = 0; i < nIn; i++) {
          this.gradW[wOffset + i] += dL_dV * xSeq[tInOffset + i];
          // Gradient wrt input
          gradX_Seq[tInOffset + i] += dL_dV * this.weights[wOffset + i];
        }
      }
    }

    return gradX_Seq;
  }

  // Update weights using Adam optimizer
  step(lr = 0.005, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, tStep = 1) {
    const b1Corr = 1.0 - Math.pow(beta1, tStep);
    const b2Corr = 1.0 - Math.pow(beta2, tStep);

    for (let i = 0; i < this.weights.length; i++) {
      let g = this.gradW[i];
      // Gradient clipping
      if (g > 5.0) g = 5.0;
      if (g < -5.0) g = -5.0;

      this.mW[i] = beta1 * this.mW[i] + (1.0 - beta1) * g;
      this.vW[i] = beta2 * this.vW[i] + (1.0 - beta2) * g * g;

      const mHat = this.mW[i] / b1Corr;
      const vHat = this.vW[i] / b2Corr;

      this.weights[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
    }

    for (let j = 0; j < this.biases.length; j++) {
      let g = this.gradB[j];
      if (g > 5.0) g = 5.0;
      if (g < -5.0) g = -5.0;

      this.mB[j] = beta1 * this.mB[j] + (1.0 - beta1) * g;
      this.vB[j] = beta2 * this.vB[j] + (1.0 - beta2) * g * g;

      const mHat = this.mB[j] / b1Corr;
      const vHat = this.vB[j] / b2Corr;

      this.biases[j] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
    }
  }
}

/**
 * Multi-layer Spiking Neural Network
 */
export class SpikingNeuralNetwork {
  constructor(layerSizes, options = {}) {
    this.layerSizes = layerSizes; // e.g., [2, 16, 2]
    this.timeSteps = options.timeSteps ?? 16;
    this.lr = options.lr ?? 0.01;
    this.stepCount = 0;
    this.encodingMode = options.encodingMode ?? 'rate'; // 'rate', 'constant', 'poisson'

    this.layers = [];
    for (let i = 0; i < layerSizes.length - 1; i++) {
      this.layers.push(
        new LIFLayer(layerSizes[i], layerSizes[i + 1], {
          beta: options.beta ?? 0.85,
          vThresh: options.vThresh ?? 1.0,
          surrogateScale: options.surrogateScale ?? 2.0
        })
      );
    }
  }

  // Encode input vector into a spike sequence or continuous current over T steps
  encodeInput(inputVec, T = this.timeSteps) {
    const nIn = inputVec.length;
    const xSeq = new Float32Array(T * nIn);

    if (this.encodingMode === 'poisson') {
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < nIn; i++) {
          // Probability of spike proportional to input (normalized 0..1)
          const prob = Math.max(0, Math.min(1, inputVec[i]));
          xSeq[t * nIn + i] = Math.random() < prob ? 1.0 : 0.0;
        }
      }
    } else if (this.encodingMode === 'constant') {
      // Direct constant input current injected continuously
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < nIn; i++) {
          xSeq[t * nIn + i] = inputVec[i];
        }
      }
    } else {
      // Default: Rate coding with smooth pulse train
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < nIn; i++) {
          const val = inputVec[i];
          // Deterministic spike train based on rate frequency
          if (val <= 0) {
            xSeq[t * nIn + i] = 0;
          } else {
            const period = Math.max(1, Math.round(1.0 / val));
            xSeq[t * nIn + i] = t % period === 0 ? 1.0 : 0.0;
          }
        }
      }
    }

    return xSeq;
  }

  // Forward pass across all layers
  forward(xSeq, T = this.timeSteps) {
    let currentIn = xSeq;
    const layerOutputs = [];

    for (const layer of this.layers) {
      const out = layer.forward(currentIn, T);
      layerOutputs.push(out);
      currentIn = out.sHist; // Spikes propagate to next layer
    }

    // Output layer spike counts
    const outLayer = this.layers[this.layers.length - 1];
    const nOut = outLayer.nOut;
    const lastOutput = layerOutputs[layerOutputs.length - 1];
    const spikeCounts = new Float32Array(nOut);

    for (let t = 0; t < T; t++) {
      for (let j = 0; j < nOut; j++) {
        spikeCounts[j] += lastOutput.sHist[t * nOut + j];
      }
    }

    return {
      spikeCounts,
      layerOutputs,
      inputSeq: xSeq
    };
  }

  // Softmax helper
  softmax(logits) {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > max) max = logits[i];
    }
    let sum = 0;
    const probs = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      probs[i] = Math.exp(logits[i] - max);
      sum += probs[i];
    }
    for (let i = 0; i < logits.length; i++) {
      probs[i] /= (sum || 1.0);
    }
    return probs;
  }

  // Train on a single sample with surrogate gradient BPTT
  trainSample(inputVec, targetClass, lr = this.lr) {
    this.stepCount++;
    const T = this.timeSteps;
    const xSeq = this.encodeInput(inputVec, T);

    // Forward
    const { spikeCounts, layerOutputs } = this.forward(xSeq, T);
    const outLayer = this.layers[this.layers.length - 1];
    const nOut = outLayer.nOut;

    // Rate-based softmax probabilities
    // Scale spike counts to reasonable logit range
    const logits = new Float32Array(nOut);
    for (let j = 0; j < nOut; j++) {
      logits[j] = (spikeCounts[j] / T) * 4.0;
    }
    const probs = this.softmax(logits);

    // Cross-Entropy Loss
    const targetProb = Math.max(1e-7, probs[targetClass]);
    const loss = -Math.log(targetProb);

    // Prediction
    let predictedClass = 0;
    let maxCount = -1;
    for (let j = 0; j < nOut; j++) {
      if (spikeCounts[j] > maxCount) {
        maxCount = spikeCounts[j];
        predictedClass = j;
      }
    }
    const isCorrect = predictedClass === targetClass;

    // Output error gradient wrt total spike rate
    // dL/d(SpikeCount[j]) = (probs[j] - target_j) * (4.0 / T)
    const dL_dSpikeCount = new Float32Array(nOut);
    for (let j = 0; j < nOut; j++) {
      const targetVal = j === targetClass ? 1.0 : 0.0;
      dL_dSpikeCount[j] = (probs[j] - targetVal) * (4.0 / T);
    }

    // Expand to each time step [T, nOut]
    const gradS_Seq = new Float32Array(T * nOut);
    for (let t = 0; t < T; t++) {
      for (let j = 0; j < nOut; j++) {
        gradS_Seq[t * nOut + j] = dL_dSpikeCount[j];
      }
    }

    // Zero grads
    for (const layer of this.layers) {
      layer.zeroGrad();
    }

    // Backward pass from last layer to first
    let currentGradS = gradS_Seq;
    for (let l = this.layers.length - 1; l >= 0; l--) {
      currentGradS = this.layers[l].backward(currentGradS);
    }

    // Optimization step
    for (const layer of this.layers) {
      layer.step(lr, 0.9, 0.999, 1e-8, this.stepCount);
    }

    return {
      loss,
      isCorrect,
      predictedClass,
      probs,
      spikeCounts,
      layerOutputs
    };
  }

  // Predict without backprop
  predict(inputVec, T = this.timeSteps) {
    const xSeq = this.encodeInput(inputVec, T);
    const { spikeCounts, layerOutputs } = this.forward(xSeq, T);
    const nOut = this.layers[this.layers.length - 1].nOut;

    const logits = new Float32Array(nOut);
    for (let j = 0; j < nOut; j++) {
      logits[j] = (spikeCounts[j] / T) * 4.0;
    }
    const probs = this.softmax(logits);

    let predictedClass = 0;
    let maxCount = -1;
    for (let j = 0; j < nOut; j++) {
      if (spikeCounts[j] > maxCount) {
        maxCount = spikeCounts[j];
        predictedClass = j;
      }
    }

    return {
      predictedClass,
      probs,
      spikeCounts,
      layerOutputs,
      inputSeq: xSeq
    };
  }
}
