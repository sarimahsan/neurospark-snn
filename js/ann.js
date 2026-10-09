/**
 * Standard Artificial Neural Network (ANN / MLP) with ReLU Activations
 * Used for direct side-by-side benchmarking and comparison against the SNN
 */

export class StandardANN {
  constructor(layerSizes, lr = 0.015) {
    this.layerSizes = layerSizes; // e.g., [2, 16, 2]
    this.lr = lr;
    this.stepCount = 0;

    this.weights = [];
    this.biases = [];
    this.mW = [];
    this.vW = [];
    this.mB = [];
    this.vB = [];

    // Initialize layers with Xavier/Kaiming normal
    for (let l = 0; l < layerSizes.length - 1; l++) {
      const nIn = layerSizes[l];
      const nOut = layerSizes[l + 1];
      const scale = Math.sqrt(2.0 / (nIn + nOut));

      const w = new Float32Array(nOut * nIn);
      for (let i = 0; i < w.length; i++) {
        w[i] = (Math.random() * 2 - 1) * scale * 1.5;
      }
      this.weights.push(w);
      this.biases.push(new Float32Array(nOut));

      // Adam buffers
      this.mW.push(new Float32Array(nOut * nIn));
      this.vW.push(new Float32Array(nOut * nIn));
      this.mB.push(new Float32Array(nOut));
      this.vB.push(new Float32Array(nOut));
    }
  }

  // Calculate dense FLOPs (Multiply-Accumulate operations) for 1 inference pass
  getFlopsPerInference() {
    let flops = 0;
    for (let l = 0; l < this.layerSizes.length - 1; l++) {
      const nIn = this.layerSizes[l];
      const nOut = this.layerSizes[l + 1];
      // Multiply + Add per weight = 2 operations
      flops += 2 * nIn * nOut + nOut;
    }
    return flops;
  }

  forward(inputVec) {
    let current = Float32Array.from(inputVec);
    const activations = [current];
    const preActivations = [];

    for (let l = 0; l < this.weights.length; l++) {
      const nIn = this.layerSizes[l];
      const nOut = this.layerSizes[l + 1];
      const w = this.weights[l];
      const b = this.biases[l];
      const z = new Float32Array(nOut);

      for (let j = 0; j < nOut; j++) {
        let sum = b[j];
        const offset = j * nIn;
        for (let i = 0; i < nIn; i++) {
          sum += w[offset + i] * current[i];
        }
        z[j] = sum;
      }
      preActivations.push(z);

      // Activation: ReLU for hidden layers, linear for output layer
      if (l < this.weights.length - 1) {
        current = new Float32Array(nOut);
        for (let j = 0; j < nOut; j++) {
          current[j] = z[j] > 0 ? z[j] : 0; // Standard ReLU
        }
      } else {
        current = z; // Output logits
      }
      activations.push(current);
    }

    // Softmax on output logits
    const logits = activations[activations.length - 1];
    let max = -Infinity;
    for (let j = 0; j < logits.length; j++) if (logits[j] > max) max = logits[j];

    let sumExp = 0;
    const probs = new Float32Array(logits.length);
    for (let j = 0; j < logits.length; j++) {
      probs[j] = Math.exp(logits[j] - max);
      sumExp += probs[j];
    }
    for (let j = 0; j < logits.length; j++) {
      probs[j] /= (sumExp || 1.0);
    }

    return { activations, preActivations, probs };
  }

  predict(inputVec) {
    const { probs, activations } = this.forward(inputVec);
    let predClass = 0;
    let maxP = -1;
    for (let j = 0; j < probs.length; j++) {
      if (probs[j] > maxP) {
        maxP = probs[j];
        predClass = j;
      }
    }
    return { predictedClass: predClass, probs, activations };
  }

  trainSample(inputVec, targetClass, lr = this.lr) {
    this.stepCount++;
    const { activations, preActivations, probs } = this.forward(inputVec);

    // Cross-Entropy Loss
    const targetProb = Math.max(1e-7, probs[targetClass]);
    const loss = -Math.log(targetProb);

    let predClass = 0;
    let maxP = -1;
    for (let j = 0; j < probs.length; j++) {
      if (probs[j] > maxP) {
        maxP = probs[j];
        predClass = j;
      }
    }
    const isCorrect = predClass === targetClass;

    // Gradient of Cross-Entropy wrt output logits: (p_j - y_j)
    let delta = new Float32Array(probs.length);
    for (let j = 0; j < probs.length; j++) {
      delta[j] = probs[j] - (j === targetClass ? 1.0 : 0.0);
    }

    // Backprop through layers
    for (let l = this.weights.length - 1; l >= 0; l--) {
      const nIn = this.layerSizes[l];
      const nOut = this.layerSizes[l + 1];
      const actIn = activations[l];
      const w = this.weights[l];
      const b = this.biases[l];

      const gradW = new Float32Array(nOut * nIn);
      const gradB = new Float32Array(nOut);
      const nextDelta = new Float32Array(nIn);

      for (let j = 0; j < nOut; j++) {
        const d = delta[j];
        gradB[j] = d;
        const offset = j * nIn;
        for (let i = 0; i < nIn; i++) {
          gradW[offset + i] = d * actIn[i];
          nextDelta[i] += d * w[offset + i];
        }
      }

      // Adam update for layer l
      const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
      const b1Corr = 1.0 - Math.pow(beta1, this.stepCount);
      const b2Corr = 1.0 - Math.pow(beta2, this.stepCount);

      const mW = this.mW[l], vW = this.vW[l];
      for (let i = 0; i < w.length; i++) {
        let g = gradW[i];
        if (g > 5.0) g = 5.0;
        if (g < -5.0) g = -5.0;
        mW[i] = beta1 * mW[i] + (1 - beta1) * g;
        vW[i] = beta2 * vW[i] + (1 - beta2) * g * g;
        const mHat = mW[i] / b1Corr;
        const vHat = vW[i] / b2Corr;
        w[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      }

      const mB = this.mB[l], vB = this.vB[l];
      for (let j = 0; j < b.length; j++) {
        let g = gradB[j];
        if (g > 5.0) g = 5.0;
        if (g < -5.0) g = -5.0;
        mB[j] = beta1 * mB[j] + (1 - beta1) * g;
        vB[j] = beta2 * vB[j] + (1 - beta2) * g * g;
        const mHat = mB[j] / b1Corr;
        const vHat = vB[j] / b2Corr;
        b[j] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      }

      // Propagate delta through ReLU for next layer
      if (l > 0) {
        const zPrev = preActivations[l - 1];
        delta = new Float32Array(nIn);
        for (let i = 0; i < nIn; i++) {
          delta[i] = zPrev[i] > 0 ? nextDelta[i] : 0; // dReLU/dz
        }
      }
    }

    return { loss, isCorrect, predictedClass: predClass, probs };
  }
}
