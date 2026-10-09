/**
 * Datasets for SNN Classification and Interactive Pattern Testing
 */

export const DATASETS_2D = {
  xor: {
    name: "Non-Linear XOR",
    description: "Classic non-linear benchmark. Cannot be solved by a linear classifier without hidden spiking representations.",
    generate: (numSamples = 160) => {
      const data = [];
      for (let i = 0; i < numSamples; i++) {
        const quadrant = Math.floor(Math.random() * 4);
        let x, y, label;
        const noiseX = (Math.random() - 0.5) * 0.35;
        const noiseY = (Math.random() - 0.5) * 0.35;

        switch (quadrant) {
          case 0: // (+, +) -> class 0
            x = 0.65 + noiseX; y = 0.65 + noiseY; label = 0; break;
          case 1: // (-, +) -> class 1
            x = 0.25 + noiseX; y = 0.65 + noiseY; label = 1; break;
          case 2: // (-, -) -> class 0
            x = 0.25 + noiseX; y = 0.25 + noiseY; label = 0; break;
          case 3: // (+, -) -> class 1
            x = 0.65 + noiseX; y = 0.25 + noiseY; label = 1; break;
        }
        data.push({
          input: [Math.max(0.05, Math.min(0.95, x)), Math.max(0.05, Math.min(0.95, y))],
          label
        });
      }
      return data;
    }
  },

  circles: {
    name: "Concentric Circles",
    description: "Inner cluster vs outer circular ring. Tests radial boundary formation in spiking frequency space.",
    generate: (numSamples = 180) => {
      const data = [];
      const half = Math.floor(numSamples / 2);
      // Inner circle (class 0)
      for (let i = 0; i < half; i++) {
        const r = Math.random() * 0.22;
        const theta = Math.random() * 2 * Math.PI;
        data.push({
          input: [0.5 + r * Math.cos(theta), 0.5 + r * Math.sin(theta)],
          label: 0
        });
      }
      // Outer ring (class 1)
      for (let i = 0; i < half; i++) {
        const r = 0.32 + Math.random() * 0.14;
        const theta = Math.random() * 2 * Math.PI;
        data.push({
          input: [0.5 + r * Math.cos(theta), 0.5 + r * Math.sin(theta)],
          label: 1
        });
      }
      return data;
    }
  },

  moons: {
    name: "Two Interleaving Moons",
    description: "Two interlocking crescent arcs. A standard machine learning benchmark for non-linear manifold separation.",
    generate: (numSamples = 180) => {
      const data = [];
      const half = Math.floor(numSamples / 2);
      // Moon 1 (top arc)
      for (let i = 0; i < half; i++) {
        const theta = Math.PI * Math.random();
        const r = 0.3 + (Math.random() - 0.5) * 0.08;
        const x = 0.4 + r * Math.cos(theta);
        const y = 0.5 + r * Math.sin(theta);
        data.push({ input: [Math.max(0.05, Math.min(0.95, x)), Math.max(0.05, Math.min(0.95, y))], label: 0 });
      }
      // Moon 2 (bottom shifted arc)
      for (let i = 0; i < half; i++) {
        const theta = Math.PI * Math.random();
        const r = 0.3 + (Math.random() - 0.5) * 0.08;
        const x = 0.6 - r * Math.cos(theta);
        const y = 0.5 - r * Math.sin(theta) + 0.1;
        data.push({ input: [Math.max(0.05, Math.min(0.95, x)), Math.max(0.05, Math.min(0.95, y))], label: 1 });
      }
      return data;
    }
  },

  spirals: {
    name: "Two Spirals",
    description: "Intertwined Archimedean spirals requiring deep temporal feature resolution across time steps.",
    generate: (numSamples = 200) => {
      const data = [];
      const half = Math.floor(numSamples / 2);
      for (let i = 0; i < half; i++) {
        const t = (i / half) * 3.5;
        const r = (t / 3.5) * 0.42;
        const theta = t * Math.PI;
        const noise = (Math.random() - 0.5) * 0.03;
        // Spiral 0
        data.push({
          input: [0.5 + (r + noise) * Math.cos(theta), 0.5 + (r + noise) * Math.sin(theta)],
          label: 0
        });
        // Spiral 1 (180 deg phase)
        data.push({
          input: [0.5 - (r + noise) * Math.cos(theta), 0.5 - (r + noise) * Math.sin(theta)],
          label: 1
        });
      }
      return data;
    }
  }
};

/**
 * 8x8 Digit and Glyph Templates
 * Clean 8x8 binary bitmap archetypes for interactive drawing & classification
 */
export const DIGIT_TEMPLATES_8X8 = [
  // 0
  [
    0, 0, 1, 1, 1, 1, 0, 0,
    0, 1, 1, 0, 0, 1, 1, 0,
    1, 1, 0, 0, 0, 0, 1, 1,
    1, 1, 0, 0, 0, 0, 1, 1,
    1, 1, 0, 0, 0, 0, 1, 1,
    1, 1, 0, 0, 0, 0, 1, 1,
    0, 1, 1, 0, 0, 1, 1, 0,
    0, 0, 1, 1, 1, 1, 0, 0
  ],
  // 1
  [
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 1, 0, 0, 0,
    0, 1, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 0
  ],
  // 2
  [
    0, 1, 1, 1, 1, 1, 0, 0,
    1, 1, 0, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1
  ],
  // 3
  [
    0, 1, 1, 1, 1, 1, 0, 0,
    1, 1, 0, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 0, 1, 1, 0,
    0, 0, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 0, 1, 1, 0,
    1, 1, 0, 0, 0, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 0, 0
  ],
  // 4
  [
    0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 1, 1, 1, 0, 0,
    0, 0, 1, 0, 1, 1, 0, 0,
    0, 1, 0, 0, 1, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0
  ]
];

/**
 * Generate augmented digit training dataset with random shifts & noise
 */
export function generateDigitDataset(samplesPerDigit = 40) {
  const dataset = [];
  const numClasses = DIGIT_TEMPLATES_8X8.length; // 5 digits (0..4) for fast crisp browser training

  for (let c = 0; c < numClasses; c++) {
    const base = DIGIT_TEMPLATES_8X8[c];
    for (let s = 0; s < samplesPerDigit; s++) {
      const sample = new Float32Array(64);
      // Random shift -1, 0, or 1
      const dx = Math.floor(Math.random() * 3) - 1;
      const dy = Math.floor(Math.random() * 3) - 1;

      for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 8; x++) {
          const srcX = x - dx;
          const srcY = y - dy;
          let val = 0;
          if (srcX >= 0 && srcX < 8 && srcY >= 0 && srcY < 8) {
            val = base[srcY * 8 + srcX];
          }
          // Add subtle pixel noise
          if (Math.random() < 0.05) {
            val = Math.random() < 0.5 ? 0.8 : 0.0;
          }
          // Continuous intensity variations
          sample[y * 8 + x] = Math.max(0, Math.min(1, val * (0.8 + Math.random() * 0.4)));
        }
      }
      dataset.push({ input: Array.from(sample), label: c });
    }
  }

  // Shuffle dataset
  for (let i = dataset.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [dataset[i], dataset[j]] = [dataset[j], dataset[i]];
  }

  return dataset;
}
