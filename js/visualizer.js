/**
 * Canvas Visualizers for Spiking Neural Networks & Biological Neurons
 */

export class NetworkVisualizer {
  constructor(canvas, snn) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.snn = snn;
    this.selectedNeuron = { layer: 1, index: 0 }; // default to hidden neuron 0
    this.pulseAnimations = []; // Traveling synaptic pulses: { x0, y0, x1, y1, progress, color }
    this.setupDPI();
    this.setupInteractivity();
  }

  setupDPI() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.width = rect.width || this.canvas.width;
    this.height = rect.height || this.canvas.height;
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.scale(dpr, dpr);
  }

  setupInteractivity() {
    this.canvas.addEventListener('click', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const nodes = this.getNodePositions();
      for (const node of nodes) {
        const dx = x - node.x;
        const dy = y - node.y;
        if (dx * dx + dy * dy <= node.r * node.r * 1.5) {
          this.selectedNeuron = { layer: node.layerIdx, index: node.neuronIdx };
          if (this.onSelectCallback) this.onSelectCallback(this.selectedNeuron);
          break;
        }
      }
    });
  }

  onSelect(cb) {
    this.onSelectCallback = cb;
  }

  getNodePositions() {
    const nodes = [];
    const layerSizes = this.snn.layerSizes;
    const numLayers = layerSizes.length;
    const paddingX = 70;
    const availableW = this.width - paddingX * 2;
    const layerSpacing = numLayers > 1 ? availableW / (numLayers - 1) : 0;

    for (let l = 0; l < numLayers; l++) {
      const x = paddingX + l * layerSpacing;
      const nNeurons = layerSizes[l];
      const maxDisplay = Math.min(nNeurons, 16); // Cap display to prevent overcrowding
      const paddingY = 40;
      const availableH = this.height - paddingY * 2;
      const neuronSpacing = maxDisplay > 1 ? availableH / (maxDisplay - 1) : 0;

      for (let n = 0; n < maxDisplay; n++) {
        const y = maxDisplay === 1 ? this.height / 2 : paddingY + n * neuronSpacing;
        nodes.push({
          x,
          y,
          r: l === 0 ? 9 : (l === numLayers - 1 ? 14 : 10),
          layerIdx: l,
          neuronIdx: n,
          totalInLayer: nNeurons
        });
      }
    }
    return nodes;
  }

  // Trigger pulse animation along synapses when spikes occur
  triggerSpikes(layerIdx, spikedIndices) {
    const nodes = this.getNodePositions();
    const fromNodes = nodes.filter(nd => nd.layerIdx === layerIdx);
    const toNodes = nodes.filter(nd => nd.layerIdx === layerIdx + 1);

    for (const srcIdx of spikedIndices) {
      const srcNode = fromNodes.find(nd => nd.neuronIdx === srcIdx);
      if (!srcNode) continue;

      for (const tgtNode of toNodes) {
        // Randomly pulse a subset of synapses for aesthetics
        if (Math.random() < 0.65) {
          this.pulseAnimations.push({
            x0: srcNode.x,
            y0: srcNode.y,
            x1: tgtNode.x,
            y1: tgtNode.y,
            progress: 0,
            speed: 0.12 + Math.random() * 0.05,
            color: layerIdx === 0 ? '#00f2fe' : '#a855f7'
          });
        }
      }
    }
  }

  render(activeStepState = null) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    const nodes = this.getNodePositions();
    const layerSizes = this.snn.layerSizes;

    // 1. Draw Synaptic Connections
    for (let l = 0; l < layerSizes.length - 1; l++) {
      const fromNodes = nodes.filter(n => n.layerIdx === l);
      const toNodes = nodes.filter(n => n.layerIdx === l + 1);
      const layer = this.snn.layers[l];

      for (const f of fromNodes) {
        for (const t of toNodes) {
          const w = layer ? layer.weights[t.neuronIdx * layer.nIn + f.neuronIdx] || 0 : 0;
          const alpha = Math.min(0.7, Math.abs(w) * 0.45 + 0.08);

          ctx.beginPath();
          ctx.moveTo(f.x, f.y);
          ctx.lineTo(t.x, t.y);
          ctx.strokeStyle = w >= 0 ? `rgba(0, 242, 254, ${alpha})` : `rgba(244, 63, 94, ${alpha})`;
          ctx.lineWidth = Math.min(3, Math.max(0.6, Math.abs(w) * 1.5));
          ctx.stroke();
        }
      }
    }

    // 2. Animate Traveling Synaptic Pulses
    for (let i = this.pulseAnimations.length - 1; i >= 0; i--) {
      const p = this.pulseAnimations[i];
      p.progress += p.speed;
      if (p.progress >= 1.0) {
        this.pulseAnimations.splice(i, 1);
        continue;
      }

      const curX = p.x0 + (p.x1 - p.x0) * p.progress;
      const curY = p.y0 + (p.y1 - p.y0) * p.progress;

      ctx.beginPath();
      ctx.arc(curX, curY, 3, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // 3. Draw Neurons
    for (const node of nodes) {
      let isSpiking = false;
      let voltage = 0;

      if (activeStepState) {
        if (node.layerIdx === 0) {
          isSpiking = activeStepState.inputSpikes ? activeStepState.inputSpikes[node.neuronIdx] > 0.5 : false;
        } else {
          const lOut = activeStepState.layerOutputs ? activeStepState.layerOutputs[node.layerIdx - 1] : null;
          if (lOut) {
            isSpiking = lOut.spikes ? lOut.spikes[node.neuronIdx] > 0.5 : false;
            voltage = lOut.voltages ? lOut.voltages[node.neuronIdx] || 0 : 0;
          }
        }
      }

      const isSelected = this.selectedNeuron.layer === node.layerIdx && this.selectedNeuron.index === node.neuronIdx;

      // Glow halo
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.r + 5, 0, Math.PI * 2);
      if (isSpiking) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.shadowColor = '#00f2fe';
        ctx.shadowBlur = 18;
      } else if (isSelected) {
        ctx.fillStyle = 'rgba(245, 158, 11, 0.2)';
        ctx.shadowColor = '#f59e0b';
        ctx.shadowBlur = 12;
      } else {
        ctx.fillStyle = 'rgba(15, 23, 42, 0.4)';
        ctx.shadowBlur = 0;
      }
      ctx.fill();
      ctx.shadowBlur = 0;

      // Soma Body
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);

      let bodyGrad = ctx.createRadialGradient(node.x - 2, node.y - 2, 1, node.x, node.y, node.r);
      if (isSpiking) {
        bodyGrad.addColorStop(0, '#ffffff');
        bodyGrad.addColorStop(0.5, '#38bdf8');
        bodyGrad.addColorStop(1, '#0284c7');
      } else if (node.layerIdx === 0) {
        bodyGrad.addColorStop(0, '#38bdf8');
        bodyGrad.addColorStop(1, '#0369a1');
      } else if (node.layerIdx === layerSizes.length - 1) {
        bodyGrad.addColorStop(0, '#fbbf24');
        bodyGrad.addColorStop(1, '#d97706');
      } else {
        // Hidden LIF neuron with potential-dependent hue
        const vNorm = Math.max(0, Math.min(1, voltage));
        bodyGrad.addColorStop(0, `rgb(${Math.round(80 + vNorm * 120)}, 50, ${Math.round(180 + vNorm * 75)})`);
        bodyGrad.addColorStop(1, '#4c1d95');
      }

      ctx.fillStyle = bodyGrad;
      ctx.fill();

      // Border ring
      ctx.lineWidth = isSelected ? 2.5 : 1.2;
      ctx.strokeStyle = isSelected ? '#f59e0b' : (isSpiking ? '#ffffff' : 'rgba(255, 255, 255, 0.3)');
      ctx.stroke();

      // Label for output neurons
      if (node.layerIdx === layerSizes.length - 1) {
        ctx.fillStyle = '#f8fafc';
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`C${node.neuronIdx}`, node.x, node.y);
      }
    }

    // Layer Titles
    ctx.font = '600 11px Outfit, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#94a3b8';
    const firstNodes = nodes.filter(n => n.neuronIdx === 0);
    firstNodes.forEach((fn, idx) => {
      const title = idx === 0 ? 'Input Layer' : (idx === layerSizes.length - 1 ? 'Output LIF' : `Hidden LIF (${layerSizes[idx]})`);
      ctx.fillText(title, fn.x, 20);
    });
  }
}

/**
 * Spike Raster Plot Visualizer
 * Plots time on the X-axis and individual neuron channels on the Y-axis
 */
export class RasterVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.setupDPI();
  }

  setupDPI() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.width = rect.width || this.canvas.width;
    this.height = rect.height || this.canvas.height;
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.scale(dpr, dpr);
  }

  render(rasterData, T = 16, currentT = -1) {
    // rasterData: array of { label: "H0", layer: "hidden", spikes: [0, 1, 0, ...] of length T }
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    if (!rasterData || rasterData.length === 0) return;

    const padLeft = 46;
    const padRight = 18;
    const padTop = 15;
    const padBottom = 26;

    const plotW = this.width - padLeft - padRight;
    const plotH = this.height - padTop - padBottom;
    const nRows = rasterData.length;
    const rowH = plotH / nRows;
    const timeStepW = plotW / T;

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let t = 0; t <= T; t++) {
      const x = padLeft + t * timeStepW;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();

      // Time step numbers
      if (t % 2 === 0 || t === T) {
        ctx.fillStyle = '#64748b';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${t}`, x, this.height - 8);
      }
    }

    // Draw spike ticks
    for (let r = 0; r < nRows; r++) {
      const row = rasterData[r];
      const y = padTop + (r + 0.5) * rowH;

      // Row label
      ctx.fillStyle = row.layer === 'input' ? '#38bdf8' : (row.layer === 'output' ? '#fbbf24' : '#c084fc');
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(row.label, padLeft - 6, y + 3);

      // Spikes
      for (let t = 0; t < T; t++) {
        if (row.spikes[t] > 0.5) {
          const x = padLeft + (t + 0.5) * timeStepW;
          const tickH = Math.max(3, rowH * 0.75);

          ctx.beginPath();
          ctx.moveTo(x, y - tickH / 2);
          ctx.lineTo(x, y + tickH / 2);
          ctx.strokeStyle = row.layer === 'input' ? '#00f2fe' : (row.layer === 'output' ? '#f59e0b' : '#a855f7');
          ctx.lineWidth = 2.5;
          ctx.stroke();
        }
      }
    }

    // Playhead line
    if (currentT >= 0 && currentT < T) {
      const xPlayhead = padLeft + (currentT + 0.5) * timeStepW;
      ctx.beginPath();
      ctx.moveTo(xPlayhead, padTop);
      ctx.lineTo(xPlayhead, padTop + plotH);
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 2]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}

/**
 * Membrane Potential Oscilloscope
 * Plots V(t) trace for selected neuron
 */
export class OscilloscopeVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.setupDPI();
  }

  setupDPI() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.width = rect.width || this.canvas.width;
    this.height = rect.height || this.canvas.height;
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.scale(dpr, dpr);
  }

  render(vTrace, sTrace, vThresh = 1.0, vReset = 0.0, neuronLabel = "Neuron") {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    const padLeft = 40;
    const padRight = 20;
    const padTop = 30;
    const padBottom = 25;
    const plotW = this.width - padLeft - padRight;
    const plotH = this.height - padTop - padBottom;

    const minV = Math.min(-0.2, vReset - 0.2);
    const maxV = Math.max(1.4, vThresh * 1.3);
    const rangeV = maxV - minV;

    const toY = (v) => padTop + plotH - ((v - minV) / rangeV) * plotH;
    const toX = (t, total) => padLeft + (t / (total - 1 || 1)) * plotW;

    // Background Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let v = 0; v <= maxV; v += 0.5) {
      const y = toY(v);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(v.toFixed(1) + 'V', padLeft - 6, y + 3);
    }

    // Threshold Line
    const yThresh = toY(vThresh);
    ctx.beginPath();
    ctx.moveTo(padLeft, yThresh);
    ctx.lineTo(padLeft + plotW, yThresh);
    ctx.strokeStyle = '#f43f5e';
    ctx.setLineDash([5, 3]);
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#f43f5e';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`Vth (${vThresh.toFixed(1)}V)`, this.width - padRight, yThresh - 6);

    // Title / Neuron
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '600 11px Outfit, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Oscilloscope — ${neuronLabel} Membrane Trace`, padLeft, 18);

    if (!vTrace || vTrace.length === 0) return;

    // Plot V(t) waveform
    ctx.beginPath();
    for (let t = 0; t < vTrace.length; t++) {
      const x = toX(t, vTrace.length);
      const y = toY(vTrace[t]);
      if (t === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Mark Action Potentials / Spikes
    if (sTrace) {
      for (let t = 0; t < sTrace.length; t++) {
        if (sTrace[t] > 0.5) {
          const x = toX(t, sTrace.length);
          const y = toY(vTrace[t]);

          // Spike burst indicator
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fillStyle = '#fbbf24';
          ctx.fill();

          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x, padTop);
          ctx.strokeStyle = 'rgba(251, 191, 36, 0.4)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }
    }
  }
}

/**
 * 2D Classification Decision Boundary Visualizer
 */
export class DecisionBoundaryVisualizer {
  constructor(canvas, model, theme = 'cyan') {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.model = model;
    this.snn = model; // Alias for backward compatibility
    this.theme = theme; // 'cyan' or 'amber'
    this.gridResolution = 28;
    this.setupDPI();
  }

  setupDPI() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.width = rect.width || this.canvas.width;
    this.height = rect.height || this.canvas.height;
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.scale(dpr, dpr);
  }

  render(dataset = [], activePoint = null) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    const res = this.gridResolution;
    const cellW = this.width / res;
    const cellH = this.height / res;

    // Draw Heatmap Decision Grid
    for (let gy = 0; gy < res; gy++) {
      for (let gx = 0; gx < res; gx++) {
        const x = (gx + 0.5) / res;
        const y = 1.0 - (gy + 0.5) / res; // Invert Y for Cartesian coordinates

        // Model could be SNN (with timeSteps) or ANN (pure forward)
        const pred = this.model.predict([x, y], 12);
        const p0 = pred.probs[0] || 0;
        const p1 = pred.probs[1] || 0;

        let r, g, b, alpha;
        if (p0 > p1) {
          if (this.theme === 'amber') {
            r = 16; g = 185; b = 129; // Emerald for ANN Class 0
          } else {
            r = 2; g = 132; b = 199; // Cyan for SNN Class 0
          }
          alpha = 0.15 + (p0 - 0.5) * 0.7;
        } else {
          r = 234; g = 88; b = 12; // Orange for Class 1
          alpha = 0.15 + (p1 - 0.5) * 0.7;
        }

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${Math.max(0.1, alpha)})`;
        ctx.fillRect(gx * cellW, gy * cellH, cellW + 0.5, cellH + 0.5);
      }
    }

    // Draw Dataset Points
    for (const pt of dataset) {
      const px = pt.input[0] * this.width;
      const py = (1.0 - pt.input[1]) * this.height;

      ctx.beginPath();
      ctx.arc(px, py, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = pt.label === 0 ? '#38bdf8' : '#fb923c';
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#0f172a';
      ctx.stroke();
    }

    // Highlight active point
    if (activePoint) {
      const px = activePoint.input[0] * this.width;
      const py = (1.0 - activePoint.input[1]) * this.height;

      ctx.beginPath();
      ctx.arc(px, py, 9, 0, Math.PI * 2);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2.5;
      ctx.shadowColor = '#00f2fe';
      ctx.shadowBlur = 10;
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
  }
}

/**
 * Interactive 8x8 Drawing Canvas
 */
export class DrawingCanvas {
  constructor(canvas, onUpdate) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onUpdate = onUpdate;
    this.grid = new Float32Array(64); // 8x8 bitmap
    this.isDrawing = false;
    this.setupDPI();
    this.setupEvents();
  }

  setupDPI() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.width = rect.width || this.canvas.width;
    this.height = rect.height || this.canvas.height;
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.scale(dpr, dpr);
  }

  setupEvents() {
    const handleAction = (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const clientX = e.clientX ?? (e.touches && e.touches[0].clientX);
      const clientY = e.clientY ?? (e.touches && e.touches[0].clientY);
      if (clientX === undefined || clientY === undefined) return;

      const x = clientX - rect.left;
      const y = clientY - rect.top;
      const gx = Math.floor((x / this.width) * 8);
      const gy = Math.floor((y / this.height) * 8);

      if (gx >= 0 && gx < 8 && gy >= 0 && gy < 8) {
        this.paintCell(gx, gy);
        this.render();
        if (this.onUpdate) this.onUpdate(Array.from(this.grid));
      }
    };

    this.canvas.addEventListener('mousedown', (e) => {
      this.isDrawing = true;
      handleAction(e);
    });
    window.addEventListener('mouseup', () => { this.isDrawing = false; });
    this.canvas.addEventListener('mousemove', (e) => {
      if (this.isDrawing) handleAction(e);
    });

    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.isDrawing = true;
      handleAction(e);
    }, { passive: false });
    window.addEventListener('touchend', () => { this.isDrawing = false; });
    this.canvas.addEventListener('touchmove', (e) => {
      if (this.isDrawing) {
        e.preventDefault();
        handleAction(e);
      }
    }, { passive: false });
  }

  paintCell(gx, gy) {
    this.grid[gy * 8 + gx] = 1.0;
    // Soft brush falloff to neighbors
    const neighbors = [
      [-1, 0, 0.4], [1, 0, 0.4], [0, -1, 0.4], [0, 1, 0.4]
    ];
    for (const [dx, dy, w] of neighbors) {
      const nx = gx + dx;
      const ny = gy + dy;
      if (nx >= 0 && nx < 8 && ny >= 0 && ny < 8) {
        this.grid[ny * 8 + nx] = Math.max(this.grid[ny * 8 + nx], w);
      }
    }
  }

  clear() {
    this.grid.fill(0);
    this.render();
    if (this.onUpdate) this.onUpdate(Array.from(this.grid));
  }

  setPattern(patternArray) {
    for (let i = 0; i < 64; i++) {
      this.grid[i] = patternArray[i] || 0;
    }
    this.render();
    if (this.onUpdate) this.onUpdate(Array.from(this.grid));
  }

  render() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    const cellW = this.width / 8;
    const cellH = this.height / 8;

    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const val = this.grid[y * 8 + x];

        // Cell background
        ctx.fillStyle = val > 0 ? `rgba(0, 242, 254, ${val * 0.9 + 0.1})` : '#0f172a';
        ctx.fillRect(x * cellW, y * cellH, cellW, cellH);

        // Grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x * cellW, y * cellH, cellW, cellH);
      }
    }
  }
}

/**
 * Izhikevich Biological Playground Visualizer
 * Plots voltage trace and Phase Plane (v vs u with nullclines)
 */
export class IzhikevichVisualizer {
  constructor(traceCanvas, phaseCanvas) {
    this.traceCanvas = traceCanvas;
    this.phaseCanvas = phaseCanvas;
    this.tCtx = traceCanvas.getContext('2d');
    this.pCtx = phaseCanvas.getContext('2d');
    this.setupDPI();
  }

  setupDPI() {
    const dpr = window.devicePixelRatio || 1;
    [this.traceCanvas, this.phaseCanvas].forEach(c => {
      const rect = c.getBoundingClientRect();
      const w = rect.width || c.width;
      const h = rect.height || c.height;
      c.width = w * dpr;
      c.height = h * dpr;
      c.getContext('2d').scale(dpr, dpr);
    });
    this.tW = this.traceCanvas.getBoundingClientRect().width;
    this.tH = this.traceCanvas.getBoundingClientRect().height;
    this.pW = this.phaseCanvas.getBoundingClientRect().width;
    this.pH = this.phaseCanvas.getBoundingClientRect().height;
  }

  render(neuron) {
    this.renderTrace(neuron);
    this.renderPhasePlane(neuron);
  }

  renderTrace(neuron) {
    const ctx = this.tCtx;
    ctx.clearRect(0, 0, this.tW, this.tH);

    const padLeft = 45;
    const padRight = 15;
    const padTop = 20;
    const padBottom = 25;
    const plotW = this.tW - padLeft - padRight;
    const plotH = this.tH - padTop - padBottom;

    const vMin = -90;
    const vMax = 40;
    const toY = (v) => padTop + plotH - ((v - vMin) / (vMax - vMin)) * plotH;

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    for (let v = -80; v <= 30; v += 20) {
      const y = toY(v);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${v}mV`, padLeft - 6, y + 3);
    }

    // Action potential spike threshold line at +30mV
    const ySpike = toY(30);
    ctx.beginPath();
    ctx.moveTo(padLeft, ySpike);
    ctx.lineTo(padLeft + plotW, ySpike);
    ctx.strokeStyle = '#f43f5e';
    ctx.setLineDash([4, 2]);
    ctx.stroke();
    ctx.setLineDash([]);

    const history = neuron.history;
    if (history.length < 2) return;

    // Draw v(t) membrane waveform
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const x = padLeft + (i / (history.length - 1)) * plotW;
      const y = toY(history[i].v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  renderPhasePlane(neuron) {
    const ctx = this.pCtx;
    ctx.clearRect(0, 0, this.pW, this.pH);

    const pad = 35;
    const plotW = this.pW - pad * 2;
    const plotH = this.pH - pad * 2;

    const vMin = -85;
    const vMax = 35;
    const uMin = -20;
    const uMax = 20;

    const toX = (v) => pad + ((v - vMin) / (vMax - vMin)) * plotW;
    const toY = (u) => pad + plotH - ((u - uMin) / (uMax - uMin)) * plotH;

    // Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, plotW, plotH);

    ctx.fillStyle = '#64748b';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('v (mV)', this.pW / 2, this.pH - 8);
    ctx.save();
    ctx.translate(12, this.pH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('u (recovery)', 0, 0);
    ctx.restore();

    // Nullclines
    const { vNull, uNull } = neuron.getNullclines(neuron.current, vMin, vMax, 80);

    // v-nullcline (yellow)
    ctx.beginPath();
    let started = false;
    for (const pt of vNull) {
      if (pt.u >= uMin && pt.u <= uMax) {
        const x = toX(pt.v);
        const y = toY(pt.u);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else ctx.lineTo(x, y);
      }
    }
    ctx.strokeStyle = '#eab308';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // u-nullcline (magenta)
    ctx.beginPath();
    started = false;
    for (const pt of uNull) {
      if (pt.u >= uMin && pt.u <= uMax) {
        const x = toX(pt.v);
        const y = toY(pt.u);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else ctx.lineTo(x, y);
      }
    }
    ctx.strokeStyle = '#ec4899';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Phase Trajectory (cyan trail)
    const history = neuron.history;
    if (history.length > 2) {
      ctx.beginPath();
      const recent = history.slice(-250);
      for (let i = 0; i < recent.length; i++) {
        const x = toX(recent[i].v);
        const y = toY(recent[i].u);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = 'rgba(0, 242, 254, 0.7)';
      ctx.lineWidth = 1.8;
      ctx.stroke();

      // Current point
      const last = recent[recent.length - 1];
      ctx.beginPath();
      ctx.arc(toX(last.v), toY(last.u), 4.5, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.shadowColor = '#00f2fe';
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }
}
