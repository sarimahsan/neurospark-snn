/**
 * NeuroSpark Application Controller
 * Coordinates SNN Simulation, Visualizers, Training Loops, and User Interactions
 */

import { SpikingNeuralNetwork } from './snn.js';
import { StandardANN } from './ann.js';
import { IzhikevichNeuron, IZHIKEVICH_PRESETS } from './izhikevich.js';
import { DATASETS_2D, DIGIT_TEMPLATES_8X8, generateDigitDataset } from './datasets.js';
import {
  NetworkVisualizer,
  RasterVisualizer,
  OscilloscopeVisualizer,
  DecisionBoundaryVisualizer,
  DrawingCanvas,
  IzhikevichVisualizer
} from './visualizer.js';

class App {
  constructor() {
    this.activeTab = 'lab';
    this.isTraining = false;
    this.epoch = 0;
    this.totalSynapticOps = 0;
    this.totalSpikesFired = 0;

    // SNN Lab Configuration
    this.datasetKey = 'circles';
    this.dataset = DATASETS_2D[this.datasetKey].generate(180);
    this.timeSteps = 16;
    this.beta = 0.85;
    this.vThresh = 1.0;
    this.lr = 0.015;
    this.hiddenSize = 16;
    this.encodingMode = 'rate';

    // Build SNN Model
    this.initSNN();

    // Standard ANN Model for Comparison Benchmark
    this.ann = new StandardANN([2, this.hiddenSize, 2], 0.02);
    this.isCompareTraining = false;
    this.compareEpoch = 0;
    this.compareDataset = this.dataset;

    // Tab 2: Digit SNN (64 -> 32 -> 5)
    this.digitSNN = new SpikingNeuralNetwork([64, 32, 5], {
      timeSteps: 16,
      beta: 0.88,
      vThresh: 1.0,
      lr: 0.02
    });
    this.digitDataset = generateDigitDataset(35);
    this.isTrainingDigits = false;

    // Tab 3: Izhikevich Neuron
    this.izhiNeuron = new IzhikevichNeuron('regularSpiking');

    // Setup Components
    this.initDOM();
    this.initVisualizers();
    this.initEvents();
    this.initIzhikevichPresets();

    // Pre-train digit SNN quickly so it's immediately responsive
    this.bootstrapDigitSNN();

    // Start Animation & Simulation Loops
    this.lastTime = performance.now();
    this.runLoop();
  }

  initSNN() {
    this.snn = new SpikingNeuralNetwork([2, this.hiddenSize, 2], {
      timeSteps: this.timeSteps,
      beta: this.beta,
      vThresh: this.vThresh,
      lr: this.lr,
      encodingMode: this.encodingMode
    });
    this.epoch = 0;
    this.totalSynapticOps = 0;
  }

  initDOM() {
    // Tabs
    this.tabButtons = document.querySelectorAll('.tab-btn');
    this.tabPanes = {
      lab: document.getElementById('pane-lab'),
      draw: document.getElementById('pane-draw'),
      izhi: document.getElementById('pane-izhi'),
      compare: document.getElementById('pane-compare')
    };

    // Telemetry Badges
    this.badgeSpikeRate = document.getElementById('telemetrySpikeRate');
    this.badgeSOPs = document.getElementById('telemetrySOPs');
    this.badgeStatus = document.getElementById('telemetryStatus');

    // Lab Controls
    this.datasetSelect = document.getElementById('datasetSelect');
    this.paramTimeSteps = document.getElementById('paramTimeSteps');
    this.valTimeSteps = document.getElementById('valTimeSteps');
    this.paramBeta = document.getElementById('paramBeta');
    this.valBeta = document.getElementById('valBeta');
    this.paramVThresh = document.getElementById('paramVThresh');
    this.valVThresh = document.getElementById('valVThresh');
    this.paramLR = document.getElementById('paramLR');
    this.valLR = document.getElementById('valLR');
    this.paramHiddenSize = document.getElementById('paramHiddenSize');
    this.paramEncoding = document.getElementById('paramEncoding');

    this.btnTrainToggle = document.getElementById('btnTrainToggle');
    this.trainIcon = document.getElementById('trainIcon');
    this.trainText = document.getElementById('trainText');
    this.btnTrainStep = document.getElementById('btnTrainStep');
    this.btnResetWeights = document.getElementById('btnResetWeights');
    this.btnResampleData = document.getElementById('btnResampleData');

    // Metrics
    this.metricEpoch = document.getElementById('metricEpoch');
    this.metricLoss = document.getElementById('metricLoss');
    this.metricAcc = document.getElementById('metricAcc');
    this.metricSpikes = document.getElementById('metricSpikes');
    this.tagSelectedNeuron = document.getElementById('tagSelectedNeuron');

    // Drawing Studio Controls
    this.btnDrawClear = document.getElementById('btnDrawClear');
    this.btnTrainDigits = document.getElementById('btnTrainDigits');
    this.digitPredBars = document.getElementById('digitPredBars');
    this.digitPredictedBadge = document.getElementById('digitPredictedBadge');

    // Izhikevich Controls
    this.izhiPresetList = document.getElementById('izhiPresetList');
    this.paramIzhiA = document.getElementById('paramIzhiA');
    this.valIzhiA = document.getElementById('valIzhiA');
    this.paramIzhiB = document.getElementById('paramIzhiB');
    this.valIzhiB = document.getElementById('valIzhiB');
    this.paramIzhiC = document.getElementById('paramIzhiC');
    this.valIzhiC = document.getElementById('valIzhiC');
    this.paramIzhiD = document.getElementById('paramIzhiD');
    this.valIzhiD = document.getElementById('valIzhiD');
    this.paramIzhiCurrent = document.getElementById('paramIzhiCurrent');
    this.valIzhiCurrent = document.getElementById('valIzhiCurrent');
    this.btnIzhiReset = document.getElementById('btnIzhiReset');
    this.izhiActiveName = document.getElementById('izhiActiveName');

    // Comparison Controls
    this.compareDatasetSelect = document.getElementById('compareDatasetSelect');
    this.btnTrainCompareToggle = document.getElementById('btnTrainCompareToggle');
    this.compareTrainIcon = document.getElementById('compareTrainIcon');
    this.compareTrainText = document.getElementById('compareTrainText');
    this.btnStepCompare = document.getElementById('btnStepCompare');
    this.btnResetCompare = document.getElementById('btnResetCompare');

    this.tagCompareSnnAcc = document.getElementById('tagCompareSnnAcc');
    this.tagCompareAnnAcc = document.getElementById('tagCompareAnnAcc');
    this.cmpMetricSops = document.getElementById('cmpMetricSops');
    this.cmpMetricFlops = document.getElementById('cmpMetricFlops');
    this.cmpMetricSnnEnergy = document.getElementById('cmpMetricSnnEnergy');
    this.cmpMetricAnnEnergy = document.getElementById('cmpMetricAnnEnergy');
    this.cmpMetricEnergyRatio = document.getElementById('cmpMetricEnergyRatio');
    this.cmpMetricSnnLoss = document.getElementById('cmpMetricSnnLoss');
    this.cmpMetricAnnLoss = document.getElementById('cmpMetricAnnLoss');
    this.cmpMetricWinning = document.getElementById('cmpMetricWinning');

    this.initPredBars();
  }

  initVisualizers() {
    // Tab 1 Visualizers
    const cNet = document.getElementById('canvasNetwork');
    const cBound = document.getElementById('canvasBoundary');
    const cRast = document.getElementById('canvasRaster');
    const cOsc = document.getElementById('canvasOscilloscope');

    this.netVis = new NetworkVisualizer(cNet, this.snn);
    this.boundaryVis = new DecisionBoundaryVisualizer(cBound, this.snn);
    this.rasterVis = new RasterVisualizer(cRast);
    this.oscVis = new OscilloscopeVisualizer(cOsc);

    this.netVis.onSelect((sel) => {
      const name = sel.layer === 0 ? `Input #${sel.index}` :
        (sel.layer === this.snn.layerSizes.length - 1 ? `Output Class #${sel.index}` : `Hidden #${sel.index}`);
      this.tagSelectedNeuron.textContent = name;
      this.updateOscilloscope();
    });

    // Tab 2 Visualizers
    const cDrawPad = document.getElementById('canvasDrawPad');
    const cDrawRast = document.getElementById('canvasDrawRaster');

    this.drawCanvas = new DrawingCanvas(cDrawPad, (grid) => {
      this.testDrawnPattern(grid);
    });
    this.drawRasterVis = new RasterVisualizer(cDrawRast);

    // Tab 3 Visualizers
    const cIzhiTrace = document.getElementById('canvasIzhiTrace');
    const cIzhiPhase = document.getElementById('canvasIzhiPhase');
    this.izhiVis = new IzhikevichVisualizer(cIzhiTrace, cIzhiPhase);

    // Tab 4 Comparison Visualizers
    const cCmpSnn = document.getElementById('canvasCompareSNN');
    const cCmpAnn = document.getElementById('canvasCompareANN');
    this.cmpSnnVis = new DecisionBoundaryVisualizer(cCmpSnn, this.snn, 'cyan');
    this.cmpAnnVis = new DecisionBoundaryVisualizer(cCmpAnn, this.ann, 'amber');

    // Initial render
    this.boundaryVis.render(this.dataset);
    this.netVis.render();
    this.cmpSnnVis.render(this.compareDataset);
    this.cmpAnnVis.render(this.compareDataset);
    this.runSingleInference();
  }

  initPredBars() {
    this.digitPredBars.innerHTML = '';
    for (let i = 0; i < 5; i++) {
      const row = document.createElement('div');
      row.className = 'pred-row';
      row.id = `predRow${i}`;
      row.innerHTML = `
        <span class="pred-label">Digit ${i}</span>
        <div class="pred-track">
          <div class="pred-fill" id="predFill${i}"></div>
        </div>
        <span class="pred-rate" id="predRate${i}">0%</span>
      `;
      this.digitPredBars.appendChild(row);
    }
  }

  initIzhikevichPresets() {
    this.izhiPresetList.innerHTML = '';
    Object.entries(IZHIKEVICH_PRESETS).forEach(([key, p]) => {
      const card = document.createElement('div');
      card.className = `preset-card ${key === 'regularSpiking' ? 'active' : ''}`;
      card.dataset.key = key;
      card.innerHTML = `
        <div class="preset-card-title">
          <span>${p.name}</span>
          <span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--neon-cyan);">I=${p.defaultCurrent}</span>
        </div>
        <div class="preset-card-desc">${p.desc}</div>
      `;

      card.addEventListener('click', () => {
        document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        this.izhiNeuron.setPreset(key);
        this.izhiNeuron.reset();
        this.izhiActiveName.textContent = p.name;

        // Update slider values
        this.paramIzhiA.value = p.a;
        this.valIzhiA.textContent = p.a.toFixed(2);
        this.paramIzhiB.value = p.b;
        this.valIzhiB.textContent = p.b.toFixed(2);
        this.paramIzhiC.value = p.c;
        this.valIzhiC.textContent = `${p.c}mV`;
        this.paramIzhiD.value = p.d;
        this.valIzhiD.textContent = p.d.toFixed(1);
        this.paramIzhiCurrent.value = p.defaultCurrent;
        this.valIzhiCurrent.textContent = p.defaultCurrent.toFixed(1);
      });

      this.izhiPresetList.appendChild(card);
    });
  }

  initEvents() {
    // Navigation Tabs
    this.tabButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        this.activeTab = tab;

        this.tabButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        Object.entries(this.tabPanes).forEach(([t, pane]) => {
          pane.classList.toggle('active', t === tab);
        });

        // Trigger redraw on resize/tab change
        if (tab === 'lab') {
          this.netVis.setupDPI();
          this.boundaryVis.setupDPI();
          this.rasterVis.setupDPI();
          this.oscVis.setupDPI();
          this.boundaryVis.render(this.dataset);
          this.netVis.render();
          this.updateOscilloscope();
        } else if (tab === 'draw') {
          this.drawCanvas.setupDPI();
          this.drawRasterVis.setupDPI();
          this.drawCanvas.render();
          this.testDrawnPattern(this.drawCanvas.grid);
        } else if (tab === 'izhi') {
          this.izhiVis.setupDPI();
        } else if (tab === 'compare') {
          this.cmpSnnVis.setupDPI();
          this.cmpAnnVis.setupDPI();
          this.cmpSnnVis.render(this.compareDataset);
          this.cmpAnnVis.render(this.compareDataset);
        }
      });
    });

    // Dataset Select
    this.datasetSelect.addEventListener('change', (e) => {
      this.datasetKey = e.target.value;
      this.dataset = DATASETS_2D[this.datasetKey].generate(180);
      this.initSNN();
      this.netVis.snn = this.snn;
      this.boundaryVis.snn = this.snn;
      this.boundaryVis.render(this.dataset);
      this.netVis.render();
      this.runSingleInference();
    });

    // Hyperparameter Sliders
    this.paramTimeSteps.addEventListener('input', (e) => {
      this.timeSteps = parseInt(e.target.value);
      this.valTimeSteps.textContent = this.timeSteps;
      this.snn.timeSteps = this.timeSteps;
      this.runSingleInference();
    });

    this.paramBeta.addEventListener('input', (e) => {
      this.beta = parseFloat(e.target.value);
      this.valBeta.textContent = this.beta.toFixed(2);
      this.snn.layers.forEach(l => l.beta = this.beta);
    });

    this.paramVThresh.addEventListener('input', (e) => {
      this.vThresh = parseFloat(e.target.value);
      this.valVThresh.textContent = `${this.vThresh.toFixed(1)}V`;
      this.snn.layers.forEach(l => l.vThresh = this.vThresh);
      this.updateOscilloscope();
    });

    this.paramLR.addEventListener('input', (e) => {
      this.lr = parseFloat(e.target.value);
      this.valLR.textContent = this.lr.toFixed(3);
      this.snn.lr = this.lr;
    });

    this.paramHiddenSize.addEventListener('change', (e) => {
      this.hiddenSize = parseInt(e.target.value);
      this.initSNN();
      this.netVis.snn = this.snn;
      this.boundaryVis.snn = this.snn;
      this.boundaryVis.render(this.dataset);
      this.netVis.render();
      this.runSingleInference();
    });

    this.paramEncoding.addEventListener('change', (e) => {
      this.encodingMode = e.target.value;
      this.snn.encodingMode = this.encodingMode;
      this.runSingleInference();
    });

    // Training Buttons
    this.btnTrainToggle.addEventListener('click', () => {
      this.isTraining = !this.isTraining;
      if (this.isTraining) {
        this.trainIcon.textContent = '⏸';
        this.trainText.textContent = 'Pause Training';
        this.badgeStatus.textContent = 'Training';
        this.badgeStatus.style.color = '#00f2fe';
      } else {
        this.trainIcon.textContent = '▶';
        this.trainText.textContent = 'Train Network';
        this.badgeStatus.textContent = 'Idle';
        this.badgeStatus.style.color = '#10b981';
      }
    });

    this.btnTrainStep.addEventListener('click', () => {
      this.trainOneEpoch();
    });

    this.btnResetWeights.addEventListener('click', () => {
      this.isTraining = false;
      this.trainIcon.textContent = '▶';
      this.trainText.textContent = 'Train Network';
      this.badgeStatus.textContent = 'Reset';
      this.initSNN();
      this.netVis.snn = this.snn;
      this.boundaryVis.snn = this.snn;
      this.boundaryVis.render(this.dataset);
      this.netVis.render();
      this.metricEpoch.textContent = '0';
      this.metricLoss.textContent = '0.000';
      this.metricAcc.textContent = '0.0%';
      this.runSingleInference();
    });

    this.btnResampleData.addEventListener('click', () => {
      this.dataset = DATASETS_2D[this.datasetKey].generate(180);
      this.boundaryVis.render(this.dataset);
      this.runSingleInference();
    });

    // Drawing Studio Preset Pills
    document.querySelectorAll('.pill-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const digit = parseInt(btn.dataset.digit);
        const template = DIGIT_TEMPLATES_8X8[digit];
        if (template) {
          this.drawCanvas.setPattern(template);
        }
      });
    });

    this.btnDrawClear.addEventListener('click', () => {
      this.drawCanvas.clear();
    });

    this.btnTrainDigits.addEventListener('click', () => {
      this.trainDigitEpochs(15);
    });

    // Izhikevich Sliders
    this.paramIzhiA.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      this.valIzhiA.textContent = val.toFixed(2);
      this.izhiNeuron.a = val;
    });

    this.paramIzhiB.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      this.valIzhiB.textContent = val.toFixed(2);
      this.izhiNeuron.b = val;
    });

    this.paramIzhiC.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      this.valIzhiC.textContent = `${val}mV`;
      this.izhiNeuron.c = val;
    });

    this.paramIzhiD.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      this.valIzhiD.textContent = val.toFixed(1);
      this.izhiNeuron.d = val;
    });

    this.paramIzhiCurrent.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      this.valIzhiCurrent.textContent = val.toFixed(1);
      this.izhiNeuron.current = val;
    });

    this.btnIzhiReset.addEventListener('click', () => {
      this.izhiNeuron.reset();
    });

    // Comparison Tab Events
    this.compareDatasetSelect.addEventListener('change', (e) => {
      const key = e.target.value;
      this.compareDataset = DATASETS_2D[key].generate(180);
      this.cmpSnnVis.render(this.compareDataset);
      this.cmpAnnVis.render(this.compareDataset);
    });

    this.btnTrainCompareToggle.addEventListener('click', () => {
      this.isCompareTraining = !this.isCompareTraining;
      if (this.isCompareTraining) {
        this.compareTrainIcon.textContent = '⏸';
        this.compareTrainText.textContent = 'Pause Benchmark';
      } else {
        this.compareTrainIcon.textContent = '▶';
        this.compareTrainText.textContent = 'Train Both Simultaneously';
      }
    });

    this.btnStepCompare.addEventListener('click', () => {
      this.trainCompareOneEpoch();
    });

    this.btnResetCompare.addEventListener('click', () => {
      this.isCompareTraining = false;
      this.compareTrainIcon.textContent = '▶';
      this.compareTrainText.textContent = 'Train Both Simultaneously';
      this.compareEpoch = 0;
      this.initSNN();
      this.ann = new StandardANN([2, this.hiddenSize, 2], 0.02);
      this.cmpSnnVis.model = this.snn;
      this.cmpAnnVis.model = this.ann;
      this.cmpSnnVis.render(this.compareDataset);
      this.cmpAnnVis.render(this.compareDataset);
      this.tagCompareSnnAcc.textContent = 'Accuracy: 0.0%';
      this.tagCompareAnnAcc.textContent = 'Accuracy: 0.0%';
      this.cmpMetricSnnLoss.textContent = 'Loss: 0.69 | Acc: 0.0%';
      this.cmpMetricAnnLoss.textContent = 'Loss: 0.69 | Acc: 0.0%';
    });

    // Handle Window Resizing
    window.addEventListener('resize', () => {
      this.netVis.setupDPI();
      this.boundaryVis.setupDPI();
      this.rasterVis.setupDPI();
      this.oscVis.setupDPI();
      this.drawCanvas.setupDPI();
      this.drawRasterVis.setupDPI();
      this.izhiVis.setupDPI();
      this.cmpSnnVis.setupDPI();
      this.cmpAnnVis.setupDPI();
    });
  }

  // Pre-train the 64 -> 32 -> 5 digit SNN with surrogate BPTT
  bootstrapDigitSNN() {
    for (let epoch = 0; epoch < 25; epoch++) {
      for (const sample of this.digitDataset) {
        this.digitSNN.trainSample(sample.input, sample.label, 0.025);
      }
    }
  }

  trainDigitEpochs(numEpochs = 10) {
    this.btnTrainDigits.textContent = 'Training...';
    let count = 0;
    const stepInterval = setInterval(() => {
      for (let s = 0; s < this.digitDataset.length; s++) {
        const item = this.digitDataset[s];
        this.digitSNN.trainSample(item.input, item.label, 0.025);
      }
      count++;
      if (count >= numEpochs) {
        clearInterval(stepInterval);
        this.btnTrainDigits.textContent = 'Train Digits';
        this.testDrawnPattern(this.drawCanvas.grid);
      }
    }, 40);
  }

  testDrawnPattern(grid) {
    const pred = this.digitSNN.predict(Array.from(grid), 16);
    const spikeCounts = pred.spikeCounts;
    const probs = pred.probs;

    let maxIdx = pred.predictedClass;
    let sumSpikes = 0;
    for (let i = 0; i < 5; i++) sumSpikes += spikeCounts[i];

    this.digitPredictedBadge.textContent = sumSpikes > 0 ? `Predicted: Digit ${maxIdx}` : 'No Input';

    // Update Bar UI
    for (let i = 0; i < 5; i++) {
      const pct = Math.round(probs[i] * 100);
      const row = document.getElementById(`predRow${i}`);
      const fill = document.getElementById(`predFill${i}`);
      const rate = document.getElementById(`predRate${i}`);

      if (fill && rate) {
        fill.style.width = `${pct}%`;
        rate.textContent = `${pct}% (${spikeCounts[i]})`;
        row.classList.toggle('top-pred', i === maxIdx && sumSpikes > 0);
      }
    }

    // Render Spike Raster for Digit Recognition
    const T = 16;
    const rasterRows = [];

    // Show a sample of active input neurons
    const activeInputs = [];
    for (let i = 0; i < 64; i++) {
      if (grid[i] > 0.1) activeInputs.push(i);
    }
    const sampledInputs = activeInputs.slice(0, 6);
    for (const inpIdx of sampledInputs) {
      const spikes = [];
      for (let t = 0; t < T; t++) {
        spikes.push(pred.inputSeq[t * 64 + inpIdx]);
      }
      rasterRows.push({ label: `In${inpIdx}`, layer: 'input', spikes });
    }

    // Sample hidden neurons (e.g. 6 neurons)
    const hiddenSpikes = pred.layerOutputs[0].sHist;
    for (let h = 0; h < 6; h++) {
      const spikes = [];
      for (let t = 0; t < T; t++) {
        spikes.push(hiddenSpikes[t * 32 + h]);
      }
      rasterRows.push({ label: `H${h}`, layer: 'hidden', spikes });
    }

    // Output neurons (all 5)
    const outSpikes = pred.layerOutputs[1].sHist;
    for (let o = 0; o < 5; o++) {
      const spikes = [];
      for (let t = 0; t < T; t++) {
        spikes.push(outSpikes[t * 5 + o]);
      }
      rasterRows.push({ label: `C${o}`, layer: 'output', spikes });
    }

    this.drawRasterVis.render(rasterRows, T);
  }

  trainOneEpoch() {
    let totalLoss = 0;
    let correctCount = 0;
    let epochSpikes = 0;

    for (let i = 0; i < this.dataset.length; i++) {
      const sample = this.dataset[i];
      const res = this.snn.trainSample(sample.input, sample.label, this.lr);

      totalLoss += res.loss;
      if (res.isCorrect) correctCount++;

      // Tally spikes for synaptic operations
      for (let j = 0; j < res.spikeCounts.length; j++) {
        epochSpikes += res.spikeCounts[j];
      }

      // Trigger synaptic pulses for a few samples
      if (i % 25 === 0) {
        const spikedHidden = [];
        const hiddenSpikes = res.layerOutputs[0].sHist;
        for (let h = 0; h < this.hiddenSize; h++) {
          if (hiddenSpikes[h] > 0.5) spikedHidden.push(h);
        }
        this.netVis.triggerSpikes(1, spikedHidden);
      }
    }

    this.epoch++;
    const avgLoss = totalLoss / this.dataset.length;
    const accuracy = (correctCount / this.dataset.length) * 100;

    // Synaptic Operations: fan-in synaptic weight additions only triggered upon pre-synaptic spikes
    const sopsPerEpoch = epochSpikes * this.hiddenSize;
    this.totalSynapticOps += sopsPerEpoch;
    this.totalSpikesFired += epochSpikes;

    // Update DOM Metrics
    this.metricEpoch.textContent = this.epoch;
    this.metricLoss.textContent = avgLoss.toFixed(3);
    this.metricAcc.textContent = `${accuracy.toFixed(1)}%`;
    this.metricSpikes.textContent = epochSpikes;

    this.badgeSOPs.textContent = `${(this.totalSynapticOps / 1000).toFixed(1)}k SOPs`;
    const rateHz = (epochSpikes / (this.dataset.length * (this.timeSteps * 0.001))).toFixed(1);
    this.badgeSpikeRate.textContent = `${rateHz} Hz`;

    // Redraw decision boundary
    this.boundaryVis.render(this.dataset);
    this.netVis.render();
    this.updateOscilloscope();
  }

  runSingleInference() {
    if (!this.dataset || this.dataset.length === 0) return;
    const testSample = this.dataset[0];
    const pred = this.snn.predict(testSample.input, this.timeSteps);
    this.lastInference = { sample: testSample, pred };

    // Format raster rows
    const T = this.timeSteps;
    const rasterRows = [];

    // Input spikes
    for (let i = 0; i < 2; i++) {
      const spikes = [];
      for (let t = 0; t < T; t++) spikes.push(pred.inputSeq[t * 2 + i]);
      rasterRows.push({ label: `In${i}`, layer: 'input', spikes });
    }

    // Hidden spikes
    const hiddenHist = pred.layerOutputs[0].sHist;
    const maxShowH = Math.min(this.hiddenSize, 8);
    for (let h = 0; h < maxShowH; h++) {
      const spikes = [];
      for (let t = 0; t < T; t++) spikes.push(hiddenHist[t * this.hiddenSize + h]);
      rasterRows.push({ label: `H${h}`, layer: 'hidden', spikes });
    }

    // Output spikes
    const outHist = pred.layerOutputs[1].sHist;
    for (let o = 0; o < 2; o++) {
      const spikes = [];
      for (let t = 0; t < T; t++) spikes.push(outHist[t * 2 + o]);
      rasterRows.push({ label: `Out${o}`, layer: 'output', spikes });
    }

    this.rasterVis.render(rasterRows, T);
    this.updateOscilloscope();
  }

  updateOscilloscope() {
    if (!this.lastInference) return;
    const { pred } = this.lastInference;
    const sel = this.netVis.selectedNeuron;
    const T = this.timeSteps;

    let vTrace = [];
    let sTrace = [];

    if (sel.layer === 0) {
      // Input has no membrane potential; synthetic trace
      for (let t = 0; t < T; t++) {
        const spk = pred.inputSeq[t * 2 + sel.index];
        vTrace.push(spk ? 1.0 : 0.0);
        sTrace.push(spk);
      }
    } else {
      const lIdx = sel.layer - 1;
      const lOut = pred.layerOutputs[lIdx];
      const nTotal = this.snn.layers[lIdx].nOut;
      for (let t = 0; t < T; t++) {
        vTrace.push(lOut.vHist[t * nTotal + sel.index]);
        sTrace.push(lOut.sHist[t * nTotal + sel.index]);
      }
    }

    const neuronLabel = sel.layer === 0 ? `Input #${sel.index}` :
      (sel.layer === this.snn.layerSizes.length - 1 ? `Output Class #${sel.index}` : `Hidden #${sel.index}`);

    this.oscVis.render(vTrace, sTrace, this.vThresh, 0.0, neuronLabel);
  }

  trainCompareOneEpoch() {
    let snnLoss = 0, annLoss = 0;
    let snnCorrect = 0, annCorrect = 0;
    let snnSpikes = 0;
    const len = this.compareDataset.length;

    // Train both on each sample
    for (let i = 0; i < len; i++) {
      const sample = this.compareDataset[i];
      // Train SNN with surrogate BPTT
      const sRes = this.snn.trainSample(sample.input, sample.label, 0.02);
      snnLoss += sRes.loss;
      if (sRes.isCorrect) snnCorrect++;
      for (let j = 0; j < sRes.spikeCounts.length; j++) snnSpikes += sRes.spikeCounts[j];

      // Train ANN with standard backprop
      const aRes = this.ann.trainSample(sample.input, sample.label, 0.02);
      annLoss += aRes.loss;
      if (aRes.isCorrect) annCorrect++;
    }

    this.compareEpoch++;
    const snnAcc = (snnCorrect / len) * 100;
    const annAcc = (annCorrect / len) * 100;
    const avgSnnLoss = snnLoss / len;
    const avgAnnLoss = annLoss / len;

    // FLOPs vs SOPs
    const annFlopsPerSample = this.ann.getFlopsPerInference();
    const sopsPerSample = Math.max(1, Math.round((snnSpikes / len) * this.hiddenSize));

    // Energy estimates (neuromorphic CMOS 0.9 pJ / SOP vs GPU/digital MAC 4.6 pJ / FLOP)
    const snnEnergyPj = (sopsPerSample * 0.9);
    const annEnergyPj = (annFlopsPerSample * 4.6);
    const energyRatio = (annEnergyPj / Math.max(0.1, snnEnergyPj)).toFixed(1);

    // Update tags
    this.tagCompareSnnAcc.textContent = `Accuracy: ${snnAcc.toFixed(1)}%`;
    this.tagCompareAnnAcc.textContent = `Accuracy: ${annAcc.toFixed(1)}%`;
    this.cmpMetricSops.textContent = `~${sopsPerSample} SOPs (Additions)`;
    this.cmpMetricFlops.textContent = `${annFlopsPerSample} FLOPs (Multiplications)`;
    this.cmpMetricSnnEnergy.textContent = `~${snnEnergyPj.toFixed(2)} pJ / inference`;
    this.cmpMetricAnnEnergy.textContent = `~${annEnergyPj.toFixed(2)} pJ / inference`;
    this.cmpMetricEnergyRatio.textContent = `~${energyRatio}× Lower Dynamic Energy!`;
    this.cmpMetricSnnLoss.textContent = `Loss: ${avgSnnLoss.toFixed(2)} | Acc: ${snnAcc.toFixed(1)}%`;
    this.cmpMetricAnnLoss.textContent = `Loss: ${avgAnnLoss.toFixed(2)} | Acc: ${annAcc.toFixed(1)}%`;

    if (snnAcc > annAcc) {
      this.cmpMetricWinning.textContent = `⚡ SNN leads by +${(snnAcc - annAcc).toFixed(1)}%`;
      this.cmpMetricWinning.style.color = '#00f2fe';
    } else if (annAcc > snnAcc) {
      this.cmpMetricWinning.textContent = `🤖 ANN leads by +${(annAcc - snnAcc).toFixed(1)}%`;
      this.cmpMetricWinning.style.color = '#f59e0b';
    } else {
      this.cmpMetricWinning.textContent = `Tied at ${snnAcc.toFixed(1)}%`;
      this.cmpMetricWinning.style.color = '#10b981';
    }

    // Render both canvases
    this.cmpSnnVis.render(this.compareDataset);
    this.cmpAnnVis.render(this.compareDataset);
  }

  runLoop() {
    const loop = () => {
      // 1. Train Lab SNN if active
      if (this.activeTab === 'lab' && this.isTraining) {
        this.trainOneEpoch();
      }

      // 2. Animate Traveling Pulses & Network State
      if (this.activeTab === 'lab') {
        this.netVis.render();
      }

      // 3. Train Comparison Benchmark if active
      if (this.activeTab === 'compare' && this.isCompareTraining) {
        this.trainCompareOneEpoch();
      }

      // 4. Izhikevich Simulation Step if Tab 3 is active
      if (this.activeTab === 'izhi') {
        // Step biological equations 3x per frame for smooth 60fps dynamics
        for (let s = 0; s < 3; s++) {
          this.izhiNeuron.step(this.izhiNeuron.current, 0.5);
        }
        this.izhiVis.render(this.izhiNeuron);
      }

      requestAnimationFrame(loop);
    };

    requestAnimationFrame(loop);
  }
}

// Instantiate application on DOM ready
window.addEventListener('DOMContentLoaded', () => {
  window.app = new App();
});
