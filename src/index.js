// @ts-nocheck
/* global tf, tfvis, Plotly */

// import Plotly from 'plotly.js-dist'; // <https://plotly.com/javascript/>
// import * as tf from '@tensorflow/tfjs'; // <https://js.tensorflow.org/api/latest/>
import * as wasm from '@tensorflow/tfjs-backend-wasm';
import * as model from './model.js';
import Menu from './menu.js';

let data = { input: [], validation: [], prediction: [], stats: {} };

let stock = {
  symbol: 'btc-usd',
  interval: '1d',
  range: '1y',
};

let params = {
  backend: 'webgl',
  dtype: 'float32',
  evalError: 0.1,
  smaError: 2.5,
  visor: false,

  inputWindow: 44,
  outputWindow: 1,
  predictWindow: 44,
  epochs: 25,
  validationSplit: 0,
  optimizer: 'adam',
  learningRate: 0.001,
  loss: 'meanSquaredError',
  targetLoss: 0.03,

  neurons: 88,
  features: 4,
  layers: 3,
  cells: 'lstmCell',
  kernelInitializer: 'leCunNormal',
  activation: 'sigmoid',
  recurrentActivation: 'tanh',
  // constraint: 'unitNorm',

  forgetBias: false,
  biasInitializer: 'glorotNormal',
  shuffle: true,
};

const markets = [
  { d: '', s: '', f: '' },
  { d: 'S&P 500', s: '^GSPC', f: 'ES=F' },
  { d: 'Dow Jones', s: '^DJI', f: 'YM=F' },
  { d: 'NASDAQ', s: '^IXIC', f: 'NQ=F' },
];

const sectors = [
  { d: '', s: '' },
  { d: 'Consumer', s: '^SPSDYUP' },
  { d: 'Health', s: '^SPSDVUP' },
  { d: 'Industry', s: '^SPSDIUP' },
  { d: 'Tech', s: '^SPSDTUP' },
  { d: 'Material', s: '^SPSDBUP' },
  { d: 'RealEstate', s: '^SPSDREUP' },
  { d: 'Comms', s: '^SPSDCSUN' },
  { d: 'Utilities', s: '^SPSDUUP' },
  { d: 'Finance', s: '^SPSDMUP' },
  { d: 'Energy', s: '^SPSDEUP' },
];

const chart = {
  element: null,
  data: [],
  layout: {
    xaxis: {
      type: 'date',
      autorange: true,
      showgrid: true,
      zeroline: true,
      showline: true,
      autotick: true,
      // dtick: 15 * 1000 * 60 * 60 * 24,
      showticklabels: true,
      gridcolor: '#555555',
    },
    yaxis: {
      autorange: true,
      rangemode: 'tozero',
      showgrid: true,
      zeroline: true,
      showline: true,
      autotick: true,
      // dtick: 10,
      tickprefix: '$',
      separatethousands: true,
      showticklabels: true,
      gridcolor: '#444444',
    },
    font: {
      family: 'system-ui',
      color: '#FFFFFF',
    },
    plot_bgcolor: '#222222',
    paper_bgcolor: '#000000',
    margin: { l: 60, r: 20, t: 60, b: 20 },
    title: '',
  },
  options: {
    scrollZoom: true,
    responsive: true,
    displaylogo: false,
  },
};

function str(...msg) {
  if (!Array.isArray(msg)) return msg;
  let line = '';
  for (const entry of msg) {
    if (typeof entry === 'object') line += JSON.stringify(entry).replace(/{|}|"|\[|\]/g, '').replace(/,/g, ' | ');
    else line += entry;
  }
  return line;
}

function ok(bool, msg) {
  return bool ? `<font color=lightgreen>${msg || 'OK'}</font>` : `<font color=lightcoral>${msg || 'ERR'}</font>`;
}

function log(...msg) {
  const dt = new Date();
  const ts = `${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}:${dt.getSeconds().toString().padStart(2, '0')}.${dt.getMilliseconds().toString().padStart(3, '0')}`;
  // eslint-disable-next-line no-console
  console.log(ts, ...msg);
  const div = document.getElementById('log') || document.createElement('div');
  div.innerHTML += `<font color=gray>${ts}</font> &nbsp ${str(msg)}<br>`;
  div.scrollTop = div.scrollHeight;
}

function advice(...msg) {
  const div = document.getElementById('advice') || document.createElement('div');
  div.innerHTML += `${str(msg)}<br>`;
  div.scrollTop = div.scrollHeight;
}

function computeWindow(input, inputWindow, outputWindow = 1) {
  const arr = [];
  for (let i = inputWindow; i <= (input.length - outputWindow); i++) {
    const inputSet = []; // history: create set of values up to index
    for (let j = inputWindow; j >= 1; j--) inputSet.push(input[i - j]);
    let outputSet = []; // future
    if (outputWindow === 1) { // outputSet is single value
      outputSet = input[i];
    } else { // create set of values starting with index
      for (let j = i; j < (i + outputWindow); j++) outputSet.push(input[j]);
    }
    const sma = inputSet.reduce((sum, val) => sum += val, 0) / inputWindow;
    const value = input[i];
    arr.push({ value, sma, inputSet, outputSet });
  }
  return arr;
}

async function drawGraph() {
  if (!data.input) return;
  const maxPrice = Math.max(...data.input.adjusted);
  const maxVolume = Math.max(...data.input.volume);
  chart.data = [];
  chart.data.push({
    name: 'Price',
    x: data.input.time,
    y: data.input.adjusted,
    type: 'lines',
    line: { color: 'lightblue', shape: 'spline', width: 3 },
  });
  chart.data.push({
    name: 'OHLC',
    x: data.input.time,
    open: data.input.open,
    close: data.input.close,
    high: data.input.high,
    low: data.input.low,
    type: 'candlestick',
  });
  chart.data.push({
    name: 'Volume',
    x: data.input.time,
    y: data.input.volume.map((val) => maxPrice * val / maxVolume / 2),
    type: 'bar',
    marker: { color: 'steelblue' },
  });
  // chart.layout.xaxis.dtick = Math.trunc(data.input.time[0] / 1000);
  // chart.layout.yaxis.dtick = Math.trunc(Math.max(...data.input.adjusted) / 10);
  chart.layout.title = `${data.input.type}: ${data.input.exchange}/${data.input.symbol} [${data.input.range}/${data.input.granularity}]`;
  Plotly.newPlot(document.getElementById('graph'), chart.data, chart.layout, chart.options);
}

async function getData() {
  const url = `https://query2.finance.yahoo.com/v8/finance/chart/${encodeURI(stock.symbol)}?range=${stock.range}&interval=${stock.interval}&indicators=quote&includeTimestamps=true&includePrePost=true`;
  const res = await fetch(`${location.href}cors`, { headers: { cors: url } });
  const json = (res && res.ok) ? await res.json() : {};
  if (!json.chart || !json.chart.result) {
    log('data error:', stock.symbol);
    return;
  }
  data.input = {
    type: json.chart.result[0].meta.instrumentType,
    exchange: json.chart.result[0].meta.exchangeName,
    symbol: json.chart.result[0].meta.symbol,
    range: json.chart.result[0].meta.range,
    granularity: json.chart.result[0].meta.dataGranularity,
    adjusted: json.chart.result[0].indicators.adjclose
      ? json.chart.result[0].indicators.adjclose[0].adjclose.map((val) => parseFloat(val))
      : json.chart.result[0].indicators.quote[0].close.map((val) => parseFloat(val)),
    volume: json.chart.result[0].indicators.quote[0].volume.map((val) => parseFloat(val)),
    open: json.chart.result[0].indicators.quote[0].open.map((val) => parseFloat(val)),
    high: json.chart.result[0].indicators.quote[0].high.map((val) => parseFloat(val)),
    low: json.chart.result[0].indicators.quote[0].low.map((val) => parseFloat(val)),
    close: json.chart.result[0].indicators.quote[0].close.map((val) => parseFloat(val)),
    time: json.chart.result[0].timestamp.map((val) => 1000 * parseInt(val)),
  };
  advice(ok(data.input.adjusted && data.input.adjusted.length > 0), `Data: ${data.input.type}: ${data.input.exchange}/${data.input.symbol} [${data.input.range}/${data.input.granularity}]`);
  await drawGraph();
  advice(ok(data.input.adjusted.length > 250), `Data set size: ${data.input.adjusted.length}`);
}

async function trainModel(input) {
  if (!input) return;
  advice('');
  advice('Training', params);

  if (params.neurons % params.features !== 0) {
    advice(ok(false), 'Params error: neurons must be divisible by features');
    return;
  }
  const ma = computeWindow(input, params.inputWindow, params.outputWindow);
  const inputs = ma.map((val) => val.inputSet);
  const outputs = ma.map((val) => val.outputSet);
  // train graph
  const lossData = [{
    x: [],
    y: [],
    name: 'Epoch',
    type: 'bar',
    texttemplate: '%{value:,.2r}',
    textposition: 'outside',
    line: { color: 'lightblue', width: 3 },
    offset: 1,
  }];
  for (let i = 0; i <= params.epochs; i++) {
    lossData[0].x.push(i);
    lossData[0].y.push(0);
  }
  const lossLayout = {
    xaxis: { type: 'scatter', autorange: false, range: [0, params.epochs + 1], dtick: 1, visible: false },
    margin: { l: 0, r: 0, t: 40, b: 0, pad: 100 },
  };
  let ms = performance.now();

  // training callback on each epoch end
  let lastEpoch = 0;
  function callback(epoch, loss, msg) {
    if (msg) advice(ok(true), msg);
    if (!Number.isNaN(loss)) {
      lastEpoch = epoch;
      lossData[0].y[epoch] = loss;
      lossLayout.yaxis = { tickprefix: '', autorange: false, range: [0, 1.2 * Math.max(...lossData[0].y)], visible: false };
      lossLayout.title = epoch === params.epochs ? `Trained: ${ms.toLocaleString()} ms` : `Training: ${Math.trunc(100 * (epoch + 1) / params.epochs)}%`;
      Plotly.newPlot(document.getElementById('train'), lossData, { ...chart.layout, ...lossLayout }, { ...chart.options, displayModeBar: false });
    }
  }

  // train
  callback(0, 0);
  await model.train(inputs, outputs, params, callback);
  ms = performance.now() - ms;
  model.stats.loss = lossData[0].y[lastEpoch];
  advice(ok(model.stats.loss < params.targetLoss), `Training loss: ${model.stats.loss}`);
  callback(params.epochs, 0);
  advice(ok(model.stats.eval < params.evalError), `Model evaluation: ${model.stats.eval}% error`);
  // advice(ok(trained.model.stats.accuracy < params.evalError), `Model accuracy: ${trained.model.stats.accuracy}% error`);
  if (tfvis) {
    tfvis.show.modelSummary({ name: 'Model Summary', tab: 'Visor' }, model.model);
    for (const i in model.model.layers) {
      tfvis.show.layer({ name: `Layer: ${model.model.layers[i].name}`, tab: 'Visor' }, model.model.getLayer(undefined, i));
    }
    document.getElementsByClassName('visor')[0].style.visibility = params.visor ? 'visible' : 'hidden';
  }
  log('Engine', tf.engine().memory());
}

async function validateModel(input, title) {
  const ma = computeWindow(input, params.inputWindow, params.outputWindow);
  const inputs = ma.map((val) => val.inputSet);
  const outputs = ma.map((val) => val.outputSet);
  const sma = ma.map((val) => val.sma);
  // validate
  data.validation = [{
    x: data.input.time.slice(params.inputWindow), // .slice(params.inputWindow - params.outputWindow), // data.input.time.slice((params.inputWindow - params.outputWindow) / 2),
    y: [],
    type: 'lines',
    line: { color: 'lightcoral', shape: 'spline', width: 2, opacity: 0.2 },
  }];
  const smaData = [{
    name: `SMA: ${params.inputWindow}`,
    x: data.input.time.slice(params.inputWindow), // .slice(params.inputWindow / 2),
    y: sma,
    type: 'lines',
    line: { color: '#888888', opacity: 0.5, shape: 'spline' },
  }];
  let pt = 0;
  while (pt < inputs.length) {
    const predictions = await model.predict(inputs[pt]);
    if (!predictions || !predictions[0] || predictions[0] > (2 * model.stats.max) || predictions[0] < (0.5 * model.stats.min)) {
      advice(ok(false), `Model fit out of range: ${predictions[0]}`);
      pt = inputs.length;
    } else {
      if (predictions.length === 1) {
        data.validation[0].y[pt] = predictions[0];
      } else {
        for (let i = 0; i < predictions.length; i++) data.prediction[0].y[pt] = predictions[i];
      }
      pt += predictions.length;
    }
  }
  let smaDistance = 0;
  model.stats.distance = 0;
  for (pt = 0; pt < inputs.length; pt++) {
    model.stats.distance += ((data.validation[0].y[pt] - outputs[pt]) ** 2) || 0;
    smaDistance += ((smaData[0].y[pt] - outputs[pt]) ** 2) || 0;
  }
  model.stats.distance = Math.trunc(100 * 100 * Math.sqrt(model.stats.distance / inputs.length) / model.stats.max) / 100;
  smaDistance = Math.trunc(100 * 100 * Math.sqrt(smaDistance / inputs.length) / model.stats.max) / 100;
  data.validation[0].name = `${title}: ${model.stats.distance}%`;
  if ((model.stats.distance - smaDistance) < params.smaError) {
    Plotly.plot(document.getElementById('graph'), smaData, chart.layout, chart.options);
    Plotly.plot(document.getElementById('graph'), data.validation, chart.layout, chart.options);
  }
  advice(ok((model.stats.distance - smaDistance) < params.smaError), `Model fit RMS: ${model.stats.distance}% | SMA RMS: ${smaDistance}%`);
}

async function predictModel(input, title) {
  // get last known sequence
  const last = [];
  for (let i = 0; i < params.inputWindow; i++) {
    last.push(input[input.length - params.inputWindow + i]);
  }
  // validate
  data.prediction = [{
    x: [],
    y: [],
    name: title,
    type: 'lines',
    line: { color: 'lightyellow', shape: 'spline', width: 3, opacity: 0.5 },
  }];
  const step = (data.input.time[data.input.time.length - 1] - data.input.time[0]) / data.input.time.length;
  let pt = 0;
  let correction = 0;
  while (pt < params.predictWindow) {
    const predictions = await model.predict(last);
    if (!predictions || !predictions[0] || predictions[0] > (2 * model.stats.max) || predictions[0] < (0.5 * model.stats.min)) {
      if (!predictions) advice(ok(false), 'No predictions');
      else advice(ok(false), `Prediction out of range: ${predictions[0]}`);
      pt = params.predictWindow;
    } else {
      if (pt === 0) correction = predictions[0] - input[input.length - 1];
      for (let i = 0; i < predictions.length; i++) {
        data.prediction[0].x[pt] = data.input.time[data.input.time.length - 1] + (pt * step) + (i * step);
        data.prediction[0].y[pt] = predictions[i] - correction;
      }
      last.push(...predictions);
      last.splice(0, predictions.length);
      pt += predictions.length;
    }
  }
  Plotly.plot(document.getElementById('graph'), data.prediction, chart.layout, chart.options);
  const perc = Math.trunc(10000 * (correction / input[input.length - 1])) / 100;
  if (perc !== 0) advice(ok(Math.abs(perc) < 20), `Predict correction to SMA: ${perc}%`);
}

async function loadModel() {
  const fileEl = document.getElementById('load');

  async function handleFiles() {
    fileEl.removeEventListener('change', handleFiles);
    if (this.files.length !== 1) return;
    const content = await this.files[0].text();
    data = JSON.parse(content);
    log('Loaded file', this.files[0].name, data.timestamp, data.notes);

    if (data.stock) stock = data.stock;
    // eslint-disable-next-line no-use-before-define
    await createMenu();
    if (data.stats.params) params = data.stats.params;
    if (data.input) drawGraph();
    if (data.validation) Plotly.plot(document.getElementById('graph'), data.validation, chart.layout, chart.options);
    if (data.prediction) Plotly.plot(document.getElementById('graph'), data.prediction, chart.layout, chart.options);
    if (data.notes) document.getElementById('notes').value = data.notes;

    advice('');
    advice(ok(data.input.adjusted && data.input.adjusted.length > 0), `Loaded data: ${data.input.type}: ${data.input.exchange}/${data.input.symbol} [${data.input.range}/${data.input.granularity}]`);
    advice('Model created', data.timestamp);
    advice('Parameters', params);
    advice(ok(data.stats.loss < params.targetLoss), `Training loss: ${data.stats.loss}`);
    advice(ok(data.stats.eval < params.evalError), `Model evaluation: ${data.stats.eval}% error`);
  }

  fileEl.addEventListener('change', handleFiles, false);
  fileEl.click();
}

async function saveModel() {
  data.stats = model.stats;
  data.stock = stock;
  data.timestamp = new Date();
  data.notes = document.getElementById('notes').value;
  const dt = new Date();
  const fileName = `${stock.symbol.toUpperCase()}-${dt.getFullYear()}-${(dt.getMonth() + 1).toString().padStart(2, '0')}-${dt.getDate().toString().padStart(2, '0')}.json`;
  const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(data, null, 2));
  const div = document.createElement('a');
  div.setAttribute('href', dataStr);
  div.setAttribute('download', fileName);
  // document.body.appendChild(downloadAnchorNode); // required for firefox
  div.click();
  div.remove();
  log('Saved file', fileName);
}

async function initTFJS() {
  await wasm.setWasmPaths('../assets/');
  await tf.setBackend(params.backend);
  await tf.enableProdMode();
  if (tf.getBackend() === 'webgl') {
    // tf.ENV.set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
    // tf.ENV.set('WEBGL_FORCE_F16_TEXTURES', false);
    tf.ENV.set('WEBGL_PACK_DEPTHWISECONV', true);
    const gl = await tf.backend().getGPGPUContext().gl;
    log(`TFJS version: ${tf.version_core} backend: ${tf.getBackend().toUpperCase()} version: ${gl.getParameter(gl.VERSION)} renderer: ${gl.getParameter(gl.RENDERER)}`);
  } else if (tf.getBackend() === 'wasm') {
    log(`TFJS version: ${tf.version_core} backend: ${tf.getBackend().toUpperCase()} execution: ${tf.ENV.flags.WASM_HAS_SIMD_SUPPORT ? 'SIMD' : 'no SIMD'} ${tf.ENV.flags.WASM_HAS_MULTITHREAD_SUPPORT ? 'multithreaded' : 'singlethreaded'}`);
  } else {
    log(`TFJS version: ${tf.version_core} backend: ${tf.getBackend().toUpperCase()}`);
  }
  await tf.ready();
  if (tfvis) await tfvis.visor();
  if (params.visor) tfvis.visor().open();
  else tfvis.visor().close();
}

async function createMenu() {
  const div = document.getElementById('params');
  if (div.childNodes.length > 0) div.innerHTML = '';
  const box = div.getBoundingClientRect();

  const menu1 = new Menu(div, '', { top: `${box.top}px`, left: `${box.left}px` });
  menu1.addButton('Init Engine', 'Init Engine', () => initTFJS());
  menu1.addList('Backend', ['cpu', 'webgl', 'wasm'], params.backend, (val) => params.backend = val);
  menu1.addList('Dtype', ['int32', 'float32'], params.dtype, (val) => params.dtype = val);
  menu1.addHTML('<hr>');
  menu1.addButton('Get Data', 'Get Data', () => getData());
  const inputSymbol = menu1.addInput('Symbol', stock, 'symbol', (val) => stock.symbol = val);
  menu1.addList('Market', markets.map((val) => val.d), '', (val) => {
    stock.symbol = (markets.find((mkt) => val === mkt.d)).s;
    inputSymbol.value = stock.symbol;
    getData();
  });
  menu1.addList('Sector', sectors.map((val) => val.d), '', (val) => {
    stock.symbol = (sectors.find((mkt) => val === mkt.d)).s;
    inputSymbol.value = stock.symbol;
    getData();
  });
  menu1.addList('Interval', ['1m', '15m', '30m', '1h', '1d', '1wk', '1mo'], stock.interval, (val) => stock.interval = val);
  menu1.addList('Range', ['1d', '5d', '1mo', '3mo', '1y', '2y'], stock.range, (val) => stock.range = val);
  menu1.addHTML('<hr>');

  const menu2 = new Menu(div, '', { top: `${box.top}px`, left: `${box.left + 170}px` });
  menu2.addLabel('Model definition');
  menu2.addHTML('<hr>');
  menu2.addRange('Input window', params, 'inputWindow', 1, 100, 1, (val) => params.inputWindow = parseInt(val));
  menu2.addRange('Output window', params, 'outputWindow', 1, 100, 1, (val) => params.outputWindow = parseInt(val));
  menu2.addHTML('<hr>');
  menu2.addRange('Training epochs', params, 'epochs', 1, 50, 1, (val) => params.epochs = parseInt(val));
  menu2.addRange('Validation split', params, 'validationSplit', 0.0, 0.9, 0.05, (val) => params.validationSplit = parseFloat(val));
  menu2.addHTML('<hr>');
  menu2.addList('Optimizer', ['sgd', 'adagrad', 'adadelta', 'adam', 'adamax', 'rmsprop'], params.optimizer, (val) => params.optimizer = val);
  menu2.addRange('Learning rate', params, 'learningRate', 0.001, 1, 0.001, (val) => params.learningRate = parseFloat(val));
  menu2.addHTML('<hr>');
  menu2.addRange('Target loss', params, 'targetLoss', 0.01, 1, 0.01, (val) => params.targetLoss = parseFloat(val));
  menu2.addRange('Max eval error', params, 'evalError', 0.01, 10, 0.01, (val) => params.evalError = parseFloat(val));
  menu2.addRange('Discard threshold', params, 'smaError', 0.01, 10, 0.01, (val) => params.smaError = parseFloat(val));

  const menu3 = new Menu(div, '', { top: `${box.top}px`, left: `${box.left + 390}px` });
  menu3.addLabel('Model definition');
  menu3.addHTML('<hr>');
  menu3.addRange('Shape neurons', params, 'neurons', 1, 100, 1, (val) => params.neurons = parseInt(val));
  menu3.addRange('Shape features', params, 'features', 1, 100, 1, (val) => params.features = parseInt(val));
  menu3.addHTML('<hr>');
  menu3.addRange('Processing cells', params, 'layers', 1, 10, 1, (val) => params.layers = parseInt(val));
  menu3.addList('Cell type', ['lstmCell', 'gruCell'], params.cells, (val) => params.cells = val);
  menu3.addHTML('<hr>');
  menu3.addList('Kernel initializer', ['glorotNormal', 'heNormal', 'leCunNormal', 'ones', 'randomNormal', 'zeros'], params.kernelInitializer, (val) => params.kernelInitializer = val);
  menu3.addList('Initial activation', ['elu', 'hardSigmoid', 'linear', 'relu', 'relu6', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh'], params.activation, (val) => params.activation = val);
  menu3.addList('Recurrent activation', ['elu', 'hardSigmoid', 'linear', 'relu', 'relu6', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh'], params.recurrentActivation, (val) => params.recurrentActivation = val);
  menu3.addHTML('<hr>');
  menu3.addBool('Forget bias', params, 'forgetBias', (val) => params.forgetBias = val);
  menu3.addList('Bias initializer', ['glorotNormal', 'heNormal', 'leCunNormal', 'ones', 'randomNormal', 'zeros'], params.biasInitializer, (val) => params.biasInitializer = val);
  menu3.addBool('Shuffle data', params, 'shuffle', (val) => params.shuffle = val);

  const menu4 = new Menu(div, '', { top: `${box.top}px`, left: `${box.left + 650}px` });
  menu4.addButton('Load Model', 'Load Model', () => loadModel());
  menu4.addButton('Save Model', 'Save Model', () => saveModel());
  menu4.addHTML('<hr>');
  menu4.addButton('Run Training', 'Train Model', async () => {
    if (!data || !data.input.adjusted) return;
    await trainModel(data.input.adjusted);
    await validateModel(data.input.adjusted, 'Fit');
  });
  menu4.addBool('Show visor', params, 'visor', (val) => {
    params.visor = val;
    if (params.visor) tfvis.visor().open();
    else tfvis.visor().close();
    document.getElementsByClassName('visor')[0].style.visibility = params.visor ? 'visible' : 'hidden';
  });
  menu4.addButton('Run Prediction', 'Run Inference', async () => {
    if (!data || !data.input.adjusted) return;
    await predictModel(data.input.adjusted, 'Predict');
    // await predictModel(data.input.open, 'Predict: Open');
    // await predictModel(data.input.high, 'Predict: High');
    // await predictModel(data.input.low, 'Predict: Low');
    // await predictModel(data.input.close, 'Predict: Close');
  });
  menu4.addRange('Predict window', params, 'predictWindow', 1, 100, 1, (val) => params.predictWindow = parseInt(val));
  menu4.addHTML('<hr>');
  menu4.addLabel('Notes');
  menu4.addHTML('<textarea id="notes" style="width: 100%; height: 8rem; background: #444444; color: white;"></textarea>');

  document.getElementById('advice').addEventListener('click', () => {
    delete model.stats.epoch;
    delete model.stats.history;
    delete model.stats.acc;
    delete model.stats.data.prediction;
    navigator.clipboard.writeText(JSON.stringify({ ...stock, ...model.stats }, null, 2));
  });
}

async function main() {
  log('Initializing');
  await createMenu();
  await initTFJS();
  await getData();
  // await trainModel(data.input.adjusted);
  // await validateModel(data.input.adjusted, 'Fit:');
  // await predictModel(data.input.adjusted, 'Predict');
}

window.onload = main;
