/* global tf, tfvis, Plotly */

// import Plotly from 'plotly.js-dist'; // <https://plotly.com/javascript/>
// import * as tf from '@tensorflow/tfjs'; // <https://js.tensorflow.org/api/latest/>
import * as wasm from '@tensorflow/tfjs-backend-wasm';
import * as model from './model.js';
import Menu from './menu.js';

let data;
let trained;

const stock = {
  symbol: 'dell',
  interval: '1d',
  range: '2y',
};

const params = {
  backend: 'webgl',
  dtype: 'float32',
  evalError: 2.5,
  smaError: 2.5,
  visor: false,

  inputWindow: 30,
  outputWindow: 1,
  predictWindow: 60,
  epochs: 25,
  validationSplit: 0.2,
  optimizer: 'adam',
  learningRate: 0.002,
  loss: 'meanSquaredError',
  targetLoss: 0.1,

  neurons: 40,
  features: 10,
  layers: 1,
  cells: 'lstmCell',
  kernelInitializer: 'leCunNormal',
  activation: 'relu',
  recurrentActivation: 'hardSigmoid',
  // constraint: 'unitNorm',

  forgetBias: false,
  biasInitializer: 'glorotNormal',
  shuffle: false,
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
  const div = document.getElementById('log');
  div.innerHTML += `<font color=gray>${ts}</font> &nbsp ${str(msg)}<br>`;
  div.scrollTop = div.scrollHeight;
}

function advice(...msg) {
  const div = document.getElementById('advice');
  div.innerHTML += `${str(msg)}<br>`;
  div.scrollTop = div.scrollHeight;
}

function computeSMA(input, inputWindow, outputWindow = 1) {
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
  if (!data) return;
  const maxPrice = Math.max(...data.adjusted);
  const maxVolume = Math.max(...data.volume);
  chart.data = [];
  chart.data.push({
    name: 'Price',
    x: data.time,
    y: data.adjusted,
    type: 'lines',
    line: { color: 'lightblue', shape: 'spline', width: 3 },
  });
  chart.data.push({
    name: 'OHLC',
    x: data.time,
    open: data.open,
    close: data.close,
    high: data.high,
    low: data.low,
    type: 'candlestick',
  });
  chart.data.push({
    name: 'Volume',
    x: data.time,
    y: data.volume.map((val) => maxPrice * val / maxVolume / 2),
    type: 'bar',
    marker: { color: 'steelblue' },
  });
  // chart.layout.xaxis.dtick = Math.trunc(data.time[0] / 1000);
  // chart.layout.yaxis.dtick = Math.trunc(Math.max(...data.adjusted) / 10);
  chart.layout.title = `${data.type}: ${data.exchange}/${data.symbol} [${data.range}/${data.granularity}]`;
  Plotly.newPlot(document.getElementById('graph'), chart.data, chart.layout, chart.options);
}

async function getData() {
  const url = `https://query2.finance.yahoo.com/v8/finance/chart/${encodeURI(stock.symbol)}?range=${stock.range}&interval=${stock.interval}&indicators=quote&includeTimestamps=true&includePrePost=true`;
  const res = await fetch('https://localhost:8000/cors', { headers: { cors: url } });
  const json = (res && res.ok) ? await res.json() : {};
  if (!json.chart || !json.chart.result) {
    log('data error:', stock.symbol);
    return;
  }
  data = {
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
  advice(ok(data.adjusted && data.adjusted.length > 0), `Data: ${data.type}: ${data.exchange}/${data.symbol} [${data.range}/${data.granularity}]`);
  await drawGraph();
  advice(ok(data.adjusted.length > 250), `Data set size: ${data.adjusted.length}`);
}

async function trainModel(input) {
  if (!input) return;
  advice('');
  advice('Training', params);

  const ma = computeSMA(input, params.inputWindow, params.outputWindow);
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

  // dispose previous model (still leaking 8 tensors)
  if (trained && trained.model && trained.model.optimizer) trained.model.optimizer.dispose();
  if (trained && trained.model) trained.model.dispose();
  // train
  callback(0, 0);
  trained = await model.train(inputs, outputs, params, callback);
  ms = performance.now() - ms;
  // callback(params.epochs, 0);
  /*
  const layers = [];
  for (const layer of trained.model.layers) {
    // eslint-disable-next-line no-console
    console.log('Layer:', layer);
    if (layer.cell) layers.push({ cell: layer.cell.cells.map((val) => val.name) });
    layers.push({ name: layer.name, shape: layer.outputShape });
  }
  log('Model', layers);
  console.log('Model summary:', trained.model.summary());
  */
  // eslint-disable-next-line no-console
  console.log('Model: ', trained.model);
  advice(ok(lossData[0].y[lastEpoch] < params.targetLoss), `Training loss: ${lossData[0].y[lastEpoch]}`);
  advice(ok(trained.stats.eval < params.evalError), `Model evaluation: ${trained.stats.eval}% error`);
  advice(ok(trained.stats.accuracy < params.evalError), `Model accuracy: ${trained.stats.accuracy}% error`);
  if (tfvis) {
    tfvis.show.modelSummary({ name: 'Model Summary', tab: 'Visor' }, trained.model);
    for (const i in trained.model.layers) {
      tfvis.show.layer({ name: `Layer: ${trained.model.layers[i].name}`, tab: 'Visor' }, trained.model.getLayer(undefined, i));
    }
    document.getElementsByClassName('visor')[0].style.visibility = params.visor ? 'visible' : 'hidden';
  }
  log('Engine', tf.engine().memory());
}

async function validateModel(input, title) {
  if (!trained || !trained.model) return;
  const ma = computeSMA(input, params.inputWindow, params.outputWindow);
  const inputs = ma.map((val) => val.inputSet);
  const outputs = ma.map((val) => val.outputSet);
  const sma = ma.map((val) => val.sma);
  // validate
  const validationData = [{
    x: data.time.slice(params.inputWindow), // .slice(params.inputWindow - params.outputWindow), // data.time.slice((params.inputWindow - params.outputWindow) / 2),
    y: [],
    type: 'lines',
    line: { color: 'lightcoral', shape: 'spline', width: 2, opacity: 0.2 },
  }];
  const smaData = [{
    name: `SMA: ${params.inputWindow}`,
    x: data.time.slice(params.inputWindow), // .slice(params.inputWindow / 2),
    y: sma,
    type: 'lines',
    line: { color: '#888888', opacity: 0.5, shape: 'spline' },
  }];
  let pt = 0;
  while (pt < inputs.length) {
    const predictions = await model.predict(trained, inputs[pt]);
    if (!predictions || !predictions[0] || predictions[0] > (2 * trained.stats.max) || predictions[0] < (0.5 * trained.stats.min)) {
      advice(ok(false), `Model fit out of range: ${predictions[0]}`);
      pt = inputs.length;
    } else {
      if (predictions.length === 1) {
        validationData[0].y[pt] = predictions[0];
      } else {
        for (let i = 0; i < predictions.length; i++) validationData[0].y[pt] = predictions[i];
      }
      pt += predictions.length;
    }
  }
  let modelDistance = 0;
  let smaDistance = 0;
  for (pt = 0; pt < inputs.length; pt++) {
    modelDistance += ((validationData[0].y[pt] - outputs[pt]) ** 2) || 0;
    smaDistance += ((smaData[0].y[pt] - outputs[pt]) ** 2) || 0;
  }
  modelDistance = Math.trunc(100 * 100 * Math.sqrt(modelDistance / inputs.length) / trained.stats.max) / 100;
  smaDistance = Math.trunc(100 * 100 * Math.sqrt(smaDistance / inputs.length) / trained.stats.max) / 100;
  validationData[0].name = `${title}: ${modelDistance}%`;
  if ((modelDistance - smaDistance) < params.smaError) {
    Plotly.plot(document.getElementById('graph'), smaData, chart.layout, chart.options);
    Plotly.plot(document.getElementById('graph'), validationData, chart.layout, chart.options);
  }
  advice(ok((modelDistance - smaDistance) < params.smaError), `Model fit RMS: ${modelDistance}% | SMA RMS: ${smaDistance}%`);
}

async function predictModel(input, title) {
  if (!trained || !trained.model) return;
  // get last known sequence
  const last = [];
  for (let i = 0; i < params.inputWindow; i++) {
    last.push(input[input.length - params.inputWindow + i]);
  }
  // validate
  const predictionData = [{
    x: [],
    y: [],
    name: title,
    type: 'lines',
    line: { color: 'lightyellow', shape: 'spline', width: 3, opacity: 0.5 },
  }];
  const step = (data.time[data.time.length - 1] - data.time[0]) / data.time.length;
  let pt = 0;
  let correction = 0;
  while (pt < params.predictWindow) {
    const predictions = await model.predict(trained, last);
    if (!predictions || !predictions[0] || predictions[0] > (2 * trained.stats.max) || predictions[0] < (0.5 * trained.stats.min)) {
      advice(ok(false), `Prediction out of range: ${predictions[0]}`);
      pt = params.predictWindow;
    } else {
      if (pt === 0) correction = predictions[0] - input[input.length - 1];
      for (let i = 0; i < predictions.length; i++) {
        predictionData[0].x[pt] = data.time[data.time.length - 1] + (pt * step) + (i * step);
        predictionData[0].y[pt] = predictions[i] - correction;
      }
      last.push(...predictions);
      last.splice(0, predictions.length);
      pt += predictions.length;
    }
  }
  Plotly.plot(document.getElementById('graph'), predictionData, chart.layout, chart.options);
  const perc = Math.trunc(10000 * (correction / input[input.length - 1])) / 100;
  advice(ok(Math.abs(perc) < 20), `Predict correction to SMA: ${perc}%`);
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
  menu1.addButton('Run Inference', 'Run Inference', async () => {
    if (!data || !data.adjusted) return;
    await predictModel(data.adjusted, 'Predict');
    // await predictModel(data.open, 'Predict: Open');
    // await predictModel(data.high, 'Predict: High');
    // await predictModel(data.low, 'Predict: Low');
    // await predictModel(data.close, 'Predict: Close');
  });
  menu1.addRange('Predict window', params, 'predictWindow', 1, 100, 1, (val) => params.predictWindow = parseInt(val));

  const menu2 = new Menu(div, '', { top: `${box.top}px`, left: `${box.left + 210}px` });
  menu2.addButton('Train Model', 'Train Model', async () => {
    if (!data || !data.adjusted) return;
    await trainModel(data.adjusted);
    await validateModel(data.adjusted, 'Fit');
  });
  menu2.addRange('Input window', params, 'inputWindow', 1, 100, 1, (val) => params.inputWindow = parseInt(val));
  menu2.addRange('Output window', params, 'outputWindow', 1, 100, 1, (val) => params.outputWindow = parseInt(val));
  menu2.addHTML('<hr>');
  menu2.addRange('Training epochs', params, 'epochs', 1, 50, 1, (val) => params.epochs = parseInt(val));
  menu2.addRange('Validation split', params, 'validationSplit', 0.1, 0.9, 0.1, (val) => params.validationSplit = parseFloat(val));
  menu2.addHTML('<hr>');
  menu2.addList('Optimizer', ['sgd', 'adagrad', 'adadelta', 'adam', 'adamax', 'rmsprop'], params.optimizer, (val) => params.optimizer = val);
  menu2.addRange('Learning rate', params, 'learningRate', 0.001, 1, 0.001, (val) => params.learningRate = parseFloat(val));
  menu2.addHTML('<hr>');
  menu2.addRange('Target loss', params, 'targetLoss', 0.01, 1, 0.1, (val) => params.targetLoss = parseFloat(val));
  menu2.addRange('Max eval error', params, 'evalError', 0.1, 10, 0.1, (val) => params.evalError = parseFloat(val));
  menu2.addRange('Discard threshold', params, 'smaError', 0.1, 10, 0.1, (val) => params.smaError = parseFloat(val));
  menu2.addHTML('<hr>');
  menu2.addBool('Show visor', params, 'visor', (val) => {
    params.visor = val;
    if (params.visor) tfvis.visor().open();
    else tfvis.visor().close();
    document.getElementsByClassName('visor')[0].style.visibility = params.visor ? 'visible' : 'hidden';
  });

  const menu3 = new Menu(div, '', { top: `${box.top}px`, left: `${box.left + 430}px` });
  menu3.addLabel('Model definition');
  menu3.addHTML('<hr>');
  menu3.addRange('Shape neurons', params, 'neurons', 1, 100, 1, (val) => params.neurons = parseInt(val));
  menu3.addRange('Shape features', params, 'features', 1, 100, 1, (val) => params.features = parseInt(val));
  menu3.addHTML('<hr>');
  menu3.addRange('Processing cells', params, 'layers', 1, 10, 1, (val) => params.layers = parseInt(val));
  menu3.addList('Cell type', ['lstmCell', 'gruCell'], params.cells, (val) => params.cells = val);
  menu3.addHTML('<hr>');
  menu3.addList('Kernel initializer', ['glorotNormal', 'heNormal', 'leCunNormal', 'ones', 'randomNormal', 'zeros'], params.kernelInitializer, (val) => params.kernelInitializer = val);
  menu3.addList('Activation', ['elu', 'hardSigmoid', 'linear', 'relu', 'relu6', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh'], params.activation, (val) => params.activation = val);
  menu3.addList('Recurrent activation', ['elu', 'hardSigmoid', 'linear', 'relu', 'relu6', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh'], params.recurrentActivation, (val) => params.recurrentActivation = val);
  menu3.addHTML('<hr>');
  menu3.addBool('Forget bias', params, 'forgetBias', (val) => params.forgetBias = val);
  menu3.addList('Bias initializer', ['glorotNormal', 'heNormal', 'leCunNormal', 'ones', 'randomNormal', 'zeros'], params.biasInitializer, (val) => params.biasInitializer = val);
  menu3.addBool('Shuffle data', params, 'shuffle', (val) => params.shuffle = val);
}

async function main() {
  log('Initializing');
  await createMenu();
  await initTFJS();
  await getData();
  // await trainModel(data.adjusted);
  // await validateModel(data.adjusted, 'Fit:');
  // await predictModel(data.adjusted, 'Predict');
}

window.onload = main;
