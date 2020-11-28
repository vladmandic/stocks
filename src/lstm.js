import Plotly from 'plotly.js-dist';
import * as tf from '@tensorflow/tfjs';
import * as wasm from '@tensorflow/tfjs-backend-wasm';
import * as model from './model.js';
import Menu from './menu.js';

let data;
let trained;

const stock = {
  symbol: 'dell',
  interval: '1d', // validIntervals:[1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo]
  range: '2y', // validRanges:[1d,5d,1mo,3mo,6mo,1y,2y,5y]
};

const params = {
  epochs: 10,
  learningRate: 0.025,
  layers: 3,
  inputWindow: 15,
  outputWindow: 1,
  predictWindow: 15,
  neurons: 30,
  features: 5,
  dtype: 'float32',
  backend: 'webgl',
};

// eslint-disable-next-line no-unused-vars
const markets = [
  { d: '', s: '', f: '' },
  { d: 'S&P 500', s: '^GSPC', f: 'ES=F' },
  { d: 'Dow Jones', s: '^DJI', f: 'YM=F' },
  { d: 'NASDAQ', s: '^IXIC', f: 'NQ=F' },
];

// eslint-disable-next-line no-unused-vars
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
      autotick: false,
      dtick: 15 * 1000 * 60 * 60 * 24,
      showticklabels: true,
      gridcolor: '#555555',
    },
    yaxis: {
      autorange: true,
      rangemode: 'tozero',
      showgrid: true,
      zeroline: true,
      showline: true,
      autotick: false,
      dtick: 10,
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
    if (typeof entry === 'object') line += JSON.stringify(entry).replace(/{|}|"|\[|\]/g, '').replace(/,/g, ' ');
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
  div.innerHTML += `${ts} &nbsp ${str(msg)}<br>`;
  div.scrollTop = div.scrollHeight;
}

function advice(...msg) {
  const div = document.getElementById('advice');
  div.innerHTML += `${str(msg)}<br>`;
  div.scrollTop = div.scrollHeight;
}

function computeMA(raw, windowSize) {
  const avg = [];
  for (let i = 0; i <= raw.length - windowSize; i++) {
    let ma = 0.00;
    const t = i + windowSize;
    for (let k = i; k < t && k <= raw.length; k++) {
      ma += raw[k] / windowSize;
    }
    const set = raw.slice(i, i + windowSize);
    avg.push({ set, ma });
  }
  return avg;
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
    name: 'MA: 30',
    x: data.time.slice(30 / 2).concat(data.time),
    y: computeMA(data.adjusted, 30).map((val) => val.ma),
    type: 'lines',
    line: { color: '#888888', opacity: 0.5, shape: 'spline' },
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
    y: data.volume.map((val) => maxPrice * val / maxVolume / 5),
    type: 'bar',
    marker: { color: 'steelblue' },
  });
  chart.layout.xaxis.dtick = Math.trunc(data.time[0] / 1000);
  chart.layout.yaxis.dtick = Math.trunc(Math.max(...data.adjusted) / 10);
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
  // log('Train:', params);
  const inputs = computeMA(input, params.inputWindow).map((val) => val['set']);
  const outputs = computeMA(input, params.outputWindow).map((val) => val['set']).slice(params.inputWindow - params.outputWindow, data.adjusted.length);

  // train graph
  const lossData = [{
    x: [],
    y: [],
    name: 'Epoch',
    type: 'bar',
    texttemplate: '%{value:,.2r}',
    textposition: 'inside',
    line: { color: 'lightblue', width: 3 },
  }];
  for (let i = 0; i <= params.epochs; i++) {
    lossData[0].x.push(i);
    lossData[0].y.push(0);
  }

  // training callback on each epoch end
  let ms = performance.now();
  const maxPrice = Math.max(...data.adjusted);
  async function callback(epoch, logs) {
    lossData[0].y[epoch + 1] = params.dtype === 'int32' ? (maxPrice * logs.loss / (255 ** 2)) : (maxPrice * logs.loss);
    const title = epoch === params.epochs ? `Training complete: ${ms.toLocaleString()} ms Loss: ${Math.trunc(1000 * lossData[0].y[epoch]) / 1000}` : `Training: ${Math.trunc(100 * (epoch + 1) / params.epochs)}%`;
    const lossLayout = {
      xaxis: { type: 'scatter', autorange: false, range: [0, params.epochs + 1], dtick: 1, visible: false },
      yaxis: { tickprefix: '', autorange: false, range: [0, Math.max(...lossData[0].y)], visible: false },
      title,
    };
    Plotly.newPlot(document.getElementById('train'), lossData, { ...chart.layout, ...lossLayout }, chart.options);
  }

  // init loss graph
  await callback(-1, { loss: 0 });
  // train
  trained = await model.train(inputs, outputs, params, callback);
  ms = performance.now() - ms;
  await callback(params.epochs, { loss: 0 });
  advice(ok(lossData[0].y[params.epochs] < (params.dtype === 'int32' ? 10 : 0.1)), `Final loss: ${Math.trunc(1000 * lossData[0].y[params.epochs]) / 1000}`);
}

async function validateModel(input, title) {
  if (!trained || !trained.model) return;
  const inputs = computeMA(data.adjusted, params.inputWindow).map((val) => val['set']);
  const outputs = computeMA(data.adjusted, params.outputWindow).map((val) => val['ma']);
  // validate
  const validationData = [{
    x: [],
    y: [],
    name: title,
    type: 'lines',
    line: { color: 'lightcoral', shape: 'spline', width: 2, opacity: 0.5 },
  }];
  let pt = 0;
  while (pt < inputs.length) {
    const predictions = await model.predict([inputs[pt]], trained);
    for (let i = 0; i < predictions.length; i++) {
      validationData[0].x[pt] = data.time[pt + params.inputWindow - params.outputWindow + i];
      validationData[0].y[pt] = predictions[i];
    }
    pt += predictions.length;
  }
  let distance = 0;
  for (pt = 0; pt < inputs.length; pt++) {
    distance += (validationData[0].y[pt] - outputs[pt]) ** 2;
  }
  distance = Math.trunc(Math.sqrt(distance) / inputs.length * 1000) / 1000;
  if (distance < 1) Plotly.plot(document.getElementById('graph'), validationData, chart.layout, chart.options);
  advice(ok(distance < 1), `Model fit distance: ${distance}`);
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
    const predictions = await model.predict([last], trained);
    if (pt === 0) correction = predictions[0] - input[input.length - 1];
    for (let i = 0; i < predictions.length; i++) {
      predictionData[0].x[pt] = data.time[data.time.length - 1] + (pt * step) + (i * step);
      predictionData[0].y[pt] = predictions[i] - correction;
    }
    last.push(...predictions);
    last.splice(0, predictions.length);
    pt += predictions.length;
  }
  Plotly.plot(document.getElementById('graph'), predictionData, chart.layout, chart.options);
  const perc = Math.trunc(100 * Math.abs(correction / input[input.length - 1])) / 100;
  advice(ok(perc < 0.1), `Correction to SMA: ${perc}`);
}

async function initTFJS() {
  wasm.setWasmPaths('../assets/');
  // await tf.setBackend('webgl');
  await tf.setBackend(params.backend);
  await tf.enableProdMode();
  if (tf.getBackend() === 'webgl') {
    // tf.ENV.set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
    // tf.ENV.set('WEBGL_FORCE_F16_TEXTURES', true);
    // tf.ENV.set('WEBGL_PACK_DEPTHWISECONV', true);
    const gl = await tf.backend().getGPGPUContext().gl;
    log(`TFJS version: ${tf.version_core} backend: ${tf.getBackend().toUpperCase()} version: ${gl.getParameter(gl.VERSION)} renderer: ${gl.getParameter(gl.RENDERER)}`);
  } else if (tf.getBackend() === 'wasm') {
    log(`TFJS version: ${tf.version_core} backend: ${tf.getBackend().toUpperCase()} execution: ${tf.ENV.flags.WASM_HAS_SIMD_SUPPORT ? 'SIMD' : 'no SIMD'} ${tf.ENV.flags.WASM_HAS_MULTITHREAD_SUPPORT ? 'multithreaded' : 'singlethreaded'}`);
  } else {
    log(`TFJS version: ${tf.version_core} backend: ${tf.getBackend().toUpperCase()}`);
  }
  await tf.ready();
}

async function createMenu() {
  const div = document.getElementById('params');
  const box = div.getBoundingClientRect();

  const menu1 = new Menu(div, '', { top: `${box.top}px`, left: `${box.left}px` });
  menu1.addButton('Init Engine', 'Init Engine', () => initTFJS());
  menu1.addList('Backend', ['cpu', 'webgl', 'wasm'], params.backend, (val) => params.backend = val);
  menu1.addList('Dtype', ['int32', 'float32'], params.dtype, (val) => params.dtype = val);
  menu1.addButton('Get Data', 'Get Data', () => getData());
  menu1.addList('Market', markets.map((val) => val.d), '', (val) => stock.symbol = (markets.find((mkt) => val === mkt.d)).s);
  menu1.addList('Sector', sectors.map((val) => val.d), '', (val) => stock.symbol = (sectors.find((mkt) => val === mkt.d)).s);
  menu1.addInput('Symbol', stock, 'symbol', (val) => stock.symbol = val);
  menu1.addList('Interval', ['1m', '15m', '30m', '1h', '1d', '1wk', '1mo'], stock.interval, (val) => stock.interval = val);
  menu1.addList('Range', ['1d', '5d', '1mo', '3mo', '1y', '2y'], stock.range, (val) => stock.range = val);

  const menu2 = new Menu(div, '', { top: `${box.top}px`, left: `${box.left + 200}px` });
  menu2.addRange('Epochs', params, 'epochs', 1, 50, 1, (val) => params.epochs = parseInt(val));
  menu2.addRange('Layers', params, 'layers', 1, 10, 1, (val) => params.layers = parseInt(val));
  menu2.addRange('Input window', params, 'inputWindow', 1, 100, 1, (val) => params.inputWindow = parseInt(val));
  menu2.addRange('Output window', params, 'outputWindow', 1, 100, 1, (val) => params.outputWindow = parseInt(val));
  menu2.addRange('Predict window', params, 'predictWindow', 1, 100, 1, (val) => params.predictWindow = parseInt(val));
  menu2.addRange('Neurons', params, 'neurons', 1, 100, 1, (val) => params.neurons = parseInt(val));
  menu2.addRange('Features', params, 'features', 1, 100, 1, (val) => params.features = parseInt(val));
  menu2.addRange('Learning rate', params, 'learningRate', 0.01, 1, 0.01, (val) => params.learningRate = parseFloat(val));
  menu2.addButton('Train Model', 'Train Model', async () => {
    if (!data || !data.adjusted) return;
    await trainModel(data.adjusted);
    await validateModel(data.adjusted, 'Model fit');
  });
  menu2.addButton('Run Inference', 'Run Inference', async () => {
    if (!data || !data.adjusted) return;
    await predictModel(data.adjusted, 'Predict');
    // await predictModel(data.open, 'Predict: Open');
    // await predictModel(data.high, 'Predict: High');
    // await predictModel(data.low, 'Predict: Low');
    // await predictModel(data.close, 'Predict: Close');
  });
}

async function main() {
  log('LSTM initializing');
  await createMenu();
  await initTFJS();

  await getData();
  // await trainModel(data.adjusted);
  // await validateModel(data.adjusted, 'Fit:');
  // await predictModel(data.adjusted, 'Predict');
}

window.onload = main;
