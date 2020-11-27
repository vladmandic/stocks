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
  windowSize: 15,
  neurons: 30,
  features: 5,
  loss: 'meanSquaredError',
  dtype: 'float',
  backend: 'webgl',
};

// eslint-disable-next-line no-unused-vars
const markets = [
  { d: 'S&P 500', s: '^GSPC', f: 'ES=F' },
  { d: 'Dow Jones', s: '^DJI', f: 'YM=F' },
  { d: 'NASDAQ', s: '^IXIC', f: 'NQ=F' },
];

// eslint-disable-next-line no-unused-vars
const sectors = [
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

function log(...msg) {
  const dt = new Date();
  const ts = `${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}:${dt.getSeconds().toString().padStart(2, '0')}.${dt.getMilliseconds().toString().padStart(3, '0')}`;
  // eslint-disable-next-line no-console
  console.log(ts, ...msg);
  document.getElementById('log').innerHTML += `${ts} &nbsp ${str(msg)}<br>`;
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
  // eslint-disable-next-line no-console
  console.log('Data:', chart.data);
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
  console.log(json.chart.result[0].indicators);
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
  log(`Data: ${data.type}: ${data.exchange}/${data.symbol} [${data.range}/${data.granularity}]`);
  await drawGraph();
}

async function trainModel() {
  if (!data || !data.adjusted) return;
  log('Train:', params);
  const ma = computeMA(data.adjusted, params.windowSize);
  window.ma = ma;
  const inputs = ma.map((input) => input['set']);
  const outputs = ma.map((output) => output['ma']);

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
  async function callback(epoch, logs) {
    lossData[0].y[epoch + 1] = logs.loss / 255;
    const title = epoch === params.epochs ? `Training complete: ${ms.toLocaleString()} ms` : `Training: ${Math.trunc(100 * (epoch + 1) / params.epochs)}%`;
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
}

async function predictModel(input, title) {
  if (!trained || !trained.model) return;
  const ma = computeMA(input, params.windowSize);
  const inputs = ma.map((val) => val['set']);
  // validate
  const validationData = [{
    x: [],
    y: [],
    name: title,
    type: 'lines',
    line: { color: 'lightcoral', shape: 'spline', width: 3 },
  }];
  for (let pt = 0; pt < inputs.length; pt++) {
    const prediction = await model.predict([inputs[pt]], trained);
    validationData[0].x[pt] = data.time[pt + Math.trunc(params.windowSize / 2)];
    validationData[0].y[pt] = prediction;
    if (pt < (inputs.length - 1)) {
      // inputs[pt + 1] = inputs[pt].concat(prediction).slice(1);
    }
  }
  Plotly.plot(document.getElementById('graph'), validationData, chart.layout, chart.options);
}

async function initTFJS() {
  wasm.setWasmPaths('../assets/');
  // await tf.setBackend('webgl');
  await tf.setBackend('wasm');
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
  menu1.addList('Dtype', ['int', 'float'], params.dtype, (val) => params.dtype = val);
  menu1.addButton('Get Data', 'Get Data', () => getData());
  menu1.addInput('Symbol', stock, 'symbol', (val) => stock.symbol = val);
  menu1.addList('Interval', ['1m', '15m', '30m', '1h', '1d', '1wk', '1mo'], stock.interval, (val) => stock.interval = val);
  menu1.addList('Range', ['1d', '5d', '1mo', '3mo', '1y', '2y'], stock.range, (val) => stock.range = val);
  menu1.addButton('Train Model', 'Train Model', () => trainModel());
  menu1.addRange('Epochs', params, 'epochs', 1, 50, 1, (val) => params.epochs = parseInt(val));
  menu1.addRange('Layers', params, 'layers', 1, 10, 1, (val) => params.layers = parseInt(val));
  menu1.addRange('Time window', params, 'windowSize', 1, 100, 1, (val) => params.windowSize = parseInt(val));
  menu1.addRange('Neurons', params, 'neurons', 1, 100, 1, (val) => params.neurons = parseInt(val));
  menu1.addRange('Features', params, 'features', 1, 100, 1, (val) => params.features = parseInt(val));
  menu1.addRange('Learning rate', params, 'learningRate', 0.01, 1, 0.01, (val) => params.learningRate = parseFloat(val));
  menu1.addButton('Run Inference', 'Run Inference', async () => {
    if (!data || !data.adjusted) return;
    await predictModel(data.adjusted, 'Predict: Adjusted');
    await predictModel(data.open, 'Predict: Open');
    await predictModel(data.high, 'Predict: High');
    await predictModel(data.low, 'Predict: Low');
    await predictModel(data.close, 'Predict: Close');
  });
}

async function main() {
  log('LSTM initializing');
  await createMenu();
  await initTFJS();
  /*
  await getData();
  await drawGraph();
  await trainModel();
  await predictModel(data.adjusted, 'Predict: Adjusted');
  await predictModel(data.open, 'Predict: Open');
  await predictModel(data.high, 'Predict: High');
  await predictModel(data.low, 'Predict: Low');
  await predictModel(data.close, 'Predict: Close');
  */
}

window.onload = main;
