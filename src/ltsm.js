import Plotly from 'plotly.js-dist';
import * as tf from '@tensorflow/tfjs';
import * as model from './model.js';

let data;

const stock = {
  symbol: 'dell',
  inverval: '1d', // validIntervals:[1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo]
  range: '2y', // validRanges:[1d,5d,1mo,3mo,6mo,1y,2y,5y]
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
      color: '#FFFFFF',
    },
    plot_bgcolor: '#222222',
    paper_bgcolor: '#000000',
    title: '', // `${data.meta.symbol} type:${data.meta.instrumentType} exchange:${data.meta.exchangeName} timezone:${data.meta.timezone} range:${data.meta.range} granularity:${data.meta.dataGranularity}`,
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

async function getData(tick, interval, range) {
  const symbol = encodeURI(tick);
  const url = `https://query2.finance.yahoo.com/v8/finance/chart/${symbol}?range=${range}&interval=${interval}&indicators=quote&includeTimestamps=true&includePrePost=true`;
  const res = await fetch('https://localhost:8000/cors', { headers: { cors: url } });
  const json = (res && res.ok) ? await res.json() : {};
  if (!json.chart || !json.chart.result) {
    log('data error:', tick);
    return {};
  }
  return json.chart.result[0];
}

function computeMA(raw, windowSize) {
  const avg = [];
  for (let i = 0; i < windowSize; i++) {
    avg.push(raw[i]);
  }
  for (let i = 0; i <= raw.length - windowSize; i++) {
    let currAvg = 0.00;
    const t = i + windowSize;
    for (let k = i; k < t && k <= raw.length; k++) currAvg += raw[k] / windowSize;
    avg.push(currAvg);
  }
  return avg;
}

async function drawGraph() {
  data = await getData(stock.symbol, stock.inverval, stock.range);
  if (!data || !data.indicators) return;
  chart.element = document.getElementById('graph');
  const maxPrice = Math.max(...data.indicators.adjclose[0].adjclose);
  const maxVolume = Math.max(...data.indicators.quote[0].volume);
  chart.data = [
    {
      x: data.timestamp.map((val) => val * 1000),
      y: data.indicators.adjclose[0].adjclose,
      name: 'Price',
      type: 'lines',
      line: {
        color: 'lightblue',
        shape: 'spline',
        width: 3,
      },
    },
    {
      x: data.timestamp.map((val) => val * 1000),
      y: computeMA(data.indicators.adjclose[0].adjclose, 5),
      name: 'MA: 5',
      type: 'lines',
      line: {
        color: '#AAAAAA',
        opacity: 0.5,
        shape: 'spline',
      },
    },
    {
      x: data.timestamp.map((val) => val * 1000),
      y: computeMA(data.indicators.adjclose[0].adjclose, 30),
      name: 'MA: 30',
      type: 'lines',
      line: {
        color: '#888888',
        opacity: 0.5,
        shape: 'spline',
      },
    },
    {
      x: data.timestamp.map((val) => val * 1000),
      open: data.indicators.quote[0].open,
      close: data.indicators.quote[0].close,
      high: data.indicators.quote[0].high,
      low: data.indicators.quote[0].low,
      name: 'OHLC',
      type: 'candlestick',
    },
    {
      x: data.timestamp.map((val) => val * 1000),
      y: data.indicators.quote[0].volume.map((val) => maxPrice * val / maxVolume / 5),
      name: 'Volume',
      type: 'bar',
      marker: {
        color: 'steelblue',
      },
      hoverformat: 'blaa',
    },
  ];
  chart.layout.yaxis.dtick = Math.trunc(chart.data[0].x.length / 25);
  chart.layout.title = `${data.meta.instrumentType}: ${data.meta.exchangeName}/${data.meta.symbol} [${data.meta.range}/${data.meta.dataGranularity}]`;
  Plotly.newPlot(chart.element, chart.data, chart.layout, chart.options);
}

async function main() {
  log('LSTM initializing');
  await tf.setBackend('webgl');
  await tf.enableProdMode();
  // tf.ENV.set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
  // tf.ENV.set('WEBGL_FORCE_F16_TEXTURES', true);
  // tf.ENV.set('WEBGL_PACK_DEPTHWISECONV', true);
  const gl = await tf.backend().getGPGPUContext().gl;
  log('TFJS version:', tf.version_core, 'backend:', tf.getBackend(), 'flags:', tf.ENV.flags);
  log(`GL version: ${gl.getParameter(gl.VERSION)} renderer: ${gl.getParameter(gl.RENDERER)}`);
  await tf.ready();
  await drawGraph();
}

window.onload = main;
