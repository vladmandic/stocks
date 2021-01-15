#!/usr/bin/env -S node --trace-warnings

const fs = require('fs');
const esbuild = require('esbuild');
const log = require('@vladmandic/pilogger');

// keeps esbuild service instance cached
let es;
const banner = `
  /*
  homepage: <https://github.com/vladmandic/lstm>
  author: <https://github.com/vladmandic>'
  */
`;

// common configuration
const common = {
  banner,
  minifyWhitespace: true, // this one is ok
  minifyIdentifiers: false, // this requires tfjs 2.8.1 to function
  minifySyntax: false, // this breaks with tfjs 2.8.4
  bundle: true,
  sourcemap: true,
  logLevel: 'error',
  target: 'es2018',
};

const targets = {
  browserBundle: {
    index: {
      platform: 'browser',
      format: 'esm',
      metafile: 'dist/index.json',
      entryPoints: ['src/index.js'],
      outfile: 'dist/index.js',
      external: ['fs', 'buffer', 'util'],
    },
  },
};

async function getStats(metafile) {
  const stats = {};
  if (!fs.existsSync(metafile)) return stats;
  const data = fs.readFileSync(metafile);
  const json = JSON.parse(data.toString());
  if (json && json.inputs && json.outputs) {
    for (const [key, val] of Object.entries(json.inputs)) {
      if (key.startsWith('node_modules')) {
        stats.modules = (stats.modules || 0) + 1;
        stats.moduleBytes = (stats.moduleBytes || 0) + val.bytes;
      } else {
        stats.imports = (stats.imports || 0) + 1;
        stats.importBytes = (stats.importBytes || 0) + val.bytes;
      }
    }
    const files = [];
    for (const [key, val] of Object.entries(json.outputs)) {
      if (!key.endsWith('.map')) {
        files.push(key);
        stats.outputBytes = (stats.outputBytes || 0) + val.bytes;
      }
    }
    stats.outputFiles = files.join(', ');
  }
  return stats;
}

// rebuild on file change
async function build(f, msg) {
  log.info('Build: file', msg, f, 'target:', common.target);
  if (!es) es = await esbuild.startService();
  // common build options
  try {
    // rebuild all target groups and types
    for (const [targetGroupName, targetGroup] of Object.entries(targets)) {
      for (const [targetName, targetOptions] of Object.entries(targetGroup)) {
        // if triggered from watch mode, rebuild only browser bundle
        if ((require.main !== module) && (targetGroupName !== 'browserBundle')) continue;
        await es.build({ ...common, ...targetOptions });
        const stats = await getStats(targetOptions.metafile);
        log.state(`Build for: ${targetGroupName} type: ${targetName}:`, stats);
      }
    }
    if (require.main === module) process.exit(0);
  } catch (err) {
    // catch errors and print where it occured
    log.error('Build error', JSON.stringify(err.errors || err, null, 2));
    if (require.main === module) process.exit(1);
  }
}

if (require.main === module) {
  log.header();
  build('all', 'startup');
} else {
  exports.build = build;
}
