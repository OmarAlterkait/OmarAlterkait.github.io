/* ---------------------------------------------------------------------------
   Interaction drift background ("the screen is a detector").

   A real simulated neutrino interaction (JAXTPC / doraemon production) is
   baked offline into a tilted 2D projection plus each deposit's TRUE drift
   distance to the anode. At runtime the cloud spawns at the click point,
   drifts down at constant speed, and each point dies when it has traveled
   its own drift distance: the fade front sweeps through the projected shape
   obliquely, so depth is felt without any 3D math ("cheating 3D").
   Landed charge accumulates as a glow histogram on the sensor strip at the
   bottom of the screen, then decays.

   Data: assets/bg/bg_event.bin, 8 bytes/point:
     int16 px, int16 py (projection, +-32767 = +-1), uint16 tnorm
     (drift distance / max), uint8 slot (track palette), uint8 de (log dE).

   Efficiency: one static VBO, all animation in the vertex shader, at most
   MAX_LIVE draw calls per frame. The rAF loop runs ONLY while charge is in
   flight or the strip glow is decaying; idle cost is zero.
   --------------------------------------------------------------------------- */
(function () {
  'use strict';

  var canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var gl = canvas.getContext('webgl', {
    alpha: true, antialias: false, depth: false, stencil: false,
    powerPreference: 'low-power',
  });
  if (!gl) return; // no WebGL: page simply has a plain background

  // ---- Tunables -----------------------------------------------------------
  var CLOUD_HALF_W = 0.33;   // cloud half-width as fraction of min(vw,vh)
  var SPEED = 0.16;          // drift speed, fraction of viewport height / s
  var FADE_FRAC = 0.07;      // fade duration as fraction of full drift time
  var GLOBAL_ALPHA = 0.55;   // overall subtlety dial (0..1)
  var POINT_SIZE = 2.2;      // base point size in px at dpr=1
  var MAX_LIVE = 3;          // concurrent interactions
  var STRIP_BINS = 512;
  var STRIP_HALFLIFE = 1.4;  // s, glow decay
  var STRIP_H_PX = 46;       // glow height above the bottom edge
  var PALETTE = [            // track slots 0..7 + "other" (muted)
    [0.43, 0.66, 0.86], [0.79, 0.64, 0.36], [0.62, 0.71, 0.66],
    [0.73, 0.63, 0.85], [0.85, 0.54, 0.54], [0.54, 0.77, 0.85],
    [0.77, 0.85, 0.54], [0.85, 0.71, 0.54], [0.42, 0.47, 0.53],
  ];

  // ---- Shaders -------------------------------------------------------------
  var VS = [
    'attribute vec2 aPos;',        // baked projection, [-1,1], y down
    'attribute float aT;',         // drift distance / max, 0..1
    'attribute float aSlot;',
    'attribute float aDe;',
    'uniform vec2 uOrigin;',       // spawn point, clip coords
    'uniform vec2 uScale;',        // cloud half-size, clip units
    'uniform float uDriftLen;',    // full-drift travel, clip units (y down)
    'uniform float uT;',           // progress in tnorm units (0..1+fade)
    'uniform float uFade;',        // fade duration, tnorm units
    'uniform float uPtSize;',
    'uniform float uAlpha;',
    'uniform vec3 uPal[9];',
    'varying vec3 vColor;',
    'varying float vA;',
    'void main(){',
    '  float travel = min(uT, aT) * uDriftLen;',
    '  vec2 p = uOrigin + vec2(aPos.x * uScale.x,',
    '                          -(aPos.y * uScale.y + travel));',
    '  float arrived = step(aT, uT);',
    '  float fade = 1.0 - clamp((uT - aT) / uFade, 0.0, 1.0);',
    '  float a = mix(1.0, fade, arrived);',
    '  vColor = uPal[int(min(aSlot, 8.0))];',
    '  vA = a * (0.25 + 0.75 * aDe) * uAlpha;',
    '  gl_Position = vec4(p, 0.0, 1.0);',
    '  gl_PointSize = uPtSize * (0.7 + 0.9 * aDe) * max(a, 0.001);',
    '}',
  ].join('\n');

  var FS = [
    'precision mediump float;',
    'varying vec3 vColor;',
    'varying float vA;',
    'void main(){',
    '  vec2 d = gl_PointCoord - vec2(0.5);',
    '  if (dot(d, d) > 0.25) discard;',
    '  gl_FragColor = vec4(vColor, vA);',
    '}',
  ].join('\n');

  // Sensor strip: full-width quad, brightness from a 1D histogram texture
  var SVS = [
    'attribute vec2 aXY;',
    'varying vec2 vUV;',
    'void main(){ vUV = aXY * 0.5 + 0.5; gl_Position = vec4(aXY, 0.0, 1.0); }',
  ].join('\n');

  var SFS = [
    'precision mediump float;',
    'uniform sampler2D uHist;',
    'uniform float uStripTop;',    // uv y where the strip begins
    'uniform float uBase;',        // baseline line brightness
    'varying vec2 vUV;',
    'void main(){',
    '  if (vUV.y > uStripTop) discard;',
    '  float h = texture2D(uHist, vec2(vUV.x, 0.5)).r;',
    '  float fall = 1.0 - vUV.y / uStripTop;',      // 1 at bottom edge
    '  float glow = h * fall * fall;',
    '  float line = uBase * step(vUV.y, uStripTop * 0.08);',
    '  vec3 c = vec3(0.43, 0.66, 0.86);',
    '  float a = min(glow * 0.8 + line, 0.9);',
    '  gl_FragColor = vec4(c, a);',
    '}',
  ].join('\n');

  function compile(vsSrc, fsSrc) {
    function sh(type, src) {
      var s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(s));
      }
      return s;
    }
    var p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(p));
    }
    return p;
  }

  // ---- State ---------------------------------------------------------------
  var meta = null, nPts = 0;
  var cpuPx = null, cpuT = null, cpuDe = null, orderByT = null;
  var prog = null, sProg = null, vbo = null, sVbo = null, histTex = null;
  var loc = {}, sLoc = {};
  var live = [];               // {ox, oy, driftLen, t0, dur, fade, ptr}
  var hist = new Float32Array(STRIP_BINS);
  var histBytes = new Uint8Array(STRIP_BINS);
  var running = false, lastNow = 0, glowLevel = 0;
  var dpr = Math.min(window.devicePixelRatio || 1, 1.5);

  function resize() {
    var w = window.innerWidth, h = window.innerHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
    drawFrame(performance.now());
  }

  function setupGL(buf) {
    prog = compile(VS, FS);
    sProg = compile(SVS, SFS);
    ['aPos', 'aT', 'aSlot', 'aDe'].forEach(function (a) {
      loc[a] = gl.getAttribLocation(prog, a);
    });
    ['uOrigin', 'uScale', 'uDriftLen', 'uT', 'uFade', 'uPtSize',
     'uAlpha'].forEach(function (u) {
      loc[u] = gl.getUniformLocation(prog, u);
    });
    loc.uPal = gl.getUniformLocation(prog, 'uPal') ||
               gl.getUniformLocation(prog, 'uPal[0]');
    sLoc.aXY = gl.getAttribLocation(sProg, 'aXY');
    ['uHist', 'uStripTop', 'uBase'].forEach(function (u) {
      sLoc[u] = gl.getUniformLocation(sProg, u);
    });

    vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, buf, gl.STATIC_DRAW);

    sVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, sVbo);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1]),
      gl.STATIC_DRAW);

    histTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, histTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, STRIP_BINS, 1, 0,
      gl.LUMINANCE, gl.UNSIGNED_BYTE, histBytes);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0, 0, 0, 0);
  }

  function cloudScale() {
    var m = Math.min(window.innerWidth, window.innerHeight) * CLOUD_HALF_W;
    return { x: 2 * m / window.innerWidth, y: 2 * m / window.innerHeight };
  }

  function spawn(clientX, clientY) {
    if (!meta) return;
    var sc = cloudScale();
    var ox = (clientX / window.innerWidth) * 2 - 1;
    var oy = -((clientY / window.innerHeight) * 2 - 1);
    // Keep the spawn point above the sensor strip with some travel room
    var stripClipY = -1 + 2 * (STRIP_H_PX / window.innerHeight);
    var minTravel = 0.25;
    if (oy < stripClipY + minTravel) oy = stripClipY + minTravel;
    var driftLen = oy - stripClipY;             // click -> strip, clip units
    var durS = (driftLen / 2) / SPEED;          // clip units -> viewport frac
    if (live.length >= MAX_LIVE) live.shift();
    live.push({
      ox: ox, oy: oy, driftLen: driftLen, t0: performance.now(),
      dur: durS * 1000, fade: FADE_FRAC, ptr: 0, scale: sc,
    });
    start();
  }

  function depositLandings(inst, tCur) {
    // Advance through arrival-sorted points, add landed charge to histogram
    var n = nPts;
    while (inst.ptr < n) {
      var i = orderByT[inst.ptr];
      if (cpuT[i] > tCur) break;
      var xClip = inst.ox + (cpuPx[i] / 32767) * inst.scale.x;
      var bin = Math.round(((xClip + 1) / 2) * (STRIP_BINS - 1));
      if (bin >= 0 && bin < STRIP_BINS) {
        hist[bin] += (0.3 + 0.7 * cpuDe[i] / 255) * 0.05;
      }
      inst.ptr++;
    }
  }

  function drawFrame(now) {
    var dt = lastNow ? (now - lastNow) / 1000 : 0.016;
    lastNow = now;
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Decay histogram
    var decay = Math.pow(0.5, dt / STRIP_HALFLIFE);
    glowLevel = 0;
    for (var b = 0; b < STRIP_BINS; b++) {
      hist[b] *= decay;
      if (hist[b] > glowLevel) glowLevel = hist[b];
      var v = hist[b] * 255;
      histBytes[b] = v > 255 ? 255 : v;
    }

    // Live interactions
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.vertexAttribPointer(loc.aPos, 2, gl.SHORT, true, 8, 0);
    gl.vertexAttribPointer(loc.aT, 1, gl.UNSIGNED_SHORT, true, 8, 4);
    gl.vertexAttribPointer(loc.aSlot, 1, gl.UNSIGNED_BYTE, false, 8, 6);
    gl.vertexAttribPointer(loc.aDe, 1, gl.UNSIGNED_BYTE, true, 8, 7);
    gl.enableVertexAttribArray(loc.aPos);
    gl.enableVertexAttribArray(loc.aT);
    gl.enableVertexAttribArray(loc.aSlot);
    gl.enableVertexAttribArray(loc.aDe);
    var pal = [];
    for (var k = 0; k < 9; k++) pal = pal.concat(PALETTE[k]);
    gl.uniform3fv(loc.uPal, pal);
    gl.uniform1f(loc.uPtSize, POINT_SIZE * dpr);
    gl.uniform1f(loc.uAlpha, GLOBAL_ALPHA);

    for (var j = live.length - 1; j >= 0; j--) {
      var inst = live[j];
      var tCur = (now - inst.t0) / inst.dur;   // tnorm units
      if (tCur > 1 + inst.fade) { live.splice(j, 1); continue; }
      depositLandings(inst, tCur);
      gl.uniform2f(loc.uOrigin, inst.ox, inst.oy);
      gl.uniform2f(loc.uScale, inst.scale.x, inst.scale.y);
      gl.uniform1f(loc.uDriftLen, inst.driftLen);
      gl.uniform1f(loc.uT, tCur);
      gl.uniform1f(loc.uFade, inst.fade);
      gl.drawArrays(gl.POINTS, 0, nPts);
    }

    // Sensor strip
    gl.useProgram(sProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, sVbo);
    gl.vertexAttribPointer(sLoc.aXY, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(sLoc.aXY);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, histTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, STRIP_BINS, 1,
      gl.LUMINANCE, gl.UNSIGNED_BYTE, histBytes);
    gl.uniform1i(sLoc.uHist, 0);
    gl.uniform1f(sLoc.uStripTop, STRIP_H_PX / window.innerHeight);
    gl.uniform1f(sLoc.uBase, 0.10);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  function loop(now) {
    drawFrame(now);
    if (live.length > 0 || glowLevel > 0.004) {
      requestAnimationFrame(loop);
    } else {
      running = false;   // idle: zero cost until the next click
      lastNow = 0;
    }
  }

  function start() {
    if (running) return;
    running = true;
    lastNow = 0;
    requestAnimationFrame(loop);
  }

  // ---- Init ----------------------------------------------------------------
  fetch('assets/bg/bg_event.json')
    .then(function (r) { return r.json(); })
    .then(function (m) {
      meta = m;
      return fetch('assets/bg/bg_event.bin');
    })
    .then(function (r) { return r.arrayBuffer(); })
    .then(function (buf) {
      nPts = meta.n;
      // CPU copies for the landing histogram
      var dv = new DataView(buf);
      cpuPx = new Int16Array(nPts);
      cpuT = new Float32Array(nPts);
      cpuDe = new Uint8Array(nPts);
      for (var i = 0; i < nPts; i++) {
        cpuPx[i] = dv.getInt16(i * 8, true);
        cpuT[i] = dv.getUint16(i * 8 + 4, true) / 65535;
        cpuDe[i] = dv.getUint8(i * 8 + 7);
      }
      orderByT = new Uint32Array(nPts);
      for (i = 0; i < nPts; i++) orderByT[i] = i;
      orderByT.sort(function (a, b) { return cpuT[a] - cpuT[b]; });

      setupGL(buf);
      window.addEventListener('resize', resize);
      resize();

      if (reduceMotion) return;  // static page, no animation, no listeners

      // Clicks anywhere on non-interactive page area spawn an interaction
      document.addEventListener('click', function (e) {
        if (e.target.closest('a, button, input, select, textarea, nav')) return;
        spawn(e.clientX, e.clientY);
      });

      // Welcome shot so first-time visitors see what the page does
      setTimeout(function () {
        spawn(window.innerWidth * 0.62, window.innerHeight * 0.30);
      }, 700);
    })
    .catch(function () { /* background stays plain on any failure */ });
})();
