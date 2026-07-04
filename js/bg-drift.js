/* ---------------------------------------------------------------------------
   Interaction drift background ("the page is a detector").

   A real simulated neutrino interaction (JAXTPC / doraemon production) is
   baked offline into a tilted 2D projection plus each deposit's depth along
   the viewing axis. Clicking spawns the cloud at the click point in DOCUMENT
   space: it drifts straight down through the page (scroll to follow it) and
   lands on a perspective-drawn sensor plane at the bottom of the page.
   Landing time and position depend on each point's depth: far-depth charge
   lands on the far edge of the plane, near charge on the near edge, so the
   fade front sweeps through the 3D shape correctly ("cheating 3D").
   Landed charge glows on the plane where it hit, then decays.

   Data: assets/bg/bg_event.bin, 8 bytes/point:
     int16 px, int16 py (projection, +-32767 = +-1, y down),
     uint16 depth (0 far .. 1 near), uint8 slot (track), uint8 de (log dE).

   Efficiency: one static VBO, all motion in the vertex shader, one draw
   call per live interaction. The rAF loop runs only while charge is in
   flight or landed glow is decaying; idle cost is zero (a scroll listener
   just re-paints the static plane).
   --------------------------------------------------------------------------- */
(function () {
  'use strict';

  var canvas = document.getElementById('bg-canvas');
  var planeEl = document.getElementById('detector');
  if (!canvas || !planeEl) return;
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var gl = canvas.getContext('webgl', {
    alpha: true, antialias: false, depth: false, stencil: false,
    powerPreference: 'low-power',
  });
  if (!gl) return;

  // ---- Tunables -----------------------------------------------------------
  var CLOUD_HALF_W = 0.30;   // cloud half-width, fraction of min(vw, vh)
  var SPEED = 185;           // drift speed, CSS px / s
  var GLOBAL_ALPHA = 0.60;   // overall subtlety dial
  var POINT_SIZE = 2.4;      // in-flight point size at dpr=1
  var HIT_SIZE = 1.8;        // landed point size at dpr=1
  var PLANE_FADE = 4.5;      // s, landed glow lifetime
  var MAX_LIVE = 16;
  var PLANE_FAR_SCALE = 0.55;   // far edge width / near edge width
  var DEPTH_SPAN = 0.40;     // cloud depth extent as fraction of plane depth
  var LAND_BLEND = 0.35;     // s, pre-landing ease toward the hit position
  var EMPH_POW = 5.0;        // JAXTPC viewer charge emphasis: de^pow
  var EMPH_AMT = 0.75;       // 0 = uniform, 1 = full emphasis
  var COSMIC_MEAN_S = 9;    // mean seconds between ambient cosmic rays
  var COSMIC_ALPHA = 0.75;   // cosmics dimmer than clicked events
  var AMBIENT_ALPHA = 0.14;  // medium dust ceiling (pulses go brighter)
  var AMBIENT_DENSITY = 1 / 15000;  // dust points per document px^2

  // ---- Point shader (document-space math) -----------------------------------
  var VS = [
    'attribute vec2 aPos;',
    'attribute float aDepth;',     // 0 far .. 1 near
    'attribute float aHue;',       // track hue (viewer golden-ratio hash)
    'attribute float aDe;',
    'uniform vec2 uVp;',           // viewport CSS px
    'uniform float uScroll;',      // window scrollY
    'uniform vec2 uOrigin;',       // spawn point, document px
    'uniform float uScalePx;',     // cloud half-size px
    'uniform float uSpeed;',       // px/s
    'uniform float uTime;',        // s since spawn',
    'uniform float uNearY;',       // plane near edge, document px
    'uniform float uFarY;',        // plane far edge, document px
    'uniform float uCx;',          // plane center x, px
    'uniform float uNearHalf;',    // plane near-edge half-width, px
    'uniform float uFarScale;',
    'uniform float uPtSize;',
    'uniform float uHitSize;',
    'uniform float uPlaneFade;',
    'uniform float uAlpha;',
    'uniform float uDepthOff;',    // per-click vertex depth, 0 far .. 1 near
    'uniform float uDepthSpan;',
    'uniform float uBlend;',       // s, pre-landing position ease
    'uniform float uEmphPow;',
    'uniform float uEmphAmt;',
    'uniform float uPersp;',       // per-click perspective size factor
    'uniform vec2 uRot;',          // (cos, sin) in-plane rotation (cosmics)
    'uniform float uAlphaMul;',    // per-instance dimming (cosmics)
    'varying vec3 vColor;',
    'varying float vA;',
    'void main(){',
    '  vec2 rp = vec2(aPos.x * uRot.x - aPos.y * uRot.y,',
    '                 aPos.x * uRot.y + aPos.y * uRot.x);',
    '  float x0 = uOrigin.x + rp.x * uScalePx;',
    '  float y0 = uOrigin.y + rp.y * uScalePx;',
    '  float effD = clamp(uDepthOff + (aDepth - 0.5) * uDepthSpan, 0.0, 1.0);',
    '  float yLand = mix(uFarY, uNearY, effD);',
    '  float valid = step(y0, yLand);',   // spawned past its landing surface: dead
    '  float tArr = max(yLand - y0, 0.0) / uSpeed;',
    '  float arrived = step(tArr, uTime);',
    '  float y = y0 + min(uTime, tArr) * uSpeed;',
    '  float x = x0;',                    // charge falls straight down, always
    // Inside the detector? Plane half-width at this landing depth.
    '  float halfW = uNearHalf * mix(uFarScale, 1.0, effD);',
    '  float inside = step(abs(x0 - uCx), halfW);',
    '  float pre = clamp((uTime - tArr) / uBlend + 1.0, 0.0, 1.0);',
    '  pre = pre * pre * (3.0 - 2.0 * pre);',
    '  float emph = pow(clamp(aDe, 0.001, 1.0), uEmphPow);',
    '  float eF = mix(1.0, emph, uEmphAmt);',
    '  float aFly = 0.85 * max(eF, 0.03);',
    '  float dt = uTime - tArr;',
    '  float d1 = 1.0 - clamp(dt / uPlaneFade, 0.0, 1.0);',
    '  float flash = 1.0 + 0.6 * exp(-max(dt, 0.0) * 5.0);',
    // Outside the detector footprint: never visible at all.
    '  float alpha = mix(aFly, aFly * flash * pow(d1, 1.8), arrived) * uAlpha * uAlphaMul * valid * inside;',
    // HSL(hue, 0.78, 0.55): exact port of the viewer's hsl2rgb track colors
    '  vec3 q = clamp(abs(mod(aHue * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);',
    '  vColor = 0.55 + 0.702 * (q - 0.5);',
    '  vA = alpha;',
    '  float sy = y - uScroll;',
    '  vec2 clip = vec2(x / uVp.x * 2.0 - 1.0, 1.0 - sy / uVp.y * 2.0);',
    '  gl_Position = vec4(clip, 0.0, 1.0);',
    '  float szF = uPtSize * mix(1.0, max(eF, 0.2), uEmphAmt);',
    '  float sz = mix(szF, uHitSize, pre) * uPersp;',
    '  gl_PointSize = sz * max(sign(alpha), 0.0);',
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

  // ---- Plane shader: static geometry in document px -------------------------
  var PVS = [
    'attribute vec2 aXY;',         // document px
    'attribute float aA;',         // per-vertex alpha
    'uniform vec2 uVp;',
    'uniform float uScroll;',
    'varying float vA;',
    'void main(){',
    '  vA = aA;',
    '  float sy = aXY.y - uScroll;',
    '  gl_Position = vec4(aXY.x / uVp.x * 2.0 - 1.0, 1.0 - sy / uVp.y * 2.0, 0.0, 1.0);',
    '}',
  ].join('\n');

  var PFS = [
    'precision mediump float;',
    'uniform vec3 uColor;',
    'varying float vA;',
    'void main(){ gl_FragColor = vec4(uColor, vA); }',
  ].join('\n');

  // ---- Ambient medium: diffusion wander + upward creep + Ar-39 pulses ------
  var AVS = [
    'attribute vec4 aSeed;',       // baseX px, baseY doc px, seed1, seed2
    'uniform vec2 uVp;',
    'uniform float uScroll;',
    'uniform float uTime;',        // s
    'uniform float uDpr;',
    'uniform float uAmbA;',
    'varying float vA;',
    'void main(){',
    '  float s1 = aSeed.z, s2 = aSeed.w;',
    '  float wob = 6.0 + 9.0 * s2;',
    '  float x = aSeed.x + wob * sin(uTime * (0.05 + 0.08 * s1) + s1 * 6.2831);',
    '  float y = aSeed.y + wob * 0.6 * cos(uTime * (0.04 + 0.07 * s2) + s2 * 6.2831);',
    // Rare Ar-39 blip: instant on, slow glow-off
    '  float per = 40.0 + 55.0 * s1;',
    '  float dtp = (fract(uTime / per + s2) - 0.1) * per;',
    '  float pulse = smoothstep(0.0, 0.15, dtp) * exp(-dtp * 1.8);',
    '  float sy = y - uScroll;',
    '  gl_Position = vec4(x / uVp.x * 2.0 - 1.0, 1.0 - sy / uVp.y * 2.0, 0.0, 1.0);',
    '  vA = uAmbA * (0.5 + 0.5 * s1) + pulse * 0.35;',
    '  gl_PointSize = (2.0 + 1.4 * s2 + pulse * 3.5) * uDpr;',
    '}',
  ].join('\n');

  var AFS = [
    'precision mediump float;',
    'varying float vA;',
    'void main(){',
    // Soft splat: (1-r^2)^2 falloff so motes glow instead of reading as pixels
    '  vec2 d = gl_PointCoord - vec2(0.5);',
    '  float r2 = dot(d, d) * 4.0;',
    '  if (r2 > 1.0) discard;',
    '  float fall = 1.0 - r2;',
    '  gl_FragColor = vec4(0.55, 0.68, 0.80, vA * fall * fall);',
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
  var manifest = null;
  var events = [];    // event pool: {vbo, n} once loaded
  var cosmics = [];   // cosmic muon pool: {vbo, n} once loaded
  var prog = null, pProg = null, aProg = null, planeVbo = null, ambVbo = null;
  var loc = {}, pLoc = {}, aLoc = {};
  var planeVerts = null, planeLineStart = 0, planeLineCount = 0;
  var ambN = 0;
  var plane = { nearY: 0, farY: 0, cx: 0, nearHalf: 0 };
  var live = [];   // {pool, idx, ox, oy, rot, alphaMul, t0, scalePx, depthOff, persp, tEnd}
  var loopStarted = false;
  var t0Wall = performance.now();
  var dpr = Math.min(window.devicePixelRatio || 1, 1.5);

  function resize() {
    canvas.width = Math.round(window.innerWidth * dpr);
    canvas.height = Math.round(window.innerHeight * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
    measurePlane();
    buildAmbient();
    paint();
  }

  function buildAmbient() {
    var h = Math.max(plane.farY, window.innerHeight);
    var n = Math.min(2500, Math.max(150,
      Math.floor(window.innerWidth * h * AMBIENT_DENSITY)));
    var v = new Float32Array(n * 4);
    for (var i = 0; i < n; i++) {
      v[i * 4] = Math.random() * window.innerWidth;
      v[i * 4 + 1] = Math.random() * h;
      v[i * 4 + 2] = Math.random();
      v[i * 4 + 3] = Math.random();
    }
    ambN = n;
    if (ambVbo) {
      gl.bindBuffer(gl.ARRAY_BUFFER, ambVbo);
      gl.bufferData(gl.ARRAY_BUFFER, v, gl.STATIC_DRAW);
    } else {
      ambPending = v;
    }
  }
  var ambPending = null;

  function measurePlane() {
    var r = planeEl.getBoundingClientRect();
    var top = r.top + window.scrollY;
    plane.farY = top + r.height * 0.22;
    plane.nearY = top + r.height * 0.88;
    plane.cx = window.innerWidth / 2;
    plane.nearHalf = window.innerWidth * 0.44;
    buildPlaneGeometry();
  }

  function buildPlaneGeometry() {
    var nearHalf = plane.nearHalf;
    var farHalf = nearHalf * PLANE_FAR_SCALE;
    var cx = plane.cx, ny = plane.nearY, fy = plane.farY;
    var fillA = 0.045, edgeA = 0.22, gridA = 0.09;
    var v = [];
    // fill: two triangles (x, y, alpha)
    v.push(cx - farHalf, fy, fillA, cx + farHalf, fy, fillA, cx - nearHalf, ny, fillA);
    v.push(cx + farHalf, fy, fillA, cx + nearHalf, ny, fillA, cx - nearHalf, ny, fillA);
    planeLineStart = v.length / 3;
    var lines = [];
    // outline
    lines.push([cx - farHalf, fy, cx + farHalf, fy, edgeA]);
    lines.push([cx - nearHalf, ny, cx + nearHalf, ny, edgeA]);
    lines.push([cx - farHalf, fy, cx - nearHalf, ny, edgeA]);
    lines.push([cx + farHalf, fy, cx + nearHalf, ny, edgeA]);
    // perspective grid: verticals converge, horizontals spaced by depth
    var i, t;
    for (i = 1; i < 8; i++) {
      t = i / 8;
      lines.push([cx - farHalf + 2 * farHalf * t, fy,
                  cx - nearHalf + 2 * nearHalf * t, ny, gridA]);
    }
    for (i = 1; i < 4; i++) {
      t = i / 4;
      var y = fy + (ny - fy) * t;
      var half = farHalf + (nearHalf - farHalf) * t;
      lines.push([cx - half, y, cx + half, y, gridA]);
    }
    for (i = 0; i < lines.length; i++) {
      var L = lines[i];
      v.push(L[0], L[1], L[4], L[2], L[3], L[4]);
    }
    planeLineCount = v.length / 3 - planeLineStart;
    planeVerts = new Float32Array(v);
    if (planeVbo) {
      gl.bindBuffer(gl.ARRAY_BUFFER, planeVbo);
      gl.bufferData(gl.ARRAY_BUFFER, planeVerts, gl.DYNAMIC_DRAW);
    }
  }

  function setupGL() {
    prog = compile(VS, FS);
    pProg = compile(PVS, PFS);
    ['aPos', 'aDepth', 'aHue', 'aDe'].forEach(function (a) {
      loc[a] = gl.getAttribLocation(prog, a);
    });
    ['uVp', 'uScroll', 'uOrigin', 'uScalePx', 'uSpeed', 'uTime', 'uNearY',
     'uFarY', 'uCx', 'uNearHalf', 'uFarScale', 'uPtSize', 'uHitSize', 'uPlaneFade',
     'uAlpha', 'uDepthOff', 'uDepthSpan', 'uBlend', 'uEmphPow', 'uEmphAmt',
     'uPersp', 'uRot', 'uAlphaMul'].forEach(function (u) {
      loc[u] = gl.getUniformLocation(prog, u);
    });
    pLoc.aXY = gl.getAttribLocation(pProg, 'aXY');
    pLoc.aA = gl.getAttribLocation(pProg, 'aA');
    ['uVp', 'uScroll', 'uColor'].forEach(function (u) {
      pLoc[u] = gl.getUniformLocation(pProg, u);
    });

    aProg = compile(AVS, AFS);
    aLoc.aSeed = gl.getAttribLocation(aProg, 'aSeed');
    ['uVp', 'uScroll', 'uTime', 'uDpr', 'uAmbA'].forEach(function (u) {
      aLoc[u] = gl.getUniformLocation(aProg, u);
    });
    ambVbo = gl.createBuffer();
    if (ambPending) {
      gl.bindBuffer(gl.ARRAY_BUFFER, ambVbo);
      gl.bufferData(gl.ARRAY_BUFFER, ambPending, gl.STATIC_DRAW);
      ambPending = null;
    }

    planeVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, planeVbo);
    gl.bufferData(gl.ARRAY_BUFFER, planeVerts, gl.DYNAMIC_DRAW);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0, 0, 0, 0);
  }

  function spawn(clientX, clientY) {
    var oy = clientY + window.scrollY;
    var ox = clientX;
    measurePlane();
    if (oy > plane.nearY) return;   // no spawns below the plane's near edge
    // Random event from whatever part of the pool has loaded so far
    var avail = [];
    for (var i = 0; i < events.length; i++) if (events[i]) avail.push(i);
    if (!avail.length) return;
    var evIdx = avail[Math.floor(Math.random() * avail.length)];
    // Random vertex depth per click: deeper clouds render smaller and land
    // nearer the far edge of the plane.
    var depthOff = 0.12 + 0.76 * Math.random();
    var persp = PLANE_FAR_SCALE + (1 - PLANE_FAR_SCALE) * depthOff;
    var scalePx = Math.min(window.innerWidth, window.innerHeight) * CLOUD_HALF_W * persp;
    addInstance({ pool: events, idx: evIdx, ox: ox, oy: oy,
                  rot: [1, 0], alphaMul: 1, depthOff: depthOff, persp: persp,
                  scalePx: scalePx });
  }

  function spawnCosmic() {
    var avail = [];
    for (var i = 0; i < cosmics.length; i++) if (cosmics[i]) avail.push(i);
    if (!avail.length) return;
    var idx = avail[Math.floor(Math.random() * avail.length)];
    // Steep-ish angle from horizontal, leaning either way (cosmic zenith-ish)
    var ang = (38 + 47 * Math.random()) * Math.PI / 180;
    if (Math.random() < 0.5) ang = Math.PI - ang;
    var depthOff = 0.12 + 0.76 * Math.random();
    var persp = PLANE_FAR_SCALE + (1 - PLANE_FAR_SCALE) * depthOff;
    var halfLen = Math.max(window.innerWidth, window.innerHeight) *
                  (0.45 + 0.35 * Math.random()) * persp;
    // Spawn near the current viewport so the visitor actually sees it
    var ox = window.innerWidth * (0.15 + 0.7 * Math.random());
    var oy = window.scrollY + window.innerHeight * (0.1 + 0.45 * Math.random());
    if (oy > plane.farY - 200) oy = Math.max(120, plane.farY - 200);
    addInstance({ pool: cosmics, idx: idx, ox: ox, oy: oy,
                  rot: [Math.cos(ang), Math.sin(ang)], alphaMul: COSMIC_ALPHA,
                  depthOff: depthOff, persp: persp, scalePx: halfLen });
  }

  function addInstance(inst) {
    inst.t0 = performance.now();
    inst.tEnd = (plane.nearY - inst.oy + inst.scalePx) / SPEED + PLANE_FADE + 0.5;
    if (live.length >= MAX_LIVE) live.shift();
    live.push(inst);
  }

  function scheduleCosmic() {
    // Poisson arrivals, clamped so it never machine-guns or goes silent
    var delay = Math.min(60, Math.max(4,
      -Math.log(1 - Math.random()) * COSMIC_MEAN_S)) * 1000;
    setTimeout(function () {
      if (!document.hidden) spawnCosmic();
      scheduleCosmic();
    }, delay);
  }

  function paint() {
    var now = performance.now();
    gl.clear(gl.COLOR_BUFFER_BIT);
    var vw = window.innerWidth, vh = window.innerHeight;
    var sc = window.scrollY;

    // Ambient medium (drawn behind everything)
    if (ambN > 0 && ambVbo) {
      gl.useProgram(aProg);
      gl.bindBuffer(gl.ARRAY_BUFFER, ambVbo);
      gl.vertexAttribPointer(aLoc.aSeed, 4, gl.FLOAT, false, 16, 0);
      gl.enableVertexAttribArray(aLoc.aSeed);
      gl.uniform2f(aLoc.uVp, vw, vh);
      gl.uniform1f(aLoc.uScroll, sc);
      gl.uniform1f(aLoc.uTime, reduceMotion ? 0 : (now - t0Wall) / 1000);
      gl.uniform1f(aLoc.uDpr, dpr);
      gl.uniform1f(aLoc.uAmbA, AMBIENT_ALPHA);
      gl.drawArrays(gl.POINTS, 0, ambN);
    }

    // Sensor plane
    gl.useProgram(pProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, planeVbo);
    gl.vertexAttribPointer(pLoc.aXY, 2, gl.FLOAT, false, 12, 0);
    gl.vertexAttribPointer(pLoc.aA, 1, gl.FLOAT, false, 12, 8);
    gl.enableVertexAttribArray(pLoc.aXY);
    gl.enableVertexAttribArray(pLoc.aA);
    gl.uniform2f(pLoc.uVp, vw, vh);
    gl.uniform1f(pLoc.uScroll, sc);
    gl.uniform3f(pLoc.uColor, 0.43, 0.66, 0.86);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.drawArrays(gl.LINES, planeLineStart, planeLineCount);

    // Interactions
    if (live.length === 0) return;
    gl.useProgram(prog);
    gl.uniform2f(loc.uVp, vw, vh);
    gl.uniform1f(loc.uScroll, sc);
    gl.uniform1f(loc.uSpeed, SPEED);
    gl.uniform1f(loc.uNearY, plane.nearY);
    gl.uniform1f(loc.uFarY, plane.farY);
    gl.uniform1f(loc.uCx, plane.cx);
    gl.uniform1f(loc.uNearHalf, plane.nearHalf);
    gl.uniform1f(loc.uFarScale, PLANE_FAR_SCALE);
    gl.uniform1f(loc.uPtSize, POINT_SIZE * dpr);
    gl.uniform1f(loc.uHitSize, HIT_SIZE * dpr);
    gl.uniform1f(loc.uPlaneFade, PLANE_FADE);
    gl.uniform1f(loc.uAlpha, GLOBAL_ALPHA);
    gl.uniform1f(loc.uDepthSpan, DEPTH_SPAN);
    gl.uniform1f(loc.uBlend, LAND_BLEND);
    gl.uniform1f(loc.uEmphPow, EMPH_POW);
    gl.uniform1f(loc.uEmphAmt, EMPH_AMT);

    for (var j = live.length - 1; j >= 0; j--) {
      var inst = live[j];
      var ev = inst.pool[inst.idx];
      var t = (now - inst.t0) / 1000;
      if (!ev || t > inst.tEnd) { live.splice(j, 1); continue; }
      gl.bindBuffer(gl.ARRAY_BUFFER, ev.vbo);
      gl.vertexAttribPointer(loc.aPos, 2, gl.SHORT, true, 8, 0);
      gl.vertexAttribPointer(loc.aDepth, 1, gl.UNSIGNED_SHORT, true, 8, 4);
      gl.vertexAttribPointer(loc.aHue, 1, gl.UNSIGNED_BYTE, true, 8, 6);
      gl.vertexAttribPointer(loc.aDe, 1, gl.UNSIGNED_BYTE, true, 8, 7);
      gl.enableVertexAttribArray(loc.aPos);
      gl.enableVertexAttribArray(loc.aDepth);
      gl.enableVertexAttribArray(loc.aHue);
      gl.enableVertexAttribArray(loc.aDe);
      gl.uniform2f(loc.uOrigin, inst.ox, inst.oy);
      gl.uniform1f(loc.uScalePx, inst.scalePx);
      gl.uniform1f(loc.uTime, t);
      gl.uniform1f(loc.uDepthOff, inst.depthOff);
      gl.uniform1f(loc.uPersp, inst.persp);
      gl.uniform2f(loc.uRot, inst.rot[0], inst.rot[1]);
      gl.uniform1f(loc.uAlphaMul, inst.alphaMul);
      gl.drawArrays(gl.POINTS, 0, ev.n);
    }
  }

  // The ambient medium animates continuously, so the loop always runs:
  // full rate while events are in flight, throttled to ~30fps otherwise.
  // rAF stops on hidden tabs, which pauses the whole chain.
  function loop() {
    paint();
    if (live.length > 0) {
      requestAnimationFrame(loop);
    } else {
      setTimeout(function () { requestAnimationFrame(loop); }, 33);
    }
  }

  function start() {
    if (loopStarted) return;
    loopStarted = true;
    requestAnimationFrame(loop);
  }

  // ---- Init ----------------------------------------------------------------
  function loadPool(list, store, i) {
    return fetch('assets/bg/' + list[i].file + '?v=__BUILD__')
      .then(function (r) { return r.arrayBuffer(); })
      .then(function (buf) {
        var b = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, buf, gl.STATIC_DRAW);
        store[i] = { vbo: b, n: list[i].n };
      });
  }
  function loadEvent(i) { return loadPool(manifest.events, events, i); }
  function loadCosmic(i) { return loadPool(manifest.cosmics, cosmics, i); }

  fetch('assets/bg/bg_events.json?v=__BUILD__')
    .then(function (r) { return r.json(); })
    .then(function (m) {
      manifest = m;
      events = new Array(m.events.length);
      cosmics = new Array((m.cosmics || []).length);
      setupGL();
      window.addEventListener('resize', resize);
      window.addEventListener('scroll', function () {
        if (!loopStarted) paint();  // static mode: keep plane glued to page
      }, { passive: true });
      resize();
      if (!reduceMotion) start();   // continuous ambient loop
      return loadEvent(0);
    })
    .then(function () {
      // Prefetch the rest of the pool immediately and in parallel: small
      // files on one HTTP/2 connection multiplex fine, so the whole pool is
      // typically resident within a couple of seconds. Skipped for
      // data-saver / very slow connections (they still get event 0 on every
      // click). A failed fetch retries once after 5s.
      var conn = navigator.connection || {};
      var frugal = conn.saveData || /2g/.test(conn.effectiveType || '');
      function retrying(fn, i) {
        fn(i).catch(function () {
          setTimeout(function () { fn(i).catch(function () {}); }, 5000);
        });
      }
      if (!frugal) {
        for (var i = 1; i < manifest.events.length; i++) retrying(loadEvent, i);
        for (var k = 0; k < cosmics.length; k++) retrying(loadCosmic, k);
      }

      if (reduceMotion) return;    // static dust + plane only

      document.addEventListener('click', function (e) {
        if (e.target.closest('a, button, input, select, textarea, nav')) return;
        spawn(e.clientX, e.clientY);
      });

      // Ambient cosmic rays (needs the cosmic pool, so not for frugal mode)
      if (!frugal && cosmics.length) scheduleCosmic();

      // Welcome shot so first-time visitors see what the page does
      setTimeout(function () {
        spawn(window.innerWidth * 0.62, window.innerHeight * 0.30);
      }, 700);
    })
    .catch(function () { /* background stays plain on any failure */ });
})();
