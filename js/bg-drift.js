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
  var SPEED = 260;           // drift speed, CSS px / s
  var GLOBAL_ALPHA = 0.55;   // overall subtlety dial
  var POINT_SIZE = 2.2;      // in-flight point size at dpr=1
  var HIT_SIZE = 1.8;        // landed point size at dpr=1
  var PLANE_FADE = 7.0;      // s, landed glow lifetime
  var MAX_LIVE = 3;
  var PLANE_FAR_SCALE = 0.55;   // far edge width / near edge width
  var PALETTE = [
    [0.43, 0.66, 0.86], [0.79, 0.64, 0.36], [0.62, 0.71, 0.66],
    [0.73, 0.63, 0.85], [0.85, 0.54, 0.54], [0.54, 0.77, 0.85],
    [0.77, 0.85, 0.54], [0.85, 0.71, 0.54], [0.42, 0.47, 0.53],
  ];

  // ---- Point shader (document-space math) -----------------------------------
  var VS = [
    'attribute vec2 aPos;',
    'attribute float aDepth;',     // 0 far .. 1 near
    'attribute float aSlot;',
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
    'uniform float uFarScale;',
    'uniform float uPtSize;',
    'uniform float uHitSize;',
    'uniform float uPlaneFade;',
    'uniform float uAlpha;',
    'uniform vec3 uPal[9];',
    'varying vec3 vColor;',
    'varying float vA;',
    'void main(){',
    '  float x0 = uOrigin.x + aPos.x * uScalePx;',
    '  float y0 = uOrigin.y + aPos.y * uScalePx;',
    '  float yLand = mix(uFarY, uNearY, aDepth);',
    '  float tArr = max(yLand - y0, 0.0) / uSpeed;',
    '  float arrived = step(tArr, uTime);',
    '  float y = y0 + min(uTime, tArr) * uSpeed;',
    '  float xLand = uCx + (x0 - uCx) * mix(uFarScale, 1.0, aDepth);',
    '  float x = mix(x0, xLand, arrived);',
    '  float aFly = 0.25 + 0.75 * aDe;',
    '  float decay = 1.0 - clamp((uTime - tArr) / uPlaneFade, 0.0, 1.0);',
    '  float alpha = mix(aFly, aFly * decay, arrived) * uAlpha;',
    '  vColor = uPal[int(min(aSlot, 8.0))];',
    '  vA = alpha;',
    '  float sy = y - uScroll;',
    '  vec2 clip = vec2(x / uVp.x * 2.0 - 1.0, 1.0 - sy / uVp.y * 2.0);',
    '  gl_Position = vec4(clip, 0.0, 1.0);',
    '  float sz = mix(uPtSize * (0.7 + 0.9 * aDe), uHitSize, arrived);',
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
  var prog = null, pProg = null, vbo = null, planeVbo = null;
  var loc = {}, pLoc = {};
  var planeVerts = null, planeLineStart = 0, planeLineCount = 0;
  var plane = { nearY: 0, farY: 0, cx: 0 };
  var live = [];   // {ox, oy, t0, scalePx, tEnd}
  var running = false;
  var dpr = Math.min(window.devicePixelRatio || 1, 1.5);

  function resize() {
    canvas.width = Math.round(window.innerWidth * dpr);
    canvas.height = Math.round(window.innerHeight * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
    measurePlane();
    paint();
  }

  function measurePlane() {
    var r = planeEl.getBoundingClientRect();
    var top = r.top + window.scrollY;
    plane.farY = top + r.height * 0.22;
    plane.nearY = top + r.height * 0.88;
    plane.cx = window.innerWidth / 2;
    buildPlaneGeometry();
  }

  function buildPlaneGeometry() {
    var vw = window.innerWidth;
    var nearHalf = vw * 0.44;
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

  function setupGL(buf) {
    prog = compile(VS, FS);
    pProg = compile(PVS, PFS);
    ['aPos', 'aDepth', 'aSlot', 'aDe'].forEach(function (a) {
      loc[a] = gl.getAttribLocation(prog, a);
    });
    ['uVp', 'uScroll', 'uOrigin', 'uScalePx', 'uSpeed', 'uTime', 'uNearY',
     'uFarY', 'uCx', 'uFarScale', 'uPtSize', 'uHitSize', 'uPlaneFade',
     'uAlpha'].forEach(function (u) {
      loc[u] = gl.getUniformLocation(prog, u);
    });
    loc.uPal = gl.getUniformLocation(prog, 'uPal') ||
               gl.getUniformLocation(prog, 'uPal[0]');
    pLoc.aXY = gl.getAttribLocation(pProg, 'aXY');
    pLoc.aA = gl.getAttribLocation(pProg, 'aA');
    ['uVp', 'uScroll', 'uColor'].forEach(function (u) {
      pLoc[u] = gl.getUniformLocation(pProg, u);
    });

    vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, buf, gl.STATIC_DRAW);

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
    if (oy > plane.farY - 120) return;   // no spawns on/below the plane
    var scalePx = Math.min(window.innerWidth, window.innerHeight) * CLOUD_HALF_W;
    var tEnd = (plane.nearY - oy + scalePx) / SPEED + PLANE_FADE + 0.5;
    if (live.length >= MAX_LIVE) live.shift();
    live.push({ ox: ox, oy: oy, t0: performance.now(), scalePx: scalePx, tEnd: tEnd });
    start();
  }

  function paint() {
    var now = performance.now();
    gl.clear(gl.COLOR_BUFFER_BIT);
    var vw = window.innerWidth, vh = window.innerHeight;
    var sc = window.scrollY;

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
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.vertexAttribPointer(loc.aPos, 2, gl.SHORT, true, 8, 0);
    gl.vertexAttribPointer(loc.aDepth, 1, gl.UNSIGNED_SHORT, true, 8, 4);
    gl.vertexAttribPointer(loc.aSlot, 1, gl.UNSIGNED_BYTE, false, 8, 6);
    gl.vertexAttribPointer(loc.aDe, 1, gl.UNSIGNED_BYTE, true, 8, 7);
    gl.enableVertexAttribArray(loc.aPos);
    gl.enableVertexAttribArray(loc.aDepth);
    gl.enableVertexAttribArray(loc.aSlot);
    gl.enableVertexAttribArray(loc.aDe);
    var pal = [];
    for (var k = 0; k < 9; k++) pal = pal.concat(PALETTE[k]);
    gl.uniform3fv(loc.uPal, pal);
    gl.uniform2f(loc.uVp, vw, vh);
    gl.uniform1f(loc.uScroll, sc);
    gl.uniform1f(loc.uSpeed, SPEED);
    gl.uniform1f(loc.uNearY, plane.nearY);
    gl.uniform1f(loc.uFarY, plane.farY);
    gl.uniform1f(loc.uCx, plane.cx);
    gl.uniform1f(loc.uFarScale, PLANE_FAR_SCALE);
    gl.uniform1f(loc.uPtSize, POINT_SIZE * dpr);
    gl.uniform1f(loc.uHitSize, HIT_SIZE * dpr);
    gl.uniform1f(loc.uPlaneFade, PLANE_FADE);
    gl.uniform1f(loc.uAlpha, GLOBAL_ALPHA);

    for (var j = live.length - 1; j >= 0; j--) {
      var inst = live[j];
      var t = (now - inst.t0) / 1000;
      if (t > inst.tEnd) { live.splice(j, 1); continue; }
      gl.uniform2f(loc.uOrigin, inst.ox, inst.oy);
      gl.uniform1f(loc.uScalePx, inst.scalePx);
      gl.uniform1f(loc.uTime, t);
      gl.drawArrays(gl.POINTS, 0, nPts);
    }
  }

  function loop() {
    paint();
    if (live.length > 0) {
      requestAnimationFrame(loop);
    } else {
      running = false;   // idle again: only scroll/resize repaints
    }
  }

  function start() {
    if (running) return;
    running = true;
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
      setupGL(buf);
      window.addEventListener('resize', resize);
      window.addEventListener('scroll', function () {
        if (!running) paint();     // keep the static plane glued to the page
      }, { passive: true });
      resize();

      if (reduceMotion) return;    // static plane only

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
