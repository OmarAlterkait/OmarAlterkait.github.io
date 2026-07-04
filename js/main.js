/* ---------------------------------------------------------------------------
   Background visualization (PLACEHOLDER).
   A subtle drifting particle field so the site has depth while we design the
   real visualization. Swap out initBackground() with the real thing later.
   --------------------------------------------------------------------------- */

/* Assemble the email address at runtime so it never appears in the page
   source, which is what basic scraper bots read. */
(function () {
  const user = ['omar', 'alterkait'].join('.');
  const domain = ['tufts', 'edu'].join('.');
  const address = user + '@' + domain;
  document.querySelectorAll('.js-email').forEach((el) => {
    el.setAttribute('href', 'mailto:' + address);
    if (el.classList.contains('js-email-text')) el.textContent = address;
  });
})();

(function () {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let width, height, particles;

  const PARTICLE_DENSITY = 1 / 18000;   // particles per pixel
  const MAX_SPEED = 0.15;

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
    const count = Math.max(30, Math.floor(width * height * PARTICLE_DENSITY));
    particles = Array.from({ length: count }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * MAX_SPEED,
      vy: (Math.random() - 0.5) * MAX_SPEED,
      r: 0.6 + Math.random() * 1.4,
      a: 0.15 + Math.random() * 0.35,
    }));
  }

  function step() {
    ctx.clearRect(0, 0, width, height);

    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x += width;
      if (p.x > width) p.x -= width;
      if (p.y < 0) p.y += height;
      if (p.y > height) p.y -= height;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(79, 209, 255, ${p.a})`;
      ctx.fill();
    }

    requestAnimationFrame(step);
  }

  // Respect users who prefer reduced motion: draw one static frame only.
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  window.addEventListener('resize', resize);
  resize();

  if (reduceMotion) {
    for (const p of particles) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(79, 209, 255, ${p.a})`;
      ctx.fill();
    }
  } else {
    requestAnimationFrame(step);
  }
})();
