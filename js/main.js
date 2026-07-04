/* Mobile nav: hamburger toggles the dropdown; any link click closes it. */
(function () {
  const nav = document.querySelector('.nav');
  const toggle = document.querySelector('.nav-toggle');
  if (!nav || !toggle) return;
  toggle.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(open));
  });
  nav.querySelectorAll('.nav-links a').forEach((a) => {
    a.addEventListener('click', () => {
      nav.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
    });
  });
})();

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

/* The background visualization lives in js/bg-drift.js. */
