/* Shared nav injected into every page. */
(function () {
  const path = location.pathname;
  const links = [
    { href: '/',          label: 'Dashboard' },
    { href: '/chat',      label: 'Chat' },
    { href: '/graph-chat', label: 'Graph' },
    { href: '/documents', label: 'Documents' },
    { href: '/eval',      label: 'Eval' },
    { href: '/docs',      label: 'API' },
  ];

  const nav = document.createElement('nav');
  nav.innerHTML = `
    <a class="brand" href="/">🤖 AgentOS</a>
    ${links.map(l => {
      const active = l.href === path || (l.href === '/' && path === '/');
      return `<a href="${l.href}" class="${active ? 'active' : ''}">${l.label}</a>`;
    }).join('')}
    <span class="spacer"></span>
    <span class="badge" id="nav-model">…</span>
  `;
  document.body.prepend(nav);

  fetch('/api/stats').then(r => r.json()).then(s => {
    const el = document.getElementById('nav-model');
    if (el) el.textContent = `${s.provider} · ${s.model}`;
  }).catch(() => {});
})();
