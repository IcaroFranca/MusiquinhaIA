import WaveSurfer from "https://unpkg.com/wavesurfer.js@7.8.6/dist/wavesurfer.esm.js";
import RegionsPlugin from "https://unpkg.com/wavesurfer.js@7.8.6/dist/plugins/regions.esm.js";

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (s) => { s = Math.max(0, Math.round(s || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
const api = async (url, opt) => {
  const r = await fetch(url, opt);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return (r.headers.get("content-type") || "").includes("json") ? r.json() : r.text();
};
const toast = (msg) => { const t = $("#toast"); t.textContent = msg; t.classList.add("on"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("on"), 2600); };

const STEM = { vocals: "voz", drums: "bateria", bass: "baixo", other: "outros" };
const SEC_COLORS = ["#7c6cff", "#2fd3b4", "#ffb547", "#ff6fa8", "#4fa3ff", "#b6e35a", "#ff8a4c", "#c28bff"];
const colorOf = (label) => SEC_COLORS[(label.charCodeAt(0) - 65) % SEC_COLORS.length];
const CAMELOT = [];
for (let n = 1; n <= 12; n++) for (const l of ["A", "B"]) CAMELOT.push(`${n}${l}`);
const CAM_NAMES = { "1A": "G#m", "2A": "D#m", "3A": "A#m", "4A": "Fm", "5A": "Cm", "6A": "Gm", "7A": "Dm", "8A": "Am", "9A": "Em", "10A": "Bm", "11A": "F#m", "12A": "C#m",
  "1B": "B", "2B": "F#", "3B": "C#", "4B": "G#", "5B": "D#", "6B": "A#", "7B": "F", "8B": "C", "9B": "G", "10B": "D", "11B": "A", "12B": "E" };
const neighbors = (c) => { const n = +c.slice(0, -1), l = c.slice(-1); return [c, `${n % 12 + 1}${l}`, `${(n + 10) % 12 + 1}${l}`, `${n}${l === "A" ? "B" : "A"}`]; };

const state = { sort: "added", desc: true, sel: null, track: null, ws: null, lastDone: -1, camelotFilter: "" };

// ---------------------------------------------------------------- biblioteca
const keySel = $("#key");
CAMELOT.forEach((c) => keySel.insertAdjacentHTML("beforeend", `<option value="${c}">${c} · ${CAM_NAMES[c]}</option>`));

async function loadTracks() {
  const p = new URLSearchParams({ sort: state.sort, desc: state.desc });
  if ($("#q").value) p.set("q", $("#q").value);
  if ($("#bmin").value) p.set("bpm_min", $("#bmin").value);
  if ($("#bmax").value) p.set("bpm_max", $("#bmax").value);
  if ($("#hd").checked) p.set("half_double", "true");
  const cam = state.camelotFilter || keySel.value;
  if (cam) p.set("camelot", cam);
  const data = await api(`/api/tracks?${p}`);
  $("#tbl tbody").innerHTML = data.items.map((t) => `
    <tr data-id="${t.id}" class="${t.status} ${t.id === state.sel ? "sel" : ""}">
      <td class="t">${esc(t.title || t.filename)}<div class="a">${t.status === "done" ? esc(t.artist || "") :
        t.status === "error" ? "erro: " + esc(t.error) : t.status === "processing" ? "analisando…" : "na fila"}</div></td>
      <td>${t.bpm ? t.bpm.toFixed(1) : "–"}</td>
      <td>${t.camelot ? `<span class="tag">${t.camelot}</span> ${esc(t.key)}` : "–"}</td>
      <td>${t.lufs != null ? t.lufs.toFixed(1) : "–"}</td>
      <td>${t.energy != null ? `<span class="ebar"><i style="width:${t.energy * 100}%"></i></span>` : "–"}</td>
      <td>${t.duration ? fmt(t.duration) : "–"}</td>
      <td class="rm"><button class="rmbtn" data-rm="${t.id}" title="Remover da biblioteca (o arquivo não é apagado)">🗑</button></td>
    </tr>`).join("");
  $("#libfoot").innerHTML = `<span>${data.items.length} de ${data.total} faixas${state.camelotFilter ? ` · tons ${state.camelotFilter}` : ""}</span>
    <button class="ghost small" id="upd" title="Reanalisa faixas feitas por versões antigas do analisador">Atualizar análises antigas</button>`;
  $("#upd").onclick = async () => { const r = await api("/api/reanalyze-all", { method: "POST" }); toast(`${r.queued} faixa(s) na fila`); loadTracks(); };
  document.querySelectorAll("th").forEach((th) => th.classList.toggle("on", th.dataset.s === state.sort));
}

async function removeTrack(tid, title) {
  if (!confirm(`Remover "${title}" da biblioteca?\n\nSó sai da lista e apaga a análise. O arquivo de música continua no seu disco.`)) return;
  await api(`/api/tracks/${tid}`, { method: "DELETE" });
  if (state.sel === tid) {
    if (state.ws) { state.ws.destroy(); state.ws = null; }
    state.sel = null; state.track = null;
    $("#detail").innerHTML = `<div class="empty"><p>Faixa removida da biblioteca.</p></div>`;
  }
  toast("Removida da biblioteca"); loadTracks();
}

$("#tbl tbody").addEventListener("click", (e) => {
  const rm = e.target.closest("[data-rm]");
  if (rm) { e.stopPropagation(); const tr = rm.closest("tr"); return removeTrack(+rm.dataset.rm, tr.querySelector("td.t").firstChild.textContent.trim()); }
  const tr = e.target.closest("tr"); if (tr) openTrack(+tr.dataset.id);
});
document.querySelectorAll("th").forEach((th) => th.addEventListener("click", () => {
  state.desc = state.sort === th.dataset.s ? !state.desc : th.dataset.s !== "title";
  state.sort = th.dataset.s; loadTracks();
}));
let deb;
["#q", "#bmin", "#bmax"].forEach((s) => $(s).addEventListener("input", () => { clearTimeout(deb); deb = setTimeout(loadTracks, 250); }));
$("#hd").addEventListener("change", loadTracks);
keySel.addEventListener("change", () => { state.camelotFilter = ""; loadTracks(); });
$("#compat").addEventListener("click", () => {
  const c = state.track?.camelot; if (!c) return toast("Abra uma faixa analisada primeiro");
  state.camelotFilter = neighbors(c).join(","); keySel.value = "";
  const b = state.track.bpm; $("#bmin").value = Math.round(b * 0.94); $("#bmax").value = Math.round(b * 1.06);
  loadTracks();
});
$("#clear").addEventListener("click", () => {
  ["#q", "#bmin", "#bmax"].forEach((s) => ($(s).value = "")); $("#hd").checked = false; keySel.value = ""; state.camelotFilter = ""; loadTracks();
});

// ---------------------------------------------------------------- adicionar
$("#pick").addEventListener("click", async () => {
  const r = await api("/api/pick-folder", { method: "POST" });
  if (r.folder) { $("#folder").value = r.folder; scan(); }
});
async function scan() {
  const folder = $("#folder").value.trim(); if (!folder) return toast("Informe a pasta");
  try { const r = await api("/api/scan", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ folder }) });
    toast(`${r.found} arquivos encontrados, ${r.queued} na fila`); loadTracks(); } catch (e) { toast("Erro: " + e.message); }
}
$("#scan").addEventListener("click", scan);
$("#folder").addEventListener("keydown", (e) => e.key === "Enter" && scan());
async function upload(files) {
  const fd = new FormData(); [...files].forEach((f) => fd.append("files", f));
  toast("Enviando…"); const r = await api("/api/upload", { method: "POST", body: fd }); toast(`${r.queued} arquivo(s) na fila`); loadTracks();
}
$("#files").addEventListener("change", (e) => upload(e.target.files));
let dragN = 0;
addEventListener("dragenter", (e) => { e.preventDefault(); dragN++; $("#drop").classList.add("on"); });
addEventListener("dragleave", () => { if (--dragN <= 0) $("#drop").classList.remove("on"); });
addEventListener("dragover", (e) => e.preventDefault());
addEventListener("drop", (e) => { e.preventDefault(); dragN = 0; $("#drop").classList.remove("on"); if (e.dataTransfer.files.length) upload(e.dataTransfer.files); });

// ---------------------------------------------------------------- status
async function pollStatus() {
  try {
    const s = await api("/api/status"); const c = s.counts; const done = c.done || 0;
    const left = (c.pending || 0) + (c.processing || 0);
    $("#status").innerHTML = `
      <span class="pill gpu" title="${esc(s.beat_backend)}">${s.gpu.cuda ? "GPU " + esc(s.gpu.device.replace("NVIDIA GeForce ", "")) : "CPU"}</span>
      <span><b>${done}</b>/${s.total} analisadas${c.error ? ` · <span class="err">${c.error} erro(s)</span>` : ""}</span>
      ${left ? `<span class="bar"><i style="width:${(done / Math.max(1, s.total)) * 100}%"></i></span><span>${left} na fila</span>` : ""}`;
    if (done !== state.lastDone || left) { state.lastDone = done; loadTracks(); }
    if (state.sel && state.track && state.track.status !== "done") { const t = await api(`/api/tracks/${state.sel}`); if (t.status === "done") openTrack(state.sel); }
  } catch {}
  setTimeout(pollStatus, 2000);
}

// ---------------------------------------------------------------- detalhe
async function openTrack(id) {
  state.sel = id;
  document.querySelectorAll("tbody tr").forEach((tr) => tr.classList.toggle("sel", +tr.dataset.id === id));
  const t = await api(`/api/tracks/${id}`); state.track = t;
  if (state.ws) { state.ws.destroy(); state.ws = null; }
  const D = $("#detail");
  if (t.status !== "done") {
    D.innerHTML = `<div class="empty"><h1>${esc(t.title || t.filename)}</h1>
      <p class="${t.status === "error" ? "err" : "muted"}">${t.status === "error" ? "Erro: " + esc(t.error) : t.status === "processing" ? "Analisando…" : "Na fila de análise"}</p>
      ${t.status === "error" ? `<p><button id="re">Tentar de novo</button></p>` : ""}</div>`;
    $("#re")?.addEventListener("click", () => reanalyze(id));
    return;
  }
  const a = t.analysis, k = a.key, L = a.loudness, tp = a.tempo, m = a.meta, ins = a.instrumentation, h = a.harmony;
  D.innerHTML = `
    <div class="dhead">
      <div><h1>${esc(m.title)}</h1><div class="sub">${esc([m.artist, m.album, m.year].filter(Boolean).join(" · ") || t.filename)}</div></div>
      <div class="actions"><button class="small" id="goclip">🎬 Criar videoclipe</button>
        <button class="ghost small" id="re">Reanalisar</button><button class="ghost small danger" id="del">🗑 Remover da biblioteca</button></div>
    </div>
    <div class="player"><div id="wave"></div>
      <div class="pctl"><button id="play">▶</button><span id="pos">0:00</span> / ${fmt(a.duration)}
        <span style="margin-left:auto">batidas: ${esc(tp.backend)} · análise em ${a.analysis_seconds}s</span></div></div>
    <div class="cards">
      <div class="card"><div class="k">BPM</div><div class="v">${tp.bpm}</div><div class="s">${tp.time_signature} · ±${tp.bpm_std}</div></div>
      <div class="card"><div class="k">Tom</div><div class="v">${esc(k.key_short)} <span class="tag">${k.camelot}</span></div><div class="s">${esc(k.key_pt)} · ${Math.round(k.confidence * 100)}% (alt. ${esc(k.alternative)})</div></div>
      <div class="card"><div class="k">Loudness</div><div class="v">${L.lufs} <small>LUFS</small></div><div class="s">LRA ${L.lra} LU · pico ${L.peak_dbfs} dBFS</div></div>
      <div class="card"><div class="k">Energia</div><div class="v">${Math.round(a.energy_score * 100)}</div><div class="s">${a.sections.length} seções</div></div>
      <div class="card"><div class="k">Brilho</div><div class="v">${a.timbre.centroid_hz} <small>Hz</small></div><div class="s">${esc(a.descriptors.brilho)}</div></div>
      <div class="card"><div class="k">Ritmo</div><div class="v">${a.timbre.onset_density}</div><div class="s">ataques/s · ${Math.round(a.timbre.percussive_ratio * 100)}% percussivo</div></div>
    </div>
    <div class="desc">${Object.entries(a.descriptors).map(([k, v]) => `<span><span class="muted">${k.replace(/_/g, " ")}:</span> ${esc(v)}</span>`).join("")}</div>
    ${ins ? `<div class="aitags">
      ${ins.genres.length ? `<div><span class="lbl">Gênero (IA)</span>${ins.genres.map((d) => `<span class="chip g">${esc(d.label)} <b>${Math.round(d.score * 100)}%</b></span>`).join("")}</div>` : ""}
      ${ins.moods.length ? `<div><span class="lbl">Clima</span>${ins.moods.map((d) => `<span class="chip m">${esc(d.label)}</span>`).join("")}</div>` : ""}
      <div><span class="lbl">Sons</span>${ins.has_vocals ? `<span class="chip v">🎤 voz</span>` : `<span class="chip">instrumental</span>`}${ins.instruments.map((d) => `<span class="chip">${esc(d.label)} <b>${Math.round(d.score * 100)}%</b></span>`).join("")}</div>
    </div>` : `<p class="muted">Análise antiga, sem instrumentos, acordes e letra. Clique em <b>Reanalisar</b>.</p>`}

    <h2 id="cliphead">🎬 Criar videoclipe</h2>
    <div class="box clip" id="clipbox">
      <div class="brief" id="brief">
        <div class="bgroup"><span class="blbl">Tipo de clipe</span><div class="chips" id="b-tipo"></div></div>
        <div class="bgroup"><span class="blbl">Técnica</span><div class="chips" id="b-tecnica"></div></div>
        <div class="brow">
          <label>Formato<select id="b-formato"></select></label>
          <label>FPS<select id="b-fps"></select></label>
          <label class="chk"><input type="checkbox" id="b-so_motion"> só motion graphics (sem filmagem)</label>
          ${a.lyrics ? `<label class="chk"><input type="checkbox" id="b-letra_na_tela"> letra na tela (karaokê)</label>` : ""}
        </div>
      </div>
      <textarea id="extra" rows="2" placeholder="Sua ideia (opcional): ex. 'estética de papel recortado, uma ascensorista num hotel art déco, cada andar é uma festa, final igual à abertura'"></textarea>
      <div class="genrow" id="genrow">
        <label>Modelo<select id="g-model"></select></label>
        <label>Esforço<select id="g-effort"></select></label>
        <button id="genclip" class="big" title="O Claude Code cria o clipe sozinho a partir da análise e renderiza o MP4">🎬 Gerar clipe com Claude</button>
        <span id="claudeinfo" class="muted"></span>
      </div>
      <div class="row subtle">
        <span class="muted">Ou faça você mesmo no Claude:</span>
        <button class="ghost small" id="exp" title="Cria uma pasta com o áudio, PROMPT.md e timing.json, pronta para abrir no Claude">📦 Exportar pacote</button>
        <button class="ghost small" id="showclip">Ver prompt do clipe</button>
        <a class="btn ghost small" href="/api/tracks/${id}/timing.json" download title="Tempos de batidas, compassos, seções, impactos, bateria, envelopes e letra palavra a palavra">timing.json</a>
      </div>
    </div>
    <div id="jobwrap" style="display:none"><h2>Clipe gerado pelo Claude</h2><div class="box job" id="job"></div></div>

    ${h?.progression ? `<h2>Harmonia</h2><div class="box prog">
      <div class="chords">${(h.progression.rle || h.progression.chords.map((c) => [c, 1])).map(([c, n]) => `<span>${esc(c)}${n > 1 ? `<small> ×${n}</small>` : ""}</span>`).join("")}</div>
      <div class="muted">progressão principal · ${h.progression.repeats}× · mais usados: ${h.most_used.map(esc).join(", ")}</div></div>` : ""}

    <h2>Curva de energia</h2><div class="box"><canvas id="energy" height="110"></canvas></div>

    <div class="grid2">
      <div><h2>Estrutura</h2><div class="box secs">${a.sections.map((s, i) => `
        <div class="sec" data-i="${i}"><span class="time">${fmt(s.start)}–${fmt(s.end)}</span>
          <span class="nm"><span class="sw" style="background:${colorOf(s.label)}"></span>${esc(s.name)} <span class="muted">${s.label}</span></span>
          <span class="muted">${s.bars} comp.</span><span>${esc(s.energy_level)}</span><span class="tag">${esc(s.local_key)}</span>
          ${s.stems || s.chords ? `<div class="sec2">${Object.entries(s.stems || {}).map(([k, v]) => `<span class="st ${({ "em destaque": "forte", "ao fundo": "fraca" })[v] || v}" title="${STEM[k]}: ${v}">${STEM[k]}</span>`).join("")}
            <span class="ch">${esc(s.chords || "")}</span></div>` : ""}</div>`).join("")}</div></div>
      <div><h2>Espectro</h2><div class="box bands">${Object.entries(a.timbre.bands).map(([b, v]) => `
        <div class="b"><span>${esc(b)}</span><span class="track"><i style="width:${Math.min(100, v * 100 / 0.5)}%"></i></span><span>${Math.round(v * 100)}%</span></div>`).join("")}
        <div class="b"><span>Imagem estéreo</span><span class="track"><i style="width:${Math.min(100, L.stereo_width * 160)}%;background:var(--accent2)"></i></span><span>${L.stereo_width}</span></div>
        <div class="b"><span>Fator de crista</span><span class="track"><i style="width:${Math.min(100, L.crest_db * 5)}%;background:var(--warn)"></i></span><span>${L.crest_db}</span></div>
      </div></div>
    </div>

    ${a.lyrics ? `<h2 class="lyrhead">Letra <span class="muted" id="lyrsrc" style="text-transform:none"></span>
        <span class="lyracts"><button class="ghost small" id="lyredit">Corrigir letra</button></span></h2>
      <div class="box lyr" id="lyrbox"></div>` : ""}

    <h2 id="prompthead">Prompt para o Claude</h2>
    <div class="box prompt">
      <div class="row"><select id="goal"></select>
        ${a.lyrics ? `<label class="chk"><input type="checkbox" id="withlyr" checked> incluir letra no prompt</label>` : ""}
        <button id="gen">Gerar prompt</button><button class="ghost" id="copy">Copiar</button></div>
      <p class="muted small-note">O objetivo "Videoclipe sincronizado" usa as escolhas do painel 🎬 Criar videoclipe.</p>
      <textarea id="ptext" readonly></textarea>
    </div>

    <h2>Músicas parecidas na biblioteca</h2><div class="box sim" id="sim"><span class="muted">Calculando…</span></div>`;

  $("#re").addEventListener("click", () => reanalyze(id));
  $("#del").addEventListener("click", () => removeTrack(id, m.title || t.filename));

  // player
  const regions = RegionsPlugin.create();
  const ws = WaveSurfer.create({ container: "#wave", height: 110, waveColor: "#3a3f52", progressColor: "#7c6cff", cursorColor: "#e8e9ee",
    barWidth: 2, barGap: 1, barRadius: 2, url: `/api/tracks/${id}/audio`, peaks: [a.peaks], duration: a.duration, plugins: [regions] });
  state.ws = ws;
  ws.on("ready", () => a.sections.forEach((s) => regions.addRegion({ start: s.start, end: s.end, color: colorOf(s.label) + "22", drag: false, resize: false, content: s.name })));
  ws.on("timeupdate", (t) => {
    $("#pos").textContent = fmt(t);
    if (!a.lyrics) return;
    document.querySelectorAll(".lyr [data-i]").forEach((el) => {
      const l = a.lyrics.lines[+el.dataset.i]; if (l) el.classList.toggle("now", t >= l.start && t < l.end + 0.5); });
  });
  if (a.lyrics) renderLyrics(id, a, ws, false);
  ws.on("play", () => ($("#play").textContent = "❚❚")); ws.on("pause", () => ($("#play").textContent = "▶"));
  $("#play").addEventListener("click", () => ws.playPause());
  document.querySelectorAll(".sec").forEach((el) => el.addEventListener("click", () => { ws.setTime(a.sections[+el.dataset.i].start); ws.play(); }));

  drawEnergy(a);

  // prompt
  const [goals, opts] = await Promise.all([api("/api/goals"), api("/api/brief-options")]);
  $("#goal").innerHTML = Object.entries(goals).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
  const opt = (sel, entries) => ($(sel).innerHTML = entries.map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join(""));
  // tipo e técnica como botões (escolha única), para ficarem visíveis
  const SHORT = {
    tipo: { historia: "📖 História", danca: "💃 Dança", abstrato: "✨ Abstrato", lyric: "🔤 Lyric video", livre: "🎲 Livre" },
    tecnica: { livre: "Livre", web: "Canvas + three.js + SVG", hyperframes: "HyperFrames", remotion: "Remotion", ae: "After Effects" },
  };
  const chips = (group, entries) => ($(`#b-${group}`).innerHTML = entries.map(([k, v]) =>
    `<button type="button" class="chipbtn" data-v="${k}" title="${esc(v)}">${esc(SHORT[group][k] || v)}</button>`).join(""));
  chips("tipo", Object.entries(opts.tipo)); chips("tecnica", Object.entries(opts.tecnica));
  opt("#b-formato", opts.formato.map((f) => [f, f])); opt("#b-fps", opts.fps.map((f) => [f, `${f} fps`]));
  const setChip = (sel, v) => $(sel).querySelectorAll(".chipbtn").forEach((b) => b.classList.toggle("on", b.dataset.v === v));
  const getChip = (sel) => $(sel).querySelector(".chipbtn.on")?.dataset.v;
  let saved = {}; try { saved = JSON.parse(localStorage.getItem("brief") || "{}"); } catch {}
  const brief0 = { ...opts.default, ...saved };
  setChip("#b-tipo", brief0.tipo); setChip("#b-tecnica", brief0.tecnica);
  ["formato", "fps"].forEach((k) => ($(`#b-${k}`).value = brief0[k]));
  ["so_motion", "letra_na_tela"].forEach((k) => { const el = $(`#b-${k}`); if (el) el.checked = !!brief0[k]; });
  const readBrief = () => {
    const b = { tipo: getChip("#b-tipo") || "livre", tecnica: getChip("#b-tecnica") || "livre", formato: $("#b-formato").value,
      fps: +$("#b-fps").value, so_motion: $("#b-so_motion").checked, letra_na_tela: $("#b-letra_na_tela")?.checked ?? false };
    try { localStorage.setItem("brief", JSON.stringify(b)); } catch {}
    return b;
  };
  const body = (goal) => JSON.stringify({ goal: goal || $("#goal").value, extra: $("#extra").value.trim() || null,
    lyrics: $("#withlyr") ? $("#withlyr").checked : true, brief: readBrief() });
  const gen = async () => {
    $("#ptext").value = await api(`/api/tracks/${id}/prompt`, { method: "POST", headers: { "content-type": "application/json" }, body: body() });
  };
  $("#gen").addEventListener("click", gen); $("#goal").addEventListener("change", gen); $("#withlyr")?.addEventListener("change", gen);
  ["#b-tipo", "#b-tecnica"].forEach((sel) => $(sel).addEventListener("click", (e) => {
    const b = e.target.closest(".chipbtn"); if (!b) return; setChip(sel, b.dataset.v); readBrief();
    if ($("#goal").value === "video") gen();
  }));
  $("#brief").addEventListener("change", () => { readBrief(); if ($("#goal").value === "video") gen(); });
  $("#showclip").addEventListener("click", async () => {
    $("#goal").value = "video"; await gen(); $("#prompthead").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  $("#goclip").addEventListener("click", () => $("#cliphead").scrollIntoView({ behavior: "smooth", block: "start" }));
  $("#copy").addEventListener("click", async () => { if (!$("#ptext").value) await gen(); await navigator.clipboard.writeText($("#ptext").value); toast("Prompt copiado. Cole no Claude."); });
  $("#exp").addEventListener("click", async () => {
    try { const r = await api(`/api/tracks/${id}/export`, { method: "POST", headers: { "content-type": "application/json" }, body: body("video") });
      toast(`Pacote criado: ${r.folder}`); } catch (e) { toast("Erro ao exportar: " + e.message); }
  });
  gen();
  setupClaude(id, body);
  const existing = await api(`/api/tracks/${id}/job`);
  if (existing && state.sel === id) watchJob(id, existing);

  // parecidas
  const sim = await api(`/api/tracks/${id}/similar`);
  $("#sim").innerHTML = sim.length ? sim.map((s) => `<div data-id="${s.id}"><span>${esc(s.title)} <span class="muted">${esc(s.artist || "")}</span></span>
      <span>${s.bpm?.toFixed(1)} BPM</span><span class="tag ${s.key_compatible ? "ok" : ""}" title="${s.key_compatible ? "tom compatível" : ""}">${s.camelot}</span>
      <span class="muted">${Math.round((s.similarity + 1) * 50)}%</span></div>`).join("")
    : `<span class="muted">Analise mais faixas para comparar.</span>`;
  $("#sim").querySelectorAll("[data-id]").forEach((el) => el.addEventListener("click", () => openTrack(+el.dataset.id)));
}

// ---------------------------------------------------------------- geração automática (Claude Code)
async function setupClaude(id, body) {
  let st;
  try { st = await api("/api/claude/status"); } catch { return; }
  if (state.sel !== id) return;
  const opt = (sel, entries) => ($(sel).innerHTML = entries.map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join(""));
  opt("#g-model", Object.entries(st.models)); opt("#g-effort", st.efforts.map((e) => [e, e]));
  let pref = {}; try { pref = JSON.parse(localStorage.getItem("gen") || "{}"); } catch {}
  $("#g-model").value = pref.model || "opus"; $("#g-effort").value = pref.effort || "low";
  const info = $("#claudeinfo"), btn = $("#genclip");
  if (!st.installed || !st.loggedIn) {
    btn.disabled = true;
    info.innerHTML = !st.installed
      ? `Claude Code não encontrado.`
      : `Falta entrar na sua conta do Claude: rode <code>"${esc(st.bin)}" auth login</code> num terminal e depois <a href="#" id="recheck">verifique de novo</a>.`;
    $("#recheck")?.addEventListener("click", (e) => { e.preventDefault(); setupClaude(id, body); });
    return;
  }
  btn.disabled = false;
  info.textContent = "Pode levar de 10 a 40 min. Usa a sua assinatura do Claude.";
  btn.onclick = async () => {
    const model = $("#g-model").value, effort = $("#g-effort").value;
    try { localStorage.setItem("gen", JSON.stringify({ model, effort })); } catch {}
    const b = JSON.parse(body()); b.goal = "video";
    try {
      const job = await api(`/api/tracks/${id}/generate`, { method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ ...b, model, effort }) });
      toast("Claude começou a criar o clipe"); watchJob(id, job);
      $("#jobwrap").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) { toast("Erro: " + e.message); }
  };
}

const JOB_LABEL = { queued: "na fila", running: "criando…", done: "pronto", error: "falhou", cancelled: "cancelado" };
const KIND_ICON = { tool: "›", text: "✎", warn: "!", info: "•", out: "·" };

function watchJob(id, job) {
  const wrap = $("#jobwrap"), box = $("#job");
  if (!wrap || !box) return;
  wrap.style.display = "";
  box.innerHTML = `<div class="jobhead"><b id="jstatus"></b><span id="jmeta" class="muted"></span>
      <span class="jacts"><button class="ghost small" id="jopen">Abrir pasta</button><button class="ghost small" id="jcancel">Cancelar</button></span></div>
    <div id="jvideo"></div><div class="joblog" id="joblog"></div>`;
  $("#jopen").onclick = () => api(`/api/jobs/${job.id}/open`, { method: "POST" });
  $("#jcancel").onclick = async () => { if (confirm("Parar a geração do clipe?")) await api(`/api/jobs/${job.id}/cancel`, { method: "POST" }); };
  let seen = 0;
  const render = (j) => {
    const lg = $("#joblog"); if (!lg) return false;
    const atBottom = lg.scrollHeight - lg.scrollTop - lg.clientHeight < 40;
    j.log.forEach((e) => lg.insertAdjacentHTML("beforeend",
      `<div class="jl ${e.kind}"><span class="jt">${fmt(e.t)}</span><span class="ji">${KIND_ICON[e.kind] || "·"}</span><span>${esc(e.text)}</span></div>`));
    seen = j.log_total;
    if (atBottom) lg.scrollTop = lg.scrollHeight;
    $("#jstatus").textContent = `${JOB_LABEL[j.status] || j.status}`;
    $("#jstatus").className = j.status;
    $("#jmeta").textContent = ` · ${j.model}/${j.effort} · ${fmt(j.elapsed)}` + (j.turns ? ` · ${j.turns} passos` : "")
      + (j.cost_usd != null ? ` · custo equivalente ~US$ ${j.cost_usd.toFixed(2)}` : "");
    $("#jcancel").style.display = ["queued", "running"].includes(j.status) ? "" : "none";
    if (j.status === "done" && j.video && !$("#jvideo").innerHTML)
      $("#jvideo").innerHTML = `<video controls preload="metadata" src="/api/jobs/${j.id}/video"></video>`;
    if (j.status === "error" && j.error && !$("#jvideo").innerHTML)
      $("#jvideo").innerHTML = `<p class="err">${esc(j.error)}</p>`;
    return ["queued", "running"].includes(j.status);
  };
  const tick = async () => {
    if (state.sel !== id || !document.body.contains(box)) return;
    let j; try { j = await api(`/api/jobs/${job.id}?since=${seen}`); } catch { return setTimeout(tick, 4000); }
    if (render(j)) setTimeout(tick, 2000);
  };
  render({ ...job, log: job.log || [] }) ? setTimeout(tick, 1500) : null;
}

// ---------------------------------------------------------------- letra (ver / corrigir)
function renderLyrics(id, a, ws, edit) {
  const L = a.lyrics, box = $("#lyrbox");
  $("#lyrsrc").textContent = L.source === "colada" ? "(colada por você · sincronizada pelo áudio)"
    : L.edited ? "(corrigida por você)" : `(Whisper · ${L.language || "?"} · pode conter erros)`;
  $(".lyracts").innerHTML = edit
    ? `<button class="small" id="lyrsave">${edit === "paste" ? "Sincronizar e salvar" : "Salvar"}</button><button class="ghost small" id="lyrcancel">Cancelar</button>`
    : `<button class="ghost small" id="lyrpaste" title="Cole a letra oficial (ex.: do Genius); o site encaixa cada palavra no tempo em que é cantada">📋 Colar letra correta</button>
       <button class="ghost small" id="lyredit">Corrigir linha a linha</button>${L.edited ? `<button class="ghost small" id="lyrrestore">Restaurar original</button>` : ""}`;
  if (!edit) {
    let part = null;
    box.innerHTML = L.lines.map((l, i) => {
      const tag = l.part && l.part !== part ? `<div class="lpart">${esc(l.part)}</div>` : "";
      part = l.part || part;
      return `${tag}<div data-i="${i}"><span class="time">${fmt(l.start)}</span>${esc(l.text)}</div>`;
    }).join("") || `<span class="muted">Sem letra.</span>`;
    box.querySelectorAll("[data-i]").forEach((el) => el.addEventListener("click", () => { ws.setTime(L.lines[+el.dataset.i].start); ws.play(); }));
    $("#lyredit").onclick = () => renderLyrics(id, a, ws, true);
    $("#lyrpaste").onclick = () => renderLyrics(id, a, ws, "paste");
    if ($("#lyrrestore")) $("#lyrrestore").onclick = async () => {
      if (!confirm("Voltar para a transcrição automática? Suas correções serão descartadas.")) return;
      a.lyrics = await api(`/api/tracks/${id}/lyrics`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ restore: true }) });
      toast("Transcrição original restaurada"); openTrack(id);
    };
    return;
  }
  if (edit === "paste") {
    box.innerHTML = `<p class="muted lyrhelp">Cole a letra completa e correta. Cabeçalhos como [Verse 1], [Chorus] e [Bridge] viram os nomes das seções, e o lixo de site ("You might also like"…) é ignorado.
      Os tempos vêm do que o Whisper ouviu: cada palavra que bate herda o tempo exato, e o resto é encaixado onde há voz.</p>
      <textarea id="lyrpastebox" class="lyrpaste" spellcheck="false" placeholder="[Verse 1]
Primeira linha da letra…"></textarea>`;
    $("#lyrpastebox").focus();
    $("#lyrcancel").onclick = () => renderLyrics(id, a, ws, false);
    $("#lyrsave").onclick = async () => {
      const text = $("#lyrpastebox").value;
      if (!text.trim()) return toast("Cole a letra primeiro");
      if (L.edited && !confirm("Substituir a letra atual pela colada?")) return;
      try {
        await api(`/api/tracks/${id}/lyrics/paste`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text }) });
      } catch (e) { return toast(e.message || "Não consegui alinhar a letra"); }
      toast("Letra sincronizada. Prompt, seções e timing.json já usam a versão correta.");
      openTrack(id);
    };
    return;
  }
  box.innerHTML = `<p class="muted lyrhelp">Corrija o texto de cada linha. Se mantiver o mesmo número de palavras, o tempo de cada palavra é preservado; senão, é redistribuído na linha. Apague o texto para remover a linha. ▶ toca o trecho.</p>`
    + L.lines.map((l, i) => `<div class="ledit" data-row="${i}"><button class="ghost small lplay" title="Ouvir">▶</button>
      <span class="time">${fmt(l.start)}</span><input value="${esc(l.text)}" spellcheck="true"><button class="ghost small ldel" title="Remover linha">✕</button></div>`).join("");
  box.querySelectorAll(".ledit").forEach((row) => {
    const l = L.lines[+row.dataset.row];
    row.querySelector(".lplay").onclick = () => { ws.setTime(l.start); ws.play(); setTimeout(() => ws.getCurrentTime() >= l.end && ws.pause(), (l.end - l.start + 0.3) * 1000); };
    row.querySelector(".ldel").onclick = () => { row.querySelector("input").value = ""; row.classList.add("gone"); };
  });
  $("#lyrcancel").onclick = () => renderLyrics(id, a, ws, false);
  $("#lyrsave").onclick = async () => {
    const lines = [...box.querySelectorAll(".ledit")].map((row) => {
      const i = +row.dataset.row, l = L.lines[i];
      return { i, start: l.start, end: l.end, text: row.querySelector("input").value };
    });
    a.lyrics = await api(`/api/tracks/${id}/lyrics`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ lines }) });
    renderLyrics(id, a, ws, false); toast("Letra salva. O prompt e o timing.json já usam a versão corrigida.");
    $("#gen")?.click();
  };
}

function drawEnergy(a) {
  const cv = $("#energy"), dpr = devicePixelRatio || 1, W = cv.clientWidth, H = 110;
  cv.width = W * dpr; cv.height = H * dpr; const g = cv.getContext("2d"); g.scale(dpr, dpr);
  const c = a.energy_curve, x = (t) => (t / a.duration) * W;
  a.sections.forEach((s) => { g.fillStyle = colorOf(s.label) + "26"; g.fillRect(x(s.start), 0, x(s.end) - x(s.start), H);
    g.fillStyle = "#8b90a0"; g.font = "11px Inter"; g.fillText(s.name, x(s.start) + 4, 13); });
  const sm = c.map((_, i) => { let s = 0, n = 0; for (let j = Math.max(0, i - 2); j <= Math.min(c.length - 1, i + 2); j++) { s += c[j]; n++; } return s / n; });
  const grad = g.createLinearGradient(0, 0, 0, H); grad.addColorStop(0, "#7c6cffcc"); grad.addColorStop(1, "#2fd3b411");
  g.beginPath(); g.moveTo(0, H); sm.forEach((v, i) => g.lineTo(x(i), H - v * (H - 20))); g.lineTo(W, H); g.closePath(); g.fillStyle = grad; g.fill();
  g.beginPath(); sm.forEach((v, i) => (i ? g.lineTo : g.moveTo).call(g, x(i), H - v * (H - 20))); g.strokeStyle = "#a79cff"; g.lineWidth = 1.5; g.stroke();
  const EVC = { "drop/explosão": "#ff5d6c", subida: "#ffb547", "queda/quebra": "#4fa3ff", "clímax": "#ffffff", "pausa/silêncio": "#8b90a0" };
  (a.events || []).filter((e) => EVC[e.type]).forEach((e) => {
    g.strokeStyle = EVC[e.type]; g.setLineDash([3, 3]); g.lineWidth = 1;
    g.beginPath(); g.moveTo(x(e.t), 18); g.lineTo(x(e.t), H); g.stroke(); g.setLineDash([]);
    if (e.type === "clímax" || e.type === "drop/explosão") { g.fillStyle = EVC[e.type]; g.font = "11px Inter"; g.fillText(e.type === "clímax" ? "★" : "drop", x(e.t) + 3, H - 4); }
  });
}

async function reanalyze(id) { await api(`/api/tracks/${id}/reanalyze`, { method: "POST" }); toast("Reanalisando…"); openTrack(id); loadTracks(); }

loadTracks(); pollStatus();
