"""Alinha uma letra colada pelo usuário (ex.: do Genius) aos tempos que o Whisper ouviu.

O Whisper erra palavras, mas acerta bem QUANDO a voz está cantando. Então:
1. a letra colada vira linhas (sem cabeçalhos [Verse], lixo de site etc.), guardando a parte de cada linha;
2. a letra inteira é alinhada às palavras do Whisper (Needleman-Wunsch, ordem preservada, então
   cada repetição do refrão cai no seu lugar); palavras iguais ou parecidas herdam o tempo exato;
3. o que sobra é distribuído entre as âncoras vizinhas, só nos intervalos em que há voz.
"""
from __future__ import annotations

import difflib
import re

_HEADER = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_JUNK_START = re.compile(r"^\s*you might also like\s*$", re.I)
_JUNK_LINE = re.compile(r"^\s*(\d+\s+contributors?.*|.*\blyrics\s*$|see .* live|get tickets.*)$", re.I)
_EMBED = re.compile(r"\d*\s*Embed\s*$", re.I)

PARTS_PT = [
    (r"pre[- ]?chorus|pré[- ]?refrão", "Pré-refrão"),
    (r"post[- ]?chorus|pós[- ]?refrão", "Pós-refrão"),
    (r"chorus|hook|refrão", "Refrão"),
    (r"verse|verso", "Verso"),
    (r"bridge|ponte", "Ponte"),
    (r"intro", "Intro"),
    (r"outro|final", "Final"),
    (r"interlude|interlúdio", "Interlúdio"),
    (r"breakdown", "Quebra"),
    (r"drop", "Drop"),
    (r"solo", "Solo"),
    (r"instrumental", "Instrumental"),
]


MAX_WORD = 2.0


def part_name(header: str) -> str:
    h = header.split(":")[0].strip()          # "[Chorus: Fulano]" -> "Chorus"
    low = h.lower()
    for pat, pt in PARTS_PT:
        if re.search(pat, low):
            num = re.search(r"\b(\d+)\b", h)
            return f"{pt} {num.group(1)}" if num and pt in ("Verso", "Refrão", "Pré-refrão") else pt
    return h


def parse_pasted(text: str) -> list[dict]:
    """-> [{text, part}] (part = nome em pt-BR do cabeçalho mais recente, ou None)."""
    raw = text.replace("\r", "").split("\n")
    has_headers = any(_HEADER.match(l) for l in raw)
    out, part, skipping, seen_header = [], None, False, False
    for line in raw:
        m = _HEADER.match(line)
        if m:
            part, skipping, seen_header = part_name(m.group(1)), False, True
            continue
        if _JUNK_START.match(line):
            skipping = True                      # "You might also like" + recomendações até o próximo cabeçalho
            continue
        if skipping:
            continue
        line = _EMBED.sub("", line).strip()
        if not line or (has_headers and not seen_header and _JUNK_LINE.match(line)):
            continue
        out.append({"text": line, "part": part})
    return out


def _norm(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower().replace("’", "'").replace("'", ""))


def _weights(toks: list[str]) -> list[float]:
    return [len(t) + 1.0 for t in toks]


def _spread(n_w: list[float], intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Distribui palavras (pesos) ao longo de uma lista de intervalos de tempo, sem passar de ~0,9 s/palavra."""
    total = sum(b - a for a, b in intervals)
    need = sum(n_w)
    if total <= 0 or not n_w:
        return []
    use = min(total, 0.9 * len(n_w))             # não esticar uma frase curta por um buraco enorme

    def real(v: float, start: bool) -> float:    # tempo "só de voz" -> tempo real
        for a, b in intervals:
            d = b - a
            if v < d or (not start and v <= d):
                return a + v
            v -= d
        return intervals[-1][1]

    out, acc = [], 0.0
    for w in n_w:
        v0, acc = acc / need * use, acc + w
        out.append((real(v0, True), real(acc / need * use, False)))
    return out


def _global_align(a: list[str], b: list[str], gap: float = -0.4) -> list[tuple[int, int]]:
    """Needleman-Wunsch: alinhamento monótono da letra inteira (refrões repetidos caem cada um no seu lugar).
    Devolve pares (índice na letra colada, índice no Whisper) de palavras iguais, parecidas ou trocadas."""
    n, m = len(a), len(b)
    if not n or not m:
        return []
    cache: dict[tuple[str, str], float] = {}

    def sc(x: str, y: str) -> float:
        if x == y:
            return 2.0
        k = (x, y)
        if k not in cache:
            r = difflib.SequenceMatcher(None, x, y).ratio()
            cache[k] = 1.0 if r >= 0.75 else 0.2 if r >= 0.5 else -1.0
        return cache[k]

    H = [[0.0] * (m + 1) for _ in range(n + 1)]
    P = [[0] * (m + 1) for _ in range(n + 1)]          # 0 diagonal, 1 cima (palavra só na letra), 2 esquerda
    for i in range(1, n + 1):
        H[i][0], P[i][0] = gap * i, 1
    for j in range(1, m + 1):
        H[0][j], P[0][j] = gap * j, 2
    for i in range(1, n + 1):
        Hi, Hp, Pi, ai = H[i], H[i - 1], P[i], a[i - 1]
        for j in range(1, m + 1):
            d, u, l = Hp[j - 1] + sc(ai, b[j - 1]), Hp[j] + gap, Hi[j - 1] + gap
            if d >= u and d >= l:
                Hi[j], Pi[j] = d, 0
            elif u >= l:
                Hi[j], Pi[j] = u, 1
            else:
                Hi[j], Pi[j] = l, 2
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        p = P[i][j]
        if p == 0:
            if sc(a[i - 1], b[j - 1]) >= 1.0:              # só vira âncora se a palavra é igual ou bem parecida
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif p == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def _drop_implausible(pairs: list[tuple[int, int]], ref: list[dict], min_per_word: float = 0.1):
    """Remove âncoras soltas que obrigariam a cantar mais de ~10 palavras/s no buraco ao lado
    (acontece quando uma palavra repetida, tipo 'glass', casa com a repetição errada)."""
    pairs = list(pairs)
    changed = True
    while changed and len(pairs) > 1:
        changed = False
        runs, start = [], 0                       # sequências de âncoras consecutivas na letra
        for n in range(1, len(pairs) + 1):
            if n == len(pairs) or pairs[n][0] != pairs[n - 1][0] + 1:
                runs.append((start, n))
                start = n
        for r in range(len(runs) - 1):
            (a0, a1), (b0, b1) = runs[r], runs[r + 1]
            missing = pairs[b0][0] - pairs[a1 - 1][0] - 1
            room = ref[pairs[b0][1]]["s"] - ref[pairs[a1 - 1][1]]["e"]
            if missing and room < missing * min_per_word:
                drop = runs[r + 1] if (b1 - b0) <= (a1 - a0) else runs[r]
                del pairs[drop[0]:drop[1]]
                changed = True
                break
    return pairs


def _clip(regions: list, a: float, b: float) -> list[tuple[float, float]]:
    iv = [(max(a, r0), min(b, r1)) for r0, r1 in regions if r1 > a and r0 < b]
    return [(x, y) for x, y in iv if y - x > 0.05]


def align(pasted: list[dict], ref_words: list[dict], vocal_regions: list, duration: float) -> list[dict]:
    """pasted: [{text, part}]; ref_words: [{w, s, e}] do Whisper em ordem. -> linhas {start,end,text,part,words}."""
    toks, owner = [], []
    for li, l in enumerate(pasted):
        for t in l["text"].split():
            toks.append(t)
            owner.append(li)
    times: list[tuple[float, float] | None] = [None] * len(toks)
    ref = [w for w in ref_words if w.get("w")]
    a_n, b_n = [_norm(t) for t in toks], [_norm(w["w"]) for w in ref]

    pairs = _drop_implausible(_global_align(a_n, b_n), ref)
    used = {j for _, j in pairs}
    for i, j in pairs:
        s, e = ref[j]["s"], ref[j]["e"]
        times[i] = (s, min(e, s + MAX_WORD))       # Whisper às vezes estica uma palavra por vários segundos
    # palavras que o Whisper ouviu mas não casaram: mostram onde há canto de verdade
    heard = [(w["s"], min(w["e"], w["s"] + 4 * MAX_WORD)) for j, w in enumerate(ref) if j not in used]

    # buracos: palavras sem tempo entre duas âncoras -> espalha onde o Whisper ouviu algo, senão onde há voz
    regions = [tuple(r) for r in (vocal_regions or [])] or [(0.0, duration)]
    k = 0
    while k < len(toks):
        if times[k] is not None:
            k += 1
            continue
        j = k
        while j < len(toks) and times[j] is None:
            j += 1
        g0 = times[k - 1][1] if k > 0 else 0.0
        g1 = times[j][0] if j < len(toks) else max(max(r[1] for r in regions), g0 + 1.0)
        need = 0.25 * (j - k)
        iv = _clip(heard, g0, g1)
        big = [x for x in iv if x[1] - x[0] >= 1.0]       # descarta "Oh!" soltos se o resto já basta
        if sum(b - a for a, b in big) >= need:
            iv = big
        if sum(b - a for a, b in iv) < need:
            iv = _clip(regions, g0, g1)
        iv = iv or ([(g0, g1)] if g1 > g0 else [(g0, g0 + 0.3 * (j - k))])
        for n, t in enumerate(_spread(_weights(toks[k:j]), iv)):
            times[k + n] = t
        k = j

    # tempos sempre crescentes
    last = 0.0
    for n, t in enumerate(times):
        s, e = t
        s = max(s, last)
        e = max(e, s + 0.05)
        times[n] = (s, e)
        last = s

    lines = []
    for li, l in enumerate(pasted):
        ws = [{"w": toks[n], "s": round(times[n][0], 2), "e": round(times[n][1], 2)}
              for n in range(len(toks)) if owner[n] == li]
        if not ws:
            continue
        lines.append({"start": ws[0]["s"], "end": ws[-1]["e"], "text": l["text"], "part": l["part"], "words": ws})
    return lines
