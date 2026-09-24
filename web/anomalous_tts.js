// Anomalous TTS node helpers:
// - reference audio / weight drop-downs only show the selected character's files;
// - a button inserts {emotion}, [character] and [pause] tags into the script.
// Data: widget values (for filtering) and GET /anomalous_tts/characters (docs/INTERFACE.md §5).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "AnomalousTTS_CharacterSpeech";
const AUTO = "自动";
const FILTERED = ["reference_audio", "gpt_weights", "sovits_weights"];
const PAUSES = ["0.5", "1", "2"];

let cache = { time: 0, data: null, pending: null };

async function fetchCharacters(force = false) {
    if (!force && cache.data && Date.now() - cache.time < 5000) return cache.data;
    if (cache.pending) return cache.pending;
    cache.pending = api
        .fetchApi("/anomalous_tts/characters")
        .then((r) => (r.ok ? r.json() : { characters: [] }))
        .catch(() => ({ characters: [] }))
        .then((data) => {
            cache = { time: Date.now(), data, pending: null };
            return data;
        });
    return cache.pending;
}

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function filterCombos(node) {
    const character = widget(node, "character")?.value;
    for (const name of FILTERED) {
        const w = widget(node, name);
        if (!w) continue;
        w._attsAll ??= [...w.options.values];
        const prefix = `${character}/`;
        const own = w._attsAll.filter((v) => v !== AUTO && v.startsWith(prefix));
        w.options.values = [AUTO, ...own];
        if (w.value !== AUTO && !own.includes(w.value)) w.value = AUTO;
    }
}

function insertIntoText(node, snippet) {
    const w = widget(node, "text");
    if (!w) return;
    const el = w.inputEl ?? w.element;
    const value = String(w.value ?? "");
    let start = value.length;
    let end = value.length;
    if (el && typeof el.selectionStart === "number" && document.activeElement === el) {
        start = el.selectionStart;
        end = el.selectionEnd;
    } else if (typeof node._attsCaret === "number" && node._attsCaret <= value.length) {
        start = end = node._attsCaret;
    }
    const next = value.slice(0, start) + snippet + value.slice(end);
    w.value = next;
    if (el) el.value = next;
    node._attsCaret = start + snippet.length;
    w.callback?.(next);
    node.setDirtyCanvas?.(true, true);
}

function rememberCaret(node) {
    const w = widget(node, "text");
    const el = w?.inputEl ?? w?.element;
    if (!el || el._attsCaretHooked) return;
    el._attsCaretHooked = true;
    const save = () => (node._attsCaret = el.selectionStart);
    el.addEventListener("keyup", save);
    el.addEventListener("click", save);
    el.addEventListener("blur", save);
}

async function openTagMenu(node, event) {
    const data = await fetchCharacters();
    const current = widget(node, "character")?.value;
    const chars = data.characters ?? [];
    const me = chars.find((c) => c.name === current);
    const emotions = Object.keys(me?.emotions ?? {});
    const entries = [];
    entries.push({ content: "情绪", disabled: true });
    entries.push({ content: "{main}　主参考", callback: () => insertIntoText(node, "{main}") });
    for (const e of emotions) entries.push({ content: `{${e}}`, callback: () => insertIntoText(node, `{${e}}`) });
    if (!emotions.length) entries.push({ content: "（这个角色还没有标注情绪）", disabled: true });
    entries.push(null);
    entries.push({ content: "换人说", disabled: true });
    const aliasCount = {};
    for (const c of chars) for (const a of c.aliases ?? []) aliasCount[a] = (aliasCount[a] ?? 0) + 1;
    for (const c of chars) {
        if (c.name === current) continue;
        const alias = (c.aliases ?? []).find((a) => aliasCount[a] === 1);
        const tag = alias ?? c.name;
        entries.push({ content: `[${tag}]`, callback: () => insertIntoText(node, `[${tag}]`) });
    }
    entries.push(null);
    entries.push({ content: "停顿", disabled: true });
    for (const s of PAUSES) entries.push({ content: `[pause:${s}]`, callback: () => insertIntoText(node, `[pause:${s}]`) });
    new LiteGraph.ContextMenu(entries, { event, title: "插入标签", scale: Math.max(1, app.canvas?.ds?.scale ?? 1) });
}

async function refreshSummary(node) {
    const btn = node._attsButton;
    if (!btn) return;
    const data = await fetchCharacters();
    const me = (data.characters ?? []).find((c) => c.name === widget(node, "character")?.value);
    const emotions = Object.keys(me?.emotions ?? {});
    btn.label = emotions.length ? `插入标签（情绪：${emotions.join("、")}）` : "插入标签（情绪 / 换人 / 停顿）";
    node.setDirtyCanvas?.(true, true);
}

function vueNodesEnabled() {
    try {
        return !!app.extensionManager?.setting?.get?.("Comfy.VueNodes.Enabled");
    } catch {
        return false;
    }
}

// Classic canvas does not hide "advanced" inputs by itself (Vue nodes mode does), so add a toggle.
function applyAdvanced(node) {
    const show = !!node.properties?.attsShowAdvanced;
    for (const w of node.widgets ?? []) if (w.options?.advanced) w.hidden = !show;
    if (node._attsAdvButton) node._attsAdvButton.label = show ? "▾ 收起高级参数" : "▸ 高级参数（语言、参考音频、权重、采样…）";
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas?.(true, true);
}

function addAdvancedToggle(node) {
    if (vueNodesEnabled() || node._attsAdvButton) return;
    node.properties ??= {};
    node._attsAdvButton = node.addWidget(
        "button",
        "anomalous_tts_advanced",
        null,
        () => {
            node.properties.attsShowAdvanced = !node.properties.attsShowAdvanced;
            applyAdvanced(node);
        },
        { serialize: false }
    );
    node._attsAdvButton.serialize = false;
    const first = node.widgets.findIndex((w) => w.options?.advanced);
    if (first >= 0) {
        node.widgets.splice(node.widgets.indexOf(node._attsAdvButton), 1);
        node.widgets.splice(first, 0, node._attsAdvButton);
    }
    applyAdvanced(node);
}

function setup(node) {
    const character = widget(node, "character");
    if (!character || node._attsReady) return;
    node._attsReady = true;
    const original = character.callback;
    character.callback = function (value, ...rest) {
        const r = original?.call(this, value, ...rest);
        filterCombos(node);
        refreshSummary(node);
        return r;
    };
    node._attsButton = node.addWidget(
        "button",
        "anomalous_tts_tags",
        null,
        (_v, _c, _n, _p, e) => openTagMenu(node, e ?? window.event),
        { serialize: false }
    );
    node._attsButton.label = "插入标签（情绪 / 换人 / 停顿）";
    node._attsButton.serialize = false;
    // Put the button right after the text box.
    const idx = node.widgets.indexOf(widget(node, "text"));
    if (idx >= 0) {
        node.widgets.splice(node.widgets.indexOf(node._attsButton), 1);
        node.widgets.splice(idx + 1, 0, node._attsButton);
    }
    addAdvancedToggle(node);
    filterCombos(node);
    refreshSummary(node);
    rememberCaret(node);
}

app.registerExtension({
    name: "Anomalous.TTS",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            setup(this);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            // Saved workflows: restore values first, then filter.
            setTimeout(() => {
                setup(this);
                filterCombos(this);
                refreshSummary(this);
                if (this._attsAdvButton) applyAdvanced(this);
            }, 0);
            return r;
        };
    },
});
