// Anomalous TTS node helpers (classic canvas and Vue nodes):
// - the weight drop-downs only show the selected character's files;
// - an "insert tag" button adds {emotion}, [character] and [pause] tags to the script;
// - classic canvas: advanced inputs fold behind a toggle (Vue nodes has its own).
// Data: GET /anomalous_tts/characters (summary list, docs/INTERFACE.md §5), cached here.
//
// Helper buttons are appended AFTER all real inputs. ComfyUI saves widget values by
// position, so a button in the middle would shift every later value on reload.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "AnomalousTTS_CharacterSpeech";
const AUTO = "自动";
const FILTERED = ["gpt_weights", "sovits_weights"];
const PAUSES = ["0.5", "1", "2"];
const HELPER_PREFIX = "anomalous_tts_";
const CACHE_MS = 30000;

let cache = { time: 0, data: null, pending: null };

function fetchCharacters() {
    if (cache.pending) return cache.pending;
    cache.pending = api
        .fetchApi("/anomalous_tts/characters")
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null)
        .then((data) => {
            cache = { time: Date.now(), data: data ?? cache.data ?? { characters: [] }, pending: null };
            return cache.data;
        });
    return cache.pending;
}

/** Cached data right away when we have any (refreshed in the background), else wait for it. */
async function characters() {
    if (cache.data) {
        if (Date.now() - cache.time > CACHE_MS) fetchCharacters();
        return cache.data;
    }
    return fetchCharacters();
}

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function realWidgets(node) {
    return (node.widgets ?? []).filter((w) => !String(w.name).startsWith(HELPER_PREFIX));
}

function filterCombos(node) {
    const character = widget(node, "character")?.value;
    for (const name of FILTERED) {
        const w = widget(node, name);
        if (!w) continue;
        w._attsAll ??= [...w.options.values];
        const own = w._attsAll.filter((v) => v !== AUTO && v.startsWith(`${character}/`));
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
    const el = widget(node, "text")?.inputEl;
    if (!el || el._attsCaretHooked) return;
    el._attsCaretHooked = true;
    const save = () => (node._attsCaret = el.selectionStart);
    el.addEventListener("keyup", save);
    el.addEventListener("click", save);
    el.addEventListener("blur", save);
}

function menuEntries(node, data) {
    const current = widget(node, "character")?.value;
    const chars = data?.characters ?? [];
    const me = chars.find((c) => c.name === current);
    const emotions = Object.keys(me?.emotions ?? {});
    const add = (text) => ({ content: text, callback: () => insertIntoText(node, text) });
    const entries = [{ content: "情绪", disabled: true }, add("{main}")];
    if (emotions.length) entries.push(...emotions.map((e) => add(`{${e}}`)));
    else entries.push({ content: "（还没有标注情绪，可在 Anomalous 里标注）", disabled: true });
    const aliasCount = {};
    for (const c of chars) for (const a of c.aliases ?? []) aliasCount[a] = (aliasCount[a] ?? 0) + 1;
    const others = chars.filter((c) => c.name !== current && !c.error);
    if (others.length) {
        entries.push(null, { content: "换人说", disabled: true });
        for (const c of others) entries.push(add(`[${(c.aliases ?? []).find((a) => aliasCount[a] === 1) ?? c.name}]`));
    }
    entries.push(null, { content: "停顿", disabled: true }, ...PAUSES.map((s) => add(`[pause:${s}]`)));
    return entries;
}

async function openTagMenu(node, event) {
    const data = await characters();
    const e = event ?? new MouseEvent("click", { clientX: innerWidth / 2, clientY: innerHeight / 3 });
    new LiteGraph.ContextMenu(menuEntries(node, data), { event: e, title: "插入标签" });
}

async function refreshLabel(node) {
    const btn = node._attsTags;
    if (!btn) return;
    const data = await characters();
    const me = (data?.characters ?? []).find((c) => c.name === widget(node, "character")?.value);
    const n = Object.keys(me?.emotions ?? {}).length;
    btn.label = n ? `插入标签（${n} 个情绪 / 换人 / 停顿）` : "插入标签（情绪 / 换人 / 停顿）";
    node.setDirtyCanvas?.(true, true);
}

function vueNodesEnabled() {
    try {
        return !!app.extensionManager?.setting?.get?.("Comfy.VueNodes.Enabled");
    } catch {
        return false;
    }
}

function applyAdvanced(node) {
    if (!node._attsAdvanced) return;
    const show = !!node.properties?.attsShowAdvanced;
    for (const w of realWidgets(node)) if (w.options?.advanced) w.hidden = !show;
    node._attsAdvanced.label = show ? "▾ 收起高级参数" : "▸ 高级参数（语言、参考、权重、采样…）";
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas?.(true, true);
}

function addHelper(node, name, label, onClick) {
    const w = node.addWidget("button", HELPER_PREFIX + name, null, onClick, { serialize: false });
    w.serialize = false;
    w.label = label;
    return w;
}

function setup(node) {
    const character = widget(node, "character");
    if (!character || node._attsReady) return;
    node._attsReady = true;
    const original = character.callback;
    character.callback = function (...args) {
        const r = original?.apply(this, args);
        filterCombos(node);
        refreshLabel(node);
        return r;
    };
    node._attsTags = addHelper(node, "tags", "插入标签（情绪 / 换人 / 停顿）", (_v, _c, _n, _p, e) =>
        openTagMenu(node, e ?? window.event)
    );
    if (!vueNodesEnabled()) {
        node.properties ??= {};
        node._attsAdvanced = addHelper(node, "advanced", "", () => {
            node.properties.attsShowAdvanced = !node.properties.attsShowAdvanced;
            applyAdvanced(node);
        });
    }
    applyAdvanced(node);
    filterCombos(node);
    refreshLabel(node);
    rememberCaret(node);
}

/**
 * Workflows saved by the first version of this script have two button slots (null) at
 * positions 2 and 6 of widgets_values. Drop them and re-apply the values by position.
 */
function migrateSavedValues(node, info) {
    const saved = info?.widgets_values;
    const real = realWidgets(node);
    if (!Array.isArray(saved) || saved.length !== real.length + 2 || saved[2] !== null || saved[6] !== null) return;
    const values = saved.filter((_, i) => i !== 2 && i !== 6);
    real.forEach((w, i) => (w.value = values[i]));
    const ref = widget(node, "reference_audio");
    if (ref && ref.value === AUTO) ref.value = ""; // was a drop-down, now a path box
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
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            setup(this);
            migrateSavedValues(this, info);
            filterCombos(this);
            applyAdvanced(this);
            refreshLabel(this);
            return r;
        };
    },
});
