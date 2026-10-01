// Anomalous TTS node helpers (classic canvas and Vue nodes):
// - the weight drop-downs only show the selected character's files;
// - an "insert tag" button adds {emotion}, [character] and [pause] tags to the script;
// - a "character" menu plays the reference clips, reloads the character list and opens
//   the character in Anomalous Model Browser (window.anomalous_open_voice, INTERFACE.md §6);
// - a warning line appears only when the character has a problem or its language's
//   Python packages are missing;
// - classic canvas: advanced inputs fold behind a toggle (Vue nodes has its own).
// Text follows ComfyUI's language (Chinese, else English).
// Data: GET /anomalous_tts/characters (summary list, docs/INTERFACE.md §5) and
// GET /anomalous_tts/status (dependencies), both cached here.
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
const AMB_URL = "https://github.com/DemonGatanjieu/Anomalous_Model_Browser";

const TEXT = {
    zh: {
        tags: "插入标签（情绪 / 换人 / 停顿）",
        tagsCount: "插入标签（{n} 个情绪 / 换人 / 停顿）",
        tagsTitle: "插入标签",
        emotions: "情绪",
        noEmotions: "（还没有标注情绪，可在 Anomalous 里标注）",
        speakers: "换人说",
        pauses: "停顿",
        character: "角色：试听 / 刷新 / 导入与编辑…",
        characterTitle: "角色：{name}",
        play: "▶ 试听 {tag}",
        stop: "■ 停止试听",
        noClips: "（这个角色还没有参考音频）",
        refresh: "↻ 刷新角色列表",
        refreshed: "角色列表已刷新",
        manage: "✎ 导入或编辑角色（Anomalous Model Browser）",
        manageInstall: "✎ 导入或编辑角色：需要安装 Anomalous Model Browser",
        manageHint: "导入角色、标注情绪、改读音需要 Anomalous Model Browser。正在打开它的项目主页。",
        brokenCharacter: "⚠ 这个角色有问题，点这里查看",
        missingPackages: "⚠ 缺少 Python 包：{packages}（点这里复制安装命令）",
        copied: "安装命令已复制。在命令行运行，完成后重启 ComfyUI。",
        copyFailed: "复制失败，请手动运行：",
        advancedShow: "▸ 高级参数（语言、参考、权重、采样…）",
        advancedHide: "▾ 收起高级参数",
    },
    en: {
        tags: "Insert tag (emotion / speaker / pause)",
        tagsCount: "Insert tag ({n} emotions / speaker / pause)",
        tagsTitle: "Insert tag",
        emotions: "Emotion",
        noEmotions: "(no emotions yet; add them in Anomalous)",
        speakers: "Switch speaker",
        pauses: "Pause",
        character: "Character: listen / reload / import & edit…",
        characterTitle: "Character: {name}",
        play: "▶ Listen to {tag}",
        stop: "■ Stop",
        noClips: "(this character has no reference clip yet)",
        refresh: "↻ Reload characters",
        refreshed: "Character list reloaded",
        manage: "✎ Import or edit characters (Anomalous Model Browser)",
        manageInstall: "✎ Import or edit characters: needs Anomalous Model Browser",
        manageHint: "Importing characters and editing emotions or pronunciations needs Anomalous Model Browser. Opening its project page.",
        brokenCharacter: "⚠ This character has a problem; click to see it",
        missingPackages: "⚠ Missing Python packages: {packages} (click to copy the install command)",
        copied: "Install command copied. Run it in a terminal, then restart ComfyUI.",
        copyFailed: "Could not copy; run this yourself:",
        advancedShow: "▸ Advanced (language, reference, weights, sampling…)",
        advancedHide: "▾ Hide advanced",
    },
};

function lang() {
    let locale = "";
    try {
        locale = app.extensionManager?.setting?.get?.("Comfy.Locale") ?? "";
    } catch {
        locale = "";
    }
    return String(locale || navigator.language || "").toLowerCase().startsWith("zh") ? "zh" : "en";
}

function t(key, params = {}) {
    return TEXT[lang()][key].replace(/\{(\w+)\}/g, (m, k) => (k in params ? String(params[k]) : m));
}

function notify(detail, severity = "info") {
    const toast = app.extensionManager?.toast;
    if (toast?.add) toast.add({ severity, summary: "Anomalous TTS", detail, life: 5000 });
    else alert(detail);
}

let cache = { time: 0, data: null, pending: null };
let statusCache = { time: 0, data: null, pending: null };

function fetchCharacters(refresh = false) {
    if (cache.pending && !refresh) return cache.pending;
    cache.pending = api
        .fetchApi(`/anomalous_tts/characters${refresh ? "?refresh=1" : ""}`)
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

async function status() {
    if (statusCache.data && Date.now() - statusCache.time < CACHE_MS) return statusCache.data;
    statusCache.pending ??= api
        .fetchApi("/anomalous_tts/status")
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null)
        .then((data) => {
            statusCache = { time: Date.now(), data, pending: null };
            return data;
        });
    return statusCache.pending;
}

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function realWidgets(node) {
    return (node.widgets ?? []).filter((w) => !String(w.name).startsWith(HELPER_PREFIX));
}

async function currentCharacter(node) {
    const data = await characters();
    return (data?.characters ?? []).find((c) => c.name === widget(node, "character")?.value) ?? null;
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

function menuEvent(event) {
    return event ?? window.event ?? new MouseEvent("click", { clientX: innerWidth / 2, clientY: innerHeight / 3 });
}

function tagEntries(node, data) {
    const current = widget(node, "character")?.value;
    const chars = data?.characters ?? [];
    const me = chars.find((c) => c.name === current);
    const emotions = Object.keys(me?.emotions ?? {});
    const add = (text) => ({ content: text, callback: () => insertIntoText(node, text) });
    const entries = [{ content: t("emotions"), disabled: true }, add("{main}")];
    if (emotions.length) entries.push(...emotions.map((e) => add(`{${e}}`)));
    else entries.push({ content: t("noEmotions"), disabled: true });
    const aliasCount = {};
    for (const c of chars) for (const a of c.aliases ?? []) aliasCount[a] = (aliasCount[a] ?? 0) + 1;
    const others = chars.filter((c) => c.name !== current && !c.error);
    if (others.length) {
        entries.push(null, { content: t("speakers"), disabled: true });
        for (const c of others) entries.push(add(`[${(c.aliases ?? []).find((a) => aliasCount[a] === 1) ?? c.name}]`));
    }
    entries.push(null, { content: t("pauses"), disabled: true }, ...PAUSES.map((s) => add(`[pause:${s}]`)));
    return entries;
}

async function openTagMenu(node, event) {
    const data = await characters();
    new LiteGraph.ContextMenu(tagEntries(node, data), { event: menuEvent(event), title: t("tagsTitle") });
}

// ---------- the character menu ----------

let player = null; // the one clip playing, from any node

function stopClip() {
    if (!player) return;
    player.pause();
    player = null;
}

function playClip(character, path) {
    stopClip();
    const audio = new Audio(api.apiURL(
        `/anomalous_tts/audio?character=${encodeURIComponent(character)}&path=${encodeURIComponent(path)}`
    ));
    player = audio;
    audio.onended = audio.onerror = () => { if (player === audio) player = null; };
    audio.play().catch(() => { if (player === audio) player = null; });
}

/** After a reload: the character drop-downs list new characters (ComfyUI's own refresh), weights re-filter. */
async function reloadCharacters() {
    await fetchCharacters(true);
    statusCache.time = 0;
    await app.refreshComboInNodes?.();
    for (const node of app.graph?._nodes ?? []) {
        if (node.type !== NODE) continue;
        for (const name of FILTERED) {
            const w = widget(node, name);
            if (w) delete w._attsAll;
        }
        filterCombos(node);
        refreshLabels(node);
    }
    notify(t("refreshed"));
}

function openInAnomalous(character) {
    if (typeof window.anomalous_open_voice === "function" && window.anomalous_open_voice(character)) return;
    notify(t("manageHint"));
    window.open(AMB_URL, "_blank", "noopener");
}

function short(text, max = 24) {
    const s = String(text || "");
    return s.length > max ? `${s.slice(0, max)}…` : s;
}

async function openCharacterMenu(node, event) {
    const me = await currentCharacter(node);
    const name = widget(node, "character")?.value ?? "";
    const clips = [];
    if (me?.reference?.audio) clips.push(["{main}", me.reference]);
    for (const [emotion, ref] of Object.entries(me?.emotions ?? {})) if (ref?.audio) clips.push([`{${emotion}}`, ref]);
    const entries = clips.length
        ? clips.map(([tag, ref]) => ({ content: `${t("play", { tag })}  ${short(ref.text)}`, callback: () => playClip(name, ref.audio) }))
        : [{ content: t("noClips"), disabled: true }];
    if (player) entries.push({ content: t("stop"), callback: stopClip });
    entries.push(null, { content: t("refresh"), callback: reloadCharacters });
    const hasAnomalous = typeof window.anomalous_open_voice === "function";
    entries.push({ content: t(hasAnomalous ? "manage" : "manageInstall"), callback: () => openInAnomalous(name) });
    new LiteGraph.ContextMenu(entries, { event: menuEvent(event), title: t("characterTitle", { name }) });
}

// ---------- the warning line ----------

/** What is wrong with the selected character right now, or null. */
async function problem(node) {
    const me = await currentCharacter(node);
    if (!me) return null;
    const error = me.error || me.settings_error;
    if (error) return { label: t("brokenCharacter"), run: () => notify(`${me.name}: ${error}`, "warn") };
    const deps = (await status())?.dependencies?.[me.language];
    if (!deps || deps.ok) return null;
    return {
        label: t("missingPackages", { packages: deps.missing.join(", ") }),
        run: () => navigator.clipboard.writeText(deps.command)
            .then(() => notify(t("copied")))
            .catch(() => notify(`${t("copyFailed")} ${deps.command}`, "warn")),
    };
}

async function refreshWarning(node) {
    const w = node._attsWarning;
    if (!w) return;
    const found = await problem(node);
    w.hidden = !found;
    w.label = found?.label ?? "";
    node._attsProblem = found;
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas?.(true, true);
}

async function refreshLabels(node) {
    if (node._attsCharacter) node._attsCharacter.label = t("character");
    const btn = node._attsTags;
    if (btn) {
        const me = await currentCharacter(node);
        const n = Object.keys(me?.emotions ?? {}).length;
        btn.label = n ? t("tagsCount", { n }) : t("tags");
    }
    await refreshWarning(node);
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
    node._attsAdvanced.label = t(show ? "advancedHide" : "advancedShow");
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
        refreshLabels(node);
        return r;
    };
    node._attsWarning = addHelper(node, "warning", "", () => node._attsProblem?.run());
    node._attsWarning.hidden = true;
    node._attsTags = addHelper(node, "tags", t("tags"), (_v, _c, _n, _p, e) => openTagMenu(node, e));
    node._attsCharacter = addHelper(node, "character", t("character"), (_v, _c, _n, _p, e) => openCharacterMenu(node, e));
    if (!vueNodesEnabled()) {
        node.properties ??= {};
        node._attsAdvanced = addHelper(node, "advanced", "", () => {
            node.properties.attsShowAdvanced = !node.properties.attsShowAdvanced;
            applyAdvanced(node);
        });
    }
    applyAdvanced(node);
    filterCombos(node);
    refreshLabels(node);
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
            refreshLabels(this);
            return r;
        };
    },
});
