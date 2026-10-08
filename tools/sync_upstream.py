"""Copy the GPT-SoVITS files we use from a checkout and re-apply our patches.

Usage:
    python tools/sync_upstream.py <path to GPT-SoVITS checkout>

Every patch must match exactly once, otherwise the script stops so the change
can be reviewed by hand. Record the new commit in UPSTREAM.md afterwards.
"""
import pathlib
import pickle
import shutil
import sys

SRC = pathlib.Path(sys.argv[1]) / "GPT_SoVITS"
ROOT = pathlib.Path(__file__).resolve().parent.parent / "vendor" / "gpt_sovits"

FILES = [
    "AR/models/t2s_model.py", "AR/models/utils.py",
    "AR/modules/embedding.py", "AR/modules/transformer.py", "AR/modules/activation.py",
    "AR/modules/scaling.py", "AR/modules/patched_mha_with_cache.py",
    "module/models.py", "module/commons.py", "module/modules.py", "module/attentions.py",
    "module/mrte_model.py", "module/quantize.py", "module/core_vq.py", "module/transforms.py",
    "module/mel_processing.py", "text/symbols.py", "text/symbols2.py",
    # Chinese front end
    "text/chinese2.py", "text/tone_sandhi.py", "text/opencpop-strict.txt",
    "text/zh_normalization/__init__.py", "text/zh_normalization/char_convert.py",
    "text/zh_normalization/chronology.py", "text/zh_normalization/constants.py",
    "text/zh_normalization/num.py", "text/zh_normalization/phonecode.py",
    "text/zh_normalization/quantifier.py", "text/zh_normalization/text_normlization.py",
    "text/g2pw/__init__.py", "text/g2pw/g2pw.py", "text/g2pw/onnx_api.py", "text/g2pw/dataset.py",
    "text/g2pw/utils.py", "text/g2pw/polyphonic.rep", "text/g2pw/polyphonic-fix.rep",
    # English front end (dictionaries are downloaded at run time, see core/paths.py)
    "text/english.py", "text/en_normalization/expend.py",
    # v2Pro / v2ProPlus speaker embedding
    "eres2net/ERes2NetV2.py", "eres2net/fusion.py", "eres2net/kaldi.py", "eres2net/pooling_layers.py",
]
for f in FILES:
    (ROOT / f).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC / f, ROOT / f)
shutil.copyfile(SRC.parent / "LICENSE", ROOT / "LICENSE")

# Upstream ships the English names dictionary only as a pickle. It is turned into text here, from
# the pinned checkout, so the node itself never unpickles anything (text/english.py reads the text).
with open(SRC / "text" / "namedict_cache.pickle", "rb") as f:
    names = pickle.load(f)
with open(ROOT / "text" / "namedict.rep", "w", encoding="utf-8", newline="\n") as f:
    for word, pronunciations in sorted(names.items()):
        f.write(f"{word}  {' '.join(pronunciations[0])}\n")

P = []
def p(f, old, new): P.append((f, old, new))

# --- package-relative imports (upstream relies on sys.path hacks) ---
p("AR/models/t2s_model.py", "from torchmetrics.classification import MulticlassAccuracy\n", "")
p("AR/models/t2s_model.py", "from AR.models.utils import (", "from .utils import (")
p("AR/models/t2s_model.py", """        self.ar_accuracy_metric = MulticlassAccuracy(
            self.vocab_size,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS,
        )
""", "        # Anomalous_TTS: training accuracy metric (torchmetrics) removed.\n")
p("AR/models/t2s_model.py", "from AR.modules.embedding import", "from ..modules.embedding import")
p("AR/models/t2s_model.py", "from AR.modules.transformer import", "from ..modules.transformer import")
p("AR/modules/transformer.py", "from AR.modules.activation import", "from .activation import")
p("AR/modules/transformer.py", "from AR.modules.scaling import", "from .scaling import")
p("AR/modules/activation.py", "from AR.modules.patched_mha_with_cache import", "from .patched_mha_with_cache import")
p("module/models.py", "from module import commons\nfrom module import modules\nfrom module import attentions\nfrom f5_tts.model import DiT\n",
  "from . import commons\nfrom . import modules\nfrom . import attentions\n# Anomalous_TTS: v3/v4 (DiT/CFM) not supported; import removed.\n")
p("module/models.py", "from module.commons import init_weights, get_padding\nfrom module.mrte_model import MRTE\nfrom module.quantize import ResidualVectorQuantizer\n",
  "from .commons import init_weights, get_padding\nfrom .mrte_model import MRTE\nfrom .quantize import ResidualVectorQuantizer\n")
p("module/models.py", "from text import symbols as symbols_v1\nfrom text import symbols2 as symbols_v2\n",
  "from ..text import symbols as symbols_v1\nfrom ..text import symbols2 as symbols_v2\n")
p("module/modules.py", "from module import commons\nfrom module.commons import init_weights, get_padding\nfrom module.transforms import",
  "from . import commons\nfrom .commons import init_weights, get_padding\nfrom .transforms import")
p("module/attentions.py", "from module import commons\nfrom module.modules import LayerNorm", "from . import commons\nfrom .modules import LayerNorm")
p("module/mrte_model.py", "from module.attentions import", "from .attentions import")
p("module/quantize.py", "from module.core_vq import", "from .core_vq import")
p("module/core_vq.py", "from module.distrib import broadcast_tensors, is_distributed\nfrom module.ddp_utils import SyncFunction\n",
  "# Anomalous_TTS: distributed-training helpers removed; inference never calls them.\n"
  "def broadcast_tensors(*args, **kwargs):\n    pass\n\n\ndef is_distributed():\n    return False\n\n\n"
  "class SyncFunction:\n    @staticmethod\n    def apply(x):\n        return x\n\n\n")
p("module/mel_processing.py", "from librosa.filters import mel as librosa_mel_fn\n",
  "# Anomalous_TTS: librosa imported lazily; only mel_spectrogram needs it.\n"
  "def librosa_mel_fn(*args, **kwargs):\n    from librosa.filters import mel\n\n    return mel(*args, **kwargs)\n")

# --- Chinese front end ---
p("text/chinese2.py", "import cn2an\n", "")
p("text/chinese2.py", 'normalizer = lambda x: cn2an.transform(x, "an2cn")\n', "")
p("text/chinese2.py", "from text.symbols import punctuation\nfrom text.tone_sandhi import ToneSandhi\nfrom text.zh_normalization.text_normlization import TextNormalizer\n",
  "from .symbols import punctuation\nfrom .tone_sandhi import ToneSandhi\nfrom .zh_normalization.text_normlization import TextNormalizer\n")
p("text/chinese2.py", """is_g2pw = True  # True if is_g2pw_str.lower() == 'true' else False
if is_g2pw:
    # print("当前使用g2pw进行拼音推理")
    from text.g2pw import G2PWPinyin, correct_pronunciation

    parent_directory = os.path.dirname(current_file_path)
    g2pw = G2PWPinyin(
        model_dir="GPT_SoVITS/text/G2PWModel",
        model_source=os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
        v_to_u=False,
        neutral_tone_with_five=True,
    )
""", """# Anomalous_TTS: g2pW is loaded on demand by enable_g2pw(); until then pypinyin is used
# (the same fallback GPT-SoVITS has when is_g2pw is False).
is_g2pw = False
g2pw = None
correct_pronunciation = None


def enable_g2pw(model_dir, tokenizer_dir):
    global is_g2pw, g2pw, correct_pronunciation
    from .g2pw import G2PWPinyin, correct_pronunciation as _correct

    g2pw = G2PWPinyin(
        model_dir=model_dir,
        model_source=tokenizer_dir,
        v_to_u=False,
        neutral_tone_with_five=True,
    )
    correct_pronunciation = _correct
    is_g2pw = True


def disable_g2pw():
    \"\"\"Anomalous_TTS: drop the g2pW session (~600 MB of RAM); pypinyin until enable_g2pw again.\"\"\"
    global is_g2pw, g2pw, correct_pronunciation
    is_g2pw = False
    g2pw = None
    correct_pronunciation = None

""")
p("text/chinese2.py", '            print("pypinyin结果", initials, finals)\n', "")
p("text/chinese2.py", "import jieba_fast\nimport logging\n\njieba_fast.setLogLevel(logging.CRITICAL)\nimport jieba_fast.posseg as psg\n",
  "import logging\n\n# Anomalous_TTS: fall back to pure-Python jieba (same results) when jieba_fast has no wheel.\n"
  "try:\n    import jieba_fast\n    import jieba_fast.posseg as psg\nexcept ImportError:\n    import jieba as jieba_fast\n    import jieba.posseg as psg\n\n"
  "jieba_fast.setLogLevel(logging.CRITICAL)\n")
p("text/tone_sandhi.py", "import jieba_fast as jieba\n",
  "try:\n    import jieba_fast as jieba\nexcept ImportError:  # Anomalous_TTS: pure-Python fallback\n    import jieba\n")
p("text/zh_normalization/__init__.py", "from text.zh_normalization.text_normlization import *", "from .text_normlization import *")
p("text/g2pw/__init__.py", "from text.g2pw.g2pw import *", "from .g2pw import *")
p("text/g2pw/onnx_api.py", "import requests\nfrom opencc import OpenCC\n", "")
p("text/g2pw/onnx_api.py", '            self.cc = OpenCC("s2tw")', '            from opencc import OpenCC\n\n            self.cc = OpenCC("s2tw")')
p("text/g2pw/onnx_api.py", "try:\n    onnxruntime.preload_dlls()\nexcept Exception:\n    pass\n", "")
p("text/g2pw/onnx_api.py", """        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            self.session_g2pw = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        else:
            self.session_g2pw = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
""", """        # CPU only: g2pW is a few milliseconds per sentence there. onnxruntime-gpu lists
        # CUDA as available even when its CUDA libraries do not match the ones torch
        # ships (e.g. onnxruntime for CUDA 12 next to torch cu130), then prints a red
        # load error on every run and falls back to CPU anyway.
        self.session_g2pw = onnxruntime.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
""")

# --- English front end ---
p("text/english.py", "from text.symbols import punctuation\n", "from .symbols import punctuation\n")
p("text/english.py", "from text.symbols2 import symbols\n", "from .symbols2 import symbols\n")
p("text/english.py", "from text.en_normalization.expend import normalize\n", "from .en_normalization.expend import normalize\n")
p("text/english.py", "current_file_path = os.path.dirname(__file__)\n",
  "# Anomalous_TTS: dictionaries live outside the package; configure() points here before first use.\n"
  "current_file_path = os.path.dirname(__file__)\n")
p("text/english.py", "_g2p = en_G2p()\n\n\ndef g2p(text):\n",
  "_g2p = None  # Anomalous_TTS: built on first use, after configure()\n\n\n"
  "def configure(dict_dir):\n"
  "    global CMU_DICT_PATH, CMU_DICT_FAST_PATH, CMU_DICT_HOT_PATH\n"
  "    CMU_DICT_PATH = os.path.join(dict_dir, \"cmudict.rep\")\n"
  "    CMU_DICT_FAST_PATH = os.path.join(dict_dir, \"cmudict-fast.rep\")\n"
  "    CMU_DICT_HOT_PATH = os.path.join(dict_dir, \"engdict-hot.rep\")\n\n\n"
  "def g2p(text):\n"
  "    global _g2p\n"
  "    if _g2p is None:\n"
  "        _g2p = en_G2p()\n")

# --- English dictionaries are UTF-8: upstream opens them with the locale encoding, which
# fails on Chinese Windows (GBK) the first time the cache is built ---
p("text/english.py", '    start_line = 49\n    with open(CMU_DICT_PATH) as f:\n',
  '    start_line = 49\n    with open(CMU_DICT_PATH, encoding="utf-8") as f:\n')
p("text/english.py", 'def read_dict_new():\n    g2p_dict = {}\n    with open(CMU_DICT_PATH) as f:\n',
  'def read_dict_new():\n    g2p_dict = {}\n    with open(CMU_DICT_PATH, encoding="utf-8") as f:\n')
p("text/english.py", '    with open(CMU_DICT_FAST_PATH) as f:\n',
  '    with open(CMU_DICT_FAST_PATH, encoding="utf-8") as f:\n')
p("text/english.py", '    with open(CMU_DICT_HOT_PATH) as f:\n',
  '    with open(CMU_DICT_HOT_PATH, encoding="utf-8") as f:\n')

# --- nothing read from data files is executed or unpickled ---
# g2pW's config.py (inside the downloaded G2PWModel) is read as data: its literal assignments.
p("text/g2pw/utils.py", """def _load_config(config_path: os.PathLike):
    import importlib.util

    spec = importlib.util.spec_from_file_location("__init__", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config
""", """def _load_config(config_path: os.PathLike):
    # Anomalous_TTS: the settings are read as data (top-level `name = literal` lines), never executed.
    import ast
    import types

    with open(config_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(config_path))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return types.SimpleNamespace(**values)
""")
# The model is found or downloaded by core/paths.py (pinned revision, SHA-256 checked); never here.
p("text/g2pw/onnx_api.py", """def download_and_decompress(model_dir: str = "G2PWModel/"):
    if not os.path.exists(model_dir):
        parent_directory = os.path.dirname(model_dir)
        zip_dir = os.path.join(parent_directory, "G2PWModel_1.1.zip")
        extract_dir = os.path.join(parent_directory, "G2PWModel_1.1")
        extract_dir_new = os.path.join(parent_directory, "G2PWModel")
        print("Downloading g2pw model...")
        modelscope_url = "https://www.modelscope.cn/models/kamiorinn/g2pw/resolve/master/G2PWModel_1.1.zip"
        with requests.get(modelscope_url, stream=True) as r:
            r.raise_for_status()
            with open(zip_dir, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print("Extracting g2pw model...")
        with zipfile.ZipFile(zip_dir, "r") as zip_ref:
            zip_ref.extractall(parent_directory)

        os.rename(extract_dir, extract_dir_new)

    return model_dir
""", """def download_and_decompress(model_dir: str = "G2PWModel/"):
    # Anomalous_TTS: never downloads; core/paths.py fetches G2PWModel at a pinned revision and
    # checks its SHA-256 before extracting it.
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"G2PWModel not found: {model_dir}")
    return model_dir
""")
# Polyphones: read from the .rep text (lists of quoted pinyin) without eval(), no pickle cache.
p("text/g2pw/g2pw.py", "import hashlib\nimport pickle\nimport os\n", "import os\nimport re\n")
p("text/g2pw/g2pw.py", """CACHE_PATH = os.path.join(current_file_path, "polyphonic.pickle")
PP_DICT_PATH = os.path.join(current_file_path, "polyphonic.rep")
PP_FIX_DICT_PATH = os.path.join(current_file_path, "polyphonic-fix.rep")
MD5_PATH = os.path.join(current_file_path, "polyphonic.md5")

def get_file_md5(file_path):
    if not os.path.exists(file_path):
        return ""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
""", """PP_DICT_PATH = os.path.join(current_file_path, "polyphonic.rep")
PP_FIX_DICT_PATH = os.path.join(current_file_path, "polyphonic-fix.rep")
""")
p("text/g2pw/g2pw.py", """def cache_dict(polyphonic_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(polyphonic_dict, pickle_file)


def get_dict():
    new_md5 = get_file_md5(PP_DICT_PATH) + get_file_md5(PP_FIX_DICT_PATH)
    old_md5 = ""
    if os.path.exists(MD5_PATH):
        with open(MD5_PATH, "r", encoding="utf-8") as f:
            old_md5 = f.read().strip()
    need_rebuild = (not os.path.exists(CACHE_PATH)) or (new_md5 != old_md5)

    if not need_rebuild:
        with open(CACHE_PATH, "rb") as pickle_file:
            polyphonic_dict = pickle.load(pickle_file)
    else:
        print("Rebuilding Polyphonic Dictionary: " + f"{old_md5} -> {new_md5}")
        polyphonic_dict = read_dict()
        cache_dict(polyphonic_dict, CACHE_PATH)
        with open(MD5_PATH, "w", encoding="utf-8") as f:
            f.write(new_md5)
    return polyphonic_dict


def read_dict():
    polyphonic_dict = {}
    with open(PP_DICT_PATH, encoding="utf-8") as f:
        line = f.readline()
        while line:
            key, value_str = line.split(":")
            value = eval(value_str.strip())
            polyphonic_dict[key.strip()] = value
            line = f.readline()
    with open(PP_FIX_DICT_PATH, encoding="utf-8") as f:
        line = f.readline()
        while line:
            key, value_str = line.split(":")
            value = eval(value_str.strip())
            polyphonic_dict[key.strip()] = value
            line = f.readline()
    return polyphonic_dict
""", """def get_dict():
    # Anomalous_TTS: read from the .rep text each time, no pickle cache.
    return read_dict()


def read_dict():
    # Anomalous_TTS: each value is a list of quoted pinyin, read as text instead of eval().
    polyphonic_dict = {}
    for path in (PP_DICT_PATH, PP_FIX_DICT_PATH):
        with open(path, encoding="utf-8") as f:
            for line in f:
                key, value_str = line.split(":")
                polyphonic_dict[key.strip()] = re.findall(r"['\\"]([^'\\"]*)['\\"]", value_str)
    return polyphonic_dict
""")
# English: the CMU dictionaries are read from their .rep text, no pickle cache; names come from
# namedict.rep, which this script writes from upstream's namedict_cache.pickle (below).
p("text/english.py", "import pickle\nimport os\n", "import os\n")
p("text/english.py", 'CACHE_PATH = os.path.join(current_file_path, "engdict_cache.pickle")\nNAMECACHE_PATH = os.path.join(current_file_path, "namedict_cache.pickle")\n',
  'NAMEDICT_PATH = os.path.join(current_file_path, "namedict.rep")  # Anomalous_TTS: text, shipped with the code\n')
p("text/english.py", """def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict_new()
        cache_dict(g2p_dict, CACHE_PATH)

    g2p_dict = hot_reload_hot(g2p_dict)

    return g2p_dict


def get_namedict():
    if os.path.exists(NAMECACHE_PATH):
        with open(NAMECACHE_PATH, "rb") as pickle_file:
            name_dict = pickle.load(pickle_file)
    else:
        name_dict = {}

    return name_dict
""", """def get_dict():
    # Anomalous_TTS: built from the .rep text each time, no pickle cache.
    return hot_reload_hot(read_dict_new())


def get_namedict():
    # Anomalous_TTS: one "word  PH PH ..." per line of namedict.rep.
    name_dict = {}
    with open(NAMEDICT_PATH, encoding="utf-8") as f:
        for line in f:
            word, _, phones = line.rstrip("\\n").partition("  ")
            if phones:
                name_dict[word] = [phones.split(" ")]
    return name_dict
""")

# --- tone sandhi: the reduplication rule only for whole reduplicated words ---
# Upstream makes the second of any two same characters in a word neutral, so 银行行长
# (one jieba word) reads yin2 hang2 hang5 zhang3. See UPSTREAM.md.
p("text/tone_sandhi.py", '                and word not in self.must_not_neural_tone_words\n            ):\n                finals[j] = finals[j][:-1] + "5"\n',
  '                and word not in self.must_not_neural_tone_words\n                # Anomalous_TTS: in a word of four or more characters only a leading pair (好好学习) or\n                # AABB (高高兴兴) is reduplication; same characters further in belong to two words\n                # (银行行长, 人民民主) and keep their tones.\n                and (len(word) <= 3 or j == 1 or (j == 3 and word[0] == word[1] and word[2] == word[3]))\n            ):\n                finals[j] = finals[j][:-1] + "5"\n')

# --- speaker verification (v2Pro) ---
p("eres2net/ERes2NetV2.py", "import pooling_layers as pooling_layers\nfrom fusion import AFF\n",
  "from . import pooling_layers as pooling_layers\nfrom .fusion import AFF\n")

bad = 0
for f, old, new in P:
    path = ROOT / f
    s = path.read_text(encoding="utf-8")
    n = s.count(old)
    if n != 1:
        print(f"PATCH FAILED ({n} matches): {f}: {old[:60]!r}"); bad += 1; continue
    path.write_text(s.replace(old, new), encoding="utf-8")
print("patches applied:", len(P) - bad, "failed:", bad)
sys.exit(1 if bad else 0)
