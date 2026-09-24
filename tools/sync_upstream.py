"""Copy the GPT-SoVITS files we use from a checkout and re-apply our patches.

Usage:
    python tools/sync_upstream.py <path to GPT-SoVITS checkout>

Every patch must match exactly once, otherwise the script stops so the change
can be reviewed by hand. Record the new commit in UPSTREAM.md afterwards.
"""
import pathlib
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
    "text/g2pw/utils.py", "text/g2pw/polyphonic.pickle", "text/g2pw/polyphonic.rep",
    "text/g2pw/polyphonic-fix.rep", "text/g2pw/polyphonic.md5",
    # English front end (dictionaries are downloaded at run time, see core/paths.py)
    "text/english.py", "text/en_normalization/expend.py",
    # v2Pro / v2ProPlus speaker embedding
    "eres2net/ERes2NetV2.py", "eres2net/fusion.py", "eres2net/kaldi.py", "eres2net/pooling_layers.py",
]
for f in FILES:
    (ROOT / f).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC / f, ROOT / f)
shutil.copyfile(SRC.parent / "LICENSE", ROOT / "LICENSE")

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
p("text/g2pw/onnx_api.py", "        with requests.get(modelscope_url, stream=True) as r:", "        import requests\n\n        with requests.get(modelscope_url, stream=True) as r:")
p("text/g2pw/onnx_api.py", '            self.cc = OpenCC("s2tw")', '            from opencc import OpenCC\n\n            self.cc = OpenCC("s2tw")')

# --- English front end ---
p("text/english.py", "from text.symbols import punctuation\n", "from .symbols import punctuation\n")
p("text/english.py", "from text.symbols2 import symbols\n", "from .symbols2 import symbols\n")
p("text/english.py", "from text.en_normalization.expend import normalize\n", "from .en_normalization.expend import normalize\n")
p("text/english.py", "current_file_path = os.path.dirname(__file__)\n",
  "# Anomalous_TTS: dictionaries live outside the package; configure() points here before first use.\n"
  "current_file_path = os.path.dirname(__file__)\n")
p("text/english.py", "_g2p = en_G2p()\n\n\ndef g2p(text):\n",
  "_g2p = None  # Anomalous_TTS: built on first use, after configure()\n\n\n"
  "def configure(dict_dir, cache_dir):\n"
  "    global CMU_DICT_PATH, CMU_DICT_FAST_PATH, CMU_DICT_HOT_PATH, CACHE_PATH, NAMECACHE_PATH\n"
  "    CMU_DICT_PATH = os.path.join(dict_dir, \"cmudict.rep\")\n"
  "    CMU_DICT_FAST_PATH = os.path.join(dict_dir, \"cmudict-fast.rep\")\n"
  "    CMU_DICT_HOT_PATH = os.path.join(dict_dir, \"engdict-hot.rep\")\n"
  "    NAMECACHE_PATH = os.path.join(dict_dir, \"namedict_cache.pickle\")\n"
  "    cached = os.path.join(dict_dir, \"engdict_cache.pickle\")\n"
  "    CACHE_PATH = cached if os.path.exists(cached) else os.path.join(cache_dir, \"engdict_cache.pickle\")\n\n\n"
  "def g2p(text):\n"
  "    global _g2p\n"
  "    if _g2p is None:\n"
  "        _g2p = en_G2p()\n")

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
