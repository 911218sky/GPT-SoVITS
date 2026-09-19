import os
import sys
import threading

from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(now_dir)

import re
import torch
from text.LangSegmenter import LangSegmenter
from text import chinese
from typing import Dict, List, Optional, Tuple
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
from transformers import AutoModelForMaskedLM, AutoTokenizer
from TTS_infer_pack.text_segmentation_method import split_big_text, splits, get_method as get_seg_method

from tools.i18n.i18n import I18nAuto, scan_language_list

language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)
punctuation = set(["!", "?", "…", ",", ".", "-"])

# 一次送進 BERT 的句子數；小說短句多時可顯著減少 kernel 啟動開銷
BERT_FEATURE_BATCH_SIZE = int(os.environ.get("GPT_SOVITS_BERT_BATCH_SIZE", "64"))


def get_first(text: str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def merge_short_text_in_array(texts: str, threshold: int) -> list:
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


class TextPreprocessor:
    def __init__(self, bert_model: AutoModelForMaskedLM, tokenizer: AutoTokenizer, device: torch.device):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.bert_lock = threading.RLock()

    def preprocess(self, text: str, lang: str, text_split_method: str, version: str = "v2") -> List[Dict]:
        print(f"############ {i18n('切分文本')} ############")
        text = self.replace_consecutive_punctuation(text)
        texts = self.pre_seg_text(text, lang, text_split_method)
        result = []
        print(f"############ {i18n('提取文本Bert特征')} ############")

        prepared: List[Optional[List[Tuple[list, list, str, str]]]] = []
        zh_norm_texts: List[str] = []
        zh_word2phs: List[list] = []
        zh_refs: List[Tuple[int, int]] = []

        with self.bert_lock:
            for text_item in texts:
                segments = self._collect_phone_segments(text_item, lang, version)
                if segments is None:
                    prepared.append(None)
                    continue
                prepared.append(segments)
                for seg_idx, (_phones, word2ph, norm_text, seg_lang) in enumerate(segments):
                    if seg_lang.replace("all_", "") == "zh":
                        zh_refs.append((len(prepared) - 1, seg_idx))
                        zh_norm_texts.append(norm_text)
                        zh_word2phs.append(word2ph)

            zh_features: List[torch.Tensor] = []
            batch_size = max(1, BERT_FEATURE_BATCH_SIZE)
            for start in tqdm(range(0, len(zh_norm_texts), batch_size), desc="BERT"):
                end = start + batch_size
                zh_features.extend(self.get_bert_feature_batch(zh_norm_texts[start:end], zh_word2phs[start:end]))

            feature_map = {ref: feat for ref, feat in zip(zh_refs, zh_features)}

            for text_idx, segments in enumerate(prepared):
                if segments is None:
                    continue
                phones_list: List[list] = []
                bert_list: List[torch.Tensor] = []
                norm_text_list: List[str] = []
                for seg_idx, (phones, _word2ph, norm_text, seg_lang) in enumerate(segments):
                    lang_key = seg_lang.replace("all_", "")
                    if lang_key == "zh":
                        bert = feature_map[(text_idx, seg_idx)]
                    else:
                        bert = torch.zeros(
                            (1024, len(phones)),
                            dtype=torch.float32,
                            device=self.device,
                        )
                    phones_list.append(phones)
                    bert_list.append(bert)
                    norm_text_list.append(norm_text)

                phones = sum(phones_list, [])
                if not phones:
                    continue
                bert_features = torch.cat(bert_list, dim=1)
                norm_text = "".join(norm_text_list)
                if norm_text == "":
                    continue
                result.append(
                    {
                        "phones": phones,
                        "bert_features": bert_features,
                        "norm_text": norm_text,
                    }
                )
        return result

    def pre_seg_text(self, text: str, lang: str, text_split_method: str):
        text = text.strip("\n")
        if len(text) == 0:
            return []
        if text[0] not in splits and len(get_first(text)) < 4:
            text = "。" + text if lang != "en" else "." + text
        print(i18n("实际输入的目标文本:"))
        print(text)

        seg_method = get_seg_method(text_split_method)
        text = seg_method(text)

        while "\n\n" in text:
            text = text.replace("\n\n", "\n")

        _texts = text.split("\n")
        _texts = self.filter_text(_texts)
        _texts = merge_short_text_in_array(_texts, 5)
        texts = []

        for text in _texts:
            # 解决输入目标文本的空行导致报错的问题
            if len(text.strip()) == 0:
                continue
            if not re.sub("\W+", "", text):
                # 检测一下，如果是纯符号，就跳过。
                continue
            if text[-1] not in splits:
                text += "。" if lang != "en" else "."

            # 解决句子过长导致Bert报错的问题
            if len(text) > 510:
                texts.extend(split_big_text(text))
            else:
                texts.append(text)

        print(i18n("实际输入的目标文本(切句后):"))
        print(texts)
        return texts

    def segment_and_extract_feature_for_text(
        self, text: str, language: str, version: str = "v1"
    ) -> Tuple[list, torch.Tensor, str]:
        return self.get_phones_and_bert(text, language, version)

    def _split_lang_segments(self, text: str, language: str) -> Tuple[List[str], List[str]]:
        textlist: List[str] = []
        langlist: List[str] = []
        if language == "all_zh":
            for tmp in LangSegmenter.getTexts(text, "zh"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_yue":
            for tmp in LangSegmenter.getTexts(text, "zh"):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_ja":
            for tmp in LangSegmenter.getTexts(text, "ja"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_ko":
            for tmp in LangSegmenter.getTexts(text, "ko"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "en":
            langlist.append("en")
            textlist.append(text)
        elif language == "auto":
            for tmp in LangSegmenter.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegmenter.getTexts(text):
                if langlist:
                    if (tmp["lang"] == "en" and langlist[-1] == "en") or (
                        tmp["lang"] != "en" and langlist[-1] != "en"
                    ):
                        textlist[-1] += tmp["text"]
                        continue
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日韩文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        return textlist, langlist

    def _collect_phone_segments(
        self, text: str, language: str, version: str, final: bool = False
    ) -> Optional[List[Tuple[list, list, str, str]]]:
        text = re.sub(r" {2,}", " ", text)
        textlist, langlist = self._split_lang_segments(text, language)
        segments: List[Tuple[list, list, str, str]] = []
        phones_len = 0
        for i in range(len(textlist)):
            phones, word2ph, norm_text = self.clean_text_inf(textlist[i], langlist[i], version)
            segments.append((phones, word2ph, norm_text, langlist[i]))
            phones_len += len(phones)

        if not final and phones_len < 6:
            return self._collect_phone_segments("." + text, language, version, final=True)
        if phones_len == 0:
            return None
        return segments

    def get_phones_and_bert(self, text: str, language: str, version: str, final: bool = False):
        with self.bert_lock:
            segments = self._collect_phone_segments(text, language, version, final=final)
            if segments is None:
                return [], torch.zeros((1024, 0), dtype=torch.float32, device=self.device), ""

            phones_list = []
            bert_list = []
            norm_text_list = []
            for phones, word2ph, norm_text, lang in segments:
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = "".join(norm_text_list)
            return phones, bert, norm_text

    def _expand_to_phone_features(self, token_features: torch.Tensor, word2ph: list) -> torch.Tensor:
        word2ph_tensor = torch.as_tensor(word2ph, device=token_features.device, dtype=torch.long)
        phone_level_feature = token_features.repeat_interleave(word2ph_tensor, dim=0)
        return phone_level_feature.T

    def get_bert_feature(self, text: str, word2ph: list) -> torch.Tensor:
        with torch.inference_mode():
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            res = self.bert_model(**inputs, output_hidden_states=True)
            # 維持在 GPU，避免每句 CPU↔GPU 來回拷貝
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0][1:-1]
        assert len(word2ph) == len(text)
        return self._expand_to_phone_features(res, word2ph)

    def get_bert_feature_batch(self, texts: List[str], word2phs: List[list]) -> List[torch.Tensor]:
        if not texts:
            return []
        if len(texts) == 1:
            return [self.get_bert_feature(texts[0], word2phs[0])]

        with torch.inference_mode():
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self.bert_model(**inputs, output_hidden_states=True)
            hidden = torch.cat(outputs["hidden_states"][-3:-2], -1)

        features: List[torch.Tensor] = []
        for index, (text, word2ph) in enumerate(zip(texts, word2phs)):
            assert len(word2ph) == len(text)
            token_features = hidden[index, 1 : 1 + len(text)]
            features.append(self._expand_to_phone_features(token_features, word2ph))
        return features

    def clean_text_inf(self, text: str, language: str, version: str = "v2"):
        language = language.replace("all_", "")
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones: list, word2ph: list, norm_text: str, language: str):
        language = language.replace("all_", "")
        if language == "zh":
            feature = self.get_bert_feature(norm_text, word2ph)
        else:
            feature = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float32,
                device=self.device,
            )

        return feature

    def filter_text(self, texts):
        _text = []
        if all(text in [None, " ", "\n", ""] for text in texts):
            raise ValueError(i18n("请输入有效文本"))
        for text in texts:
            if text in [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text

    def replace_consecutive_punctuation(self, text):
        punctuations = "".join(re.escape(p) for p in punctuation)
        pattern = f"([{punctuations}])([{punctuations}])+"
        result = re.sub(pattern, r"\1", text)
        return result
