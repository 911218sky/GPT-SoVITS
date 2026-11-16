import os
import re
from typing import Callable

punctuation = set(["!", "?", "…", ",", ".", "-", " "])
METHODS = dict()


def get_method(name: str) -> Callable:
    method = METHODS.get(name, None)
    if method is None:
        raise ValueError(f"Method {name} not found")
    return method


def get_method_names() -> list:
    return list(METHODS.keys())


def register_method(name):
    def decorator(func):
        METHODS[name] = func
        return func

    return decorator


splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}


def split_big_text(text, max_len=510):
    # 定义全角和半角标点符号
    punctuation = "".join(splits)

    # 切割文本
    segments = re.split("([" + punctuation + "])", text)

    # 初始化结果列表和当前片段
    result = []
    current_segment = ""

    for segment in segments:
        # 如果当前片段加上新的片段长度超过max_len，就将当前片段加入结果列表，并重置当前片段
        if len(current_segment + segment) > max_len:
            result.append(current_segment)
            current_segment = segment
        else:
            current_segment += segment

    # 将最后一个片段加入结果列表
    if current_segment:
        result.append(current_segment)

    return result



def split_for_fast_t2s(text, max_len=None):
    """Split only a run-on that has no punctuation of its own.

    Punctuation is handled by cut5 or cut20 before this runs. Cutting those
    groups again every 20 characters restarts the voice on every clause.
    A run-on longer than this still holds T2S for hundreds of steps, because
    one sentence is allowed to continue past the usual 320-step stop.
    20 characters is the 2026-11-16 remeasure that holds this chapter near 300 chars/s. 120 was 271.
    Override with GPT_SOVITS_FAST_SPLIT. Shorter text is returned unchanged.
    A tail shorter than 5 characters stays with the previous piece.
    """
    if max_len is None:
        try:
            max_len = int(os.environ.get("GPT_SOVITS_FAST_SPLIT", "20"))
        except ValueError:
            max_len = 20
        max_len = max(20, max_len)
    body = text
    end = ""
    if body and body[-1] in splits:
        end = body[-1]
        body = body[:-1]
    if len(body) <= max_len:
        return [text]
    chunks = []
    while body:
        chunks.append(body[:max_len])
        body = body[max_len:]
    if len(chunks) >= 2 and len(chunks[-1]) < 5:
        chunks[-2] += chunks[-1]
        chunks.pop()
    pieces = [chunk + "，" for chunk in chunks[:-1]]
    pieces.append(chunks[-1] + (end or "。"))
    return pieces


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


# 不切
@register_method("cut0")
def cut0(inp):
    if not set(inp).issubset(punctuation):
        return inp
    else:
        return "\n"


# 凑四句一切
@register_method("cut1")
def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# 凑50字一切
@register_method("cut2")
def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# 按中文句号。切
@register_method("cut3")
def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# 按英文句号.切
@register_method("cut4")
def cut4(inp):
    inp = inp.strip("\n")
    opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# 按标点符号切
# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
@register_method("cut5")
def cut5(inp):
    inp = inp.strip("\n")
    punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == "." and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


def default_split_method(version: str) -> str:
    """v5 DiT is faster on ~20-character breaths. v2Pro is faster on cut5."""
    if str(version).startswith("v5"):
        return "cut20"
    return "cut5"


def _pack_clauses(inp: str, limit: int) -> str:
    """Join punctuation clauses until about `limit` characters.

    One clause longer than `limit` is cut every `limit` characters.
    A tail shorter than 5 characters stays with the previous piece.
    """
    inp = inp.strip("\n")
    clauses = split(inp)
    pieces = []
    for clause in clauses:
        start = 0
        while start < len(clause):
            pieces.append(clause[start : start + limit])
            start += limit
    opts = []
    buf = ""
    for piece in pieces:
        if buf and len(buf) + len(piece) > limit:
            opts.append(buf)
            buf = piece
        else:
            buf += piece
    if buf:
        opts.append(buf)
    if len(opts) > 1 and len(opts[-1]) < 5:
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if item and not set(item).issubset(punctuation)]
    return "\n".join(opts)


@register_method("cut20")
def cut20(inp):
    """About 20 characters a segment. Used by v5turbo."""
    return _pack_clauses(inp, 20)


if __name__ == "__main__":
    method = get_method("cut5")
    print(method("你好，我是小明。你好，我是小红。你好，我是小刚。你好，我是小张。"))
