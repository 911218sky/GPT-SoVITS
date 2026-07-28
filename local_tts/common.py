from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(__file__).resolve().parent / "assets"
MODEL_ROOT = ASSET_ROOT
DATA_ROOT = ASSET_ROOT / "Data"


@dataclass(frozen=True)
class RoleProfile:
    gpt_weights_path: Path
    sovits_weights_path: Path
    ref_audio_path: Path
    prompt_text: str
    prompt_lang: str = "zh"
    speed_factor: float = 1.0


ROLE_PROFILES = {
    "Lele": RoleProfile(
        gpt_weights_path=MODEL_ROOT / "GPT_weights_v2Pro" / "Lele-e15.ckpt",
        sovits_weights_path=MODEL_ROOT / "SoVITS_weights_v2Pro" / "Lele_e8_s96.pth",
        ref_audio_path=DATA_ROOT / "Lele" / "因为它的面料挺柔软，轻薄的，不是那种厚厚的，比较有弹性.wav",
        prompt_text="因为它的面料挺柔软，轻薄的，不是那种厚厚的，比较有弹性",
        speed_factor=1.1,
    ),
    "Lele_Pro": RoleProfile(
        gpt_weights_path=MODEL_ROOT / "GPT_weights_v2Pro" / "Lele-e15.ckpt",
        sovits_weights_path=MODEL_ROOT / "SoVITS_weights_v2Pro" / "Lele_e8_s96.pth",
        ref_audio_path=DATA_ROOT / "Lele" / "因为它的面料挺柔软，轻薄的，不是那种厚厚的，比较有弹性.wav",
        prompt_text="因为它的面料挺柔软，轻薄的，不是那种厚厚的，比较有弹性",
        speed_factor=1.1,
    ),
    "阿甘": RoleProfile(
        gpt_weights_path=MODEL_ROOT / "GPT_weights_v2Pro" / "阿甘-e15.ckpt",
        sovits_weights_path=MODEL_ROOT / "SoVITS_weights_v2Pro" / "阿甘_e8_s88.pth",
        ref_audio_path=DATA_ROOT / "阿甘" / "对了，大熊师傅在第三区，像你这样的高手多不多呀？.wav",
        prompt_text="对了，大熊师傅在第三区，像你这样的高手多不多呀？",
        speed_factor=1.01,
    ),
    "Sesame": RoleProfile(
        gpt_weights_path=MODEL_ROOT / "GPT_weights_v2Pro" / "Sesame-e15.ckpt",
        sovits_weights_path=MODEL_ROOT / "SoVITS_weights_v2Pro" / "Sesame_e8_s144.pth",
        ref_audio_path=DATA_ROOT / "Sesame" / "因为我的朋友跟我说，他这里还是显示只有互关的朋友才能评论。.wav",
        prompt_text="因为我的朋友跟我说，他这里还是显示只有互关的朋友才能评论。",
        speed_factor=0.9,
    ),
    "真人男": RoleProfile(
        gpt_weights_path=MODEL_ROOT / "GPT_weights_v2Pro" / "真人男-e15.ckpt",
        sovits_weights_path=MODEL_ROOT / "SoVITS_weights_v2Pro" / "真人男_e8_s112.pth",
        ref_audio_path=DATA_ROOT / "真人男" / "还是你来吧，我突然间觉得好像也没有那么迫切的想要脱单了。.wav",
        prompt_text="还是你来吧，我突然间觉得好像也没有那么迫切的想要脱单了。",
        speed_factor=0.9,
    ),
}


def get_role_profile(role_name: str) -> RoleProfile:
    try:
        return ROLE_PROFILES[role_name]
    except KeyError as error:
        choices = ", ".join(ROLE_PROFILES)
        raise ValueError(f"未知角色 {role_name!r}，可用角色：{choices}") from error


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")
    return path
