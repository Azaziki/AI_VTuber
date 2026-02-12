# -*- coding: utf-8 -*-
"""
AI VTuber 主脚本（最终整合稳定版）
- 嘴型：VoiceVolumePlusMouthOpen + VoiceFrequencyPlusMouthSmile
- 情绪：从 LLM 回复中提取关键词/标签（并在 TTS 前自动删除，避免读出来）
- 表情：可选启用 exp3（按 EMO_EXP_FILES 映射）
- 眨眼：监听 UDP（blink_sender.py 发 BLINK:0.xx），并注入 Tracking 参数
  - 若 VTS 没有现成 EyeOpen tracking 参数：自动创建 custom tracking 参数 AiEyeOpenL/AiEyeOpenR
  - 你需要在 VTS 里把 AiEyeOpenL/R 映射到模型的 Live2D 眼睛参数（只需一次）
- Windows：自动“新开控制台”运行 blink_sender.py（异步，不阻塞）
"""

import asyncio
import json
import os
import platform
import random
import shutil
import sys
import time
import wave
import re
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import requests
import websockets
import edge_tts
import sounddevice as sd
from scipy.io import wavfile

if platform.system() == "Windows":
    import winsound

# ================== VTS 配置 ==================
VTS_WS_URL = "ws://127.0.0.1:8001"
API_NAME = "VTubeStudioPublicAPI"
API_VERSION = "1.0"

PLUGIN_NAME = "AI_VTuber_Full"
PLUGIN_AUTHOR = "User"
TOKEN_FILE = "vts_token_ai_vtuber.json"

# ================== 嘴型（VTS Tracking 参数） ==================
P_MOUTH_OPEN = "VoiceVolumePlusMouthOpen"
P_MOUTH_SMILE = "VoiceFrequencyPlusMouthSmile"

# ================== 眨眼（Tracking 参数） ==================
# 先找现成的 tracking 眼睛开合参数；找不到就创建 custom tracking 参数
EYE_L_CANDIDATES = [
    "EyeOpenLeft", "EyeOpenL", "EyeLeftOpen", "EyeOpen_L",
    "EyeBlinkLeft", "EyeBlinkL", "EyeL", "EyeOpen"
]
EYE_R_CANDIDATES = [
    "EyeOpenRight", "EyeOpenR", "EyeRightOpen", "EyeOpen_R",
    "EyeBlinkRight", "EyeBlinkR", "EyeR", "EyeOpen"
]

CUSTOM_EYE_L = "AiEyeOpenL"
CUSTOM_EYE_R = "AiEyeOpenR"
CUSTOM_HEAD_X = "AiHeadX"
CUSTOM_HEAD_Y = "AiHeadY"
CUSTOM_HEAD_Z = "AiHeadZ"
CUSTOM_GAZE_X = "AiGazeX"
CUSTOM_GAZE_Y = "AiGazeY"


CUSTOM_HAND_LX = "AiHandLX"
CUSTOM_HAND_LY = "AiHandLY"
CUSTOM_HAND_RX = "AiHandRX"
CUSTOM_HAND_RY = "AiHandRY"
CUSTOM_HAND_WAVE = "AiHandWave"  # 可选：0..1（用于挥手/强调）
# ===== 待机头部/眼球 tracking 参数候选（会自动探测可用项） =====
HEAD_CANDIDATES_X = ["FaceAngleX", "HeadAngleX", "AngleX", "RotationX"]
HEAD_CANDIDATES_Y = ["FaceAngleY", "HeadAngleY", "AngleY", "RotationY"]
HEAD_CANDIDATES_Z = ["FaceAngleZ", "HeadAngleZ", "AngleZ", "RotationZ"]
EYE_CANDIDATES_X = ["EyeX", "EyeBallX", "GazeX", "LookX"]
EYE_CANDIDATES_Y = ["EyeY", "EyeBallY", "GazeY", "LookY"]
# ===== 手部 tracking 参数候选（如果 VTS 输入里有，就优先用；没有就创建 custom）=====
HAND_CANDIDATES_LX = ["HandLX", "LeftHandX", "ArmLX", "LeftArmX", "LArmX", "LHandX"]
HAND_CANDIDATES_LY = ["HandLY", "LeftHandY", "ArmLY", "LeftArmY", "LArmY", "LHandY"]
HAND_CANDIDATES_RX = ["HandRX", "RightHandX", "ArmRX", "RightArmX", "RArmX", "RHandX"]
HAND_CANDIDATES_RY = ["HandRY", "RightHandY", "ArmRY", "RightArmY", "RArmY", "RHandY"]
HAND_CANDIDATES_WAVE = ["HandWave", "Wave", "Gesture", "HandGesture"]

P_HAND_LX: Optional[str] = None
P_HAND_LY: Optional[str] = None
P_HAND_RX: Optional[str] = None
P_HAND_RY: Optional[str] = None
P_HAND_WAVE: Optional[str] = None

P_HEAD_X: Optional[str] = None
P_HEAD_Y: Optional[str] = None
P_HEAD_Z: Optional[str] = None
P_GAZE_X: Optional[str] = None
P_GAZE_Y: Optional[str] = None

# 注入到 tracking 的参数名（运行时确定）
P_EYE_L_OPEN: Optional[str] = None
P_EYE_R_OPEN: Optional[str] = None
USING_CUSTOM_EYE_PARAMS = False

# 眨眼互斥与冷却（防卡顿/双眨）
blink_lock = asyncio.Lock()
_last_blink_ts = 0.0

# 如果你映射后发现“总是闭眼/反了”，把这个改成 True（等价于 VTS 映射里勾 Invert）
INVERT_EYE = False  # True: 0=睁 1=闭；False: 0=闭 1=睁（默认）

# 眨眼开眼幅度限制：防止“瞪眼”（把 1.0 压到更小的开眼值）
# 0.0=最闭，1.0=最开；如果你映射到模型后太夸张，把 EYE_OPEN_MAX 调小，比如 0.6~0.85
EYE_OPEN_MIN = 0.05
EYE_OPEN_MAX = 0.65

# ================== UDP 眨眼联动 ==================
ENABLE_UDP_BLINK = True
UDP_HOST = "127.0.0.1"
UDP_PORT = 49721
ENABLE_NATURAL_BLINK = True  # 没收到 UDP 时，也会随机自然眨眼

# Windows 自动新开控制台运行 blink_sender.py（你也可以手动开）
AUTO_START_BLINK_SENDER = True
BLINK_SENDER_PATH = "blink_sender.py"

# ================== LLM / TTS ==================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "vtuber:latest"  # 如果你做了 Modelfile 自定义镜像，在这里改名

VOICE = "zh-CN-XiaoxiaoNeural"
TMP_MP3 = "tts.mp3"
TMP_WAV = "tts.wav"

# ================== 音频输出设备（虚拟麦克风）==================
# None = 使用系统默认播放设备（原来的行为）
# 字符串 = 设备名称中包含的关键词（推荐方式）
# 整数 = 直接使用设备索引（最精确，运行后看打印的 [数字]）
AUDIO_OUTPUT_DEVICE = "Voicemeeter Input"   # ←←← 这里改成你的虚拟设备名称
# 常见示例：
# "Voicemeeter Input" / "CABLE Input" / "VB-Audio" / "BlackHole" / "Synchronous"

# 音频分析（用于嘴型）
FRAME_MS = 40  # 25fps
VOL_GAIN = 1.20
SMILE_GAIN = 1.00
FREQ_MIN = 80.0
FREQ_MAX = 350.0

# ================== 表情（exp3）可选 ==================
ENABLE_EXPRESSIONS = True
EMO_EXP_FILES = {
    "happy": ["xinxin.exp3.json", "bq3.exp3.json"],
    "sad": ["ku.exp3.json", "st.exp3.json"],
    "angry": ["ga.exp3.json"],
    "surprise": ["sq.exp3.json"],
    "neutral": [],
}
USE_RANDOM_EXPRESSION = True

# ================== 调试开关 ==================
DEBUG_BLINK = False
DEBUG_EMO = False

# 待机随机头部/眼球转动（说话时自动暂停，避免影响嘴型/眨眼）
ENABLE_IDLE_MOTION = True
DEBUG_MOTION = False
MOTION_HZ = 20  # 注入频率（真人感建议 18~24；太高会挤占其它注入）
MOTION_INTERVAL_SEC = (3.0, 6.0)  # 注视停留更久，更像真人
MOTION_EASE_SEC = (0.60, 1.40)     # 过渡更慢更自然
# 头部角度范围（单位通常 -30~30；按你的模型调小/调大）
HEAD_X_RANGE = (-4.0, 4.0)
HEAD_Y_RANGE = (-3.0, 3.0)
HEAD_Z_RANGE = (-2.0, 2.0)
# 眼球范围（通常 -1~1 左右；太夸张就调小）
EYE_X_RANGE = (-0.28, 0.28)
EYE_Y_RANGE = (-0.18, 0.18)
# ================== 手部运动（自定义 tracking 参数 + 映射） ==================
ENABLE_HAND_MOTION = True
DEBUG_HAND = False

HAND_X_RANGE = (-0.35, 0.35)   # 左右摆动
HAND_Y_RANGE = (-0.20, 0.20)   # 上下摆动
HAND_HZ = 10                   # 注入频率（建议 6~12）
HAND_INTERVAL_SEC = (2.2, 4.8) # 选新目标间隔
HAND_EASE_SEC = (0.45, 1.05)   # 平滑过渡时长

SPEAK_GESTURE_GAIN = 0.55      # 说话时手势增益（0..1）
WAVE_DURATION_SEC = 0.75       # 一次挥手持续时间
WAVE_COOLDOWN_SEC = (2.0, 5.0) # 两次挥手间隔

# 眨眼稳定性：冷却/防双眨
BLINK_COOLDOWN_SEC = 0.35  # 两次眨眼最小间隔（建议 0.25~0.45）
BLINK_QUEUE_FLUSH = True  # 收到一次眨眼后清空队列，只保留最后一次（防连发双眨）
# 眨眼回睁眼关键帧（0~1 归一化，1.0 会映射到 EYE_OPEN_MAX）
BLINK_OPEN_FRAMES = (0.25, 0.55, 0.80, 1.0)


# -------------------------------------------------
# 工具：token
# -------------------------------------------------
def load_token() -> Optional[str]:
    if not os.path.exists(TOKEN_FILE):
        return None
    try:
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("token")
    except Exception:
        return None


def save_token(token: str) -> None:
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        json.dump({"token": token}, f, ensure_ascii=False, indent=2)


# -------------------------------------------------
# VTS Client（串行 rpc）
# -------------------------------------------------
class VTSClient:
    def __init__(self, ws):
        self.ws = ws
        self.lock = asyncio.Lock()

    async def rpc(self, payload: dict) -> dict:
        async with self.lock:
            await self.ws.send(json.dumps(payload))
            return json.loads(await self.ws.recv())

    async def authenticate(self) -> bool:
        token = load_token()
        if not token:
            req = {
                "apiName": API_NAME, "apiVersion": API_VERSION,
                "requestID": "token",
                "messageType": "AuthenticationTokenRequest",
                "data": {"pluginName": PLUGIN_NAME, "pluginDeveloper": PLUGIN_AUTHOR},
            }
            resp = await self.rpc(req)
            token = resp.get("data", {}).get("authenticationToken")
            if not token:
                return False
            save_token(token)
            print("✅ token saved. VTS 弹窗请点 Allow/允许")

        auth_req = {
            "apiName": API_NAME, "apiVersion": API_VERSION,
            "requestID": "auth",
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": PLUGIN_NAME,
                "pluginDeveloper": PLUGIN_AUTHOR,
                "authenticationToken": token,
            },
        }
        resp = await self.rpc(auth_req)
        return bool(resp.get("data", {}).get("authenticated", False))

    async def inject(self, params: Dict[str, float], mode: str = "Set") -> dict:
        payload = {
            "apiName": API_NAME, "apiVersion": API_VERSION,
            "requestID": f"inject-{int(time.time()*1000)}-{random.randint(100,999)}",
            "messageType": "InjectParameterDataRequest",
            "data": {
                "mode": mode,
                "parameterValues": [{"id": k, "value": float(v)} for k, v in params.items()],
            },
        }
        return await self.rpc(payload)

    async def set_expression(self, expression_file: str, active: bool, fade_time: float = 0.25) -> dict:
        payload = {
            "apiName": API_NAME, "apiVersion": API_VERSION,
            "requestID": f"expr-{int(time.time()*1000)}-{random.randint(100,999)}",
            "messageType": "ExpressionActivationRequest",
            "data": {
                "expressionFile": expression_file,
                "active": bool(active),
                "fadeTime": float(fade_time),
            },
        }
        return await self.rpc(payload)

    async def list_input_parameters(self) -> List[str]:
        payload = {
            "apiName": API_NAME, "apiVersion": API_VERSION,
            "requestID": "ilist",
            "messageType": "InputParameterListRequest",
            "data": {},
        }
        resp = await self.rpc(payload)
        params = []
        for p in (resp.get("data", {}).get("parameters") or []):
            # 官方字段可能是 "name"/"parameterName"
            pid = p.get("name") or p.get("parameterName") or p.get("id")
            if pid:
                params.append(pid)
        return params

    async def create_custom_parameter(
        self,
        name: str,
        default_value: float = 0.85,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ) -> dict:
        payload = {
            "apiName": API_NAME, "apiVersion": API_VERSION,
            "requestID": f"pcreate-{name}",
            "messageType": "ParameterCreationRequest",
            "data": {
                "parameterName": name,
                "explanation": f"Created by {PLUGIN_NAME}",
                "min": float(min_value),
                "max": float(max_value),
                "defaultValue": float(default_value),
            },
        }
        return await self.rpc(payload)


async def connect_vts() -> VTSClient:
    ws = await websockets.connect(VTS_WS_URL, compression=None, ping_interval=None)
    vts = VTSClient(ws)
    ok = await vts.authenticate()
    if not ok:
        await ws.close()
        raise RuntimeError("VTS authenticate failed（请确认 VTS 弹窗已 Allow，并启用插件权限）")
    return vts


# -------------------------------------------------
# Windows 新控制台启动 blink_sender.py
# -------------------------------------------------
async def start_blink_sender_new_console() -> Optional[asyncio.subprocess.Process]:
    here = os.path.dirname(os.path.abspath(__file__))
    sender = BLINK_SENDER_PATH
    if not os.path.isabs(sender):
        sender = os.path.join(here, sender)

    if not os.path.exists(sender):
        print(f"⚠️ 找不到 {sender}，自动启动 blink_sender 失败（你可手动运行）")
        return None

    if platform.system() == "Windows":
        CREATE_NEW_CONSOLE = 0x00000010
        proc = await asyncio.create_subprocess_exec(
            sys.executable, sender,
            cwd=here,
            creationflags=CREATE_NEW_CONSOLE,
        )
        print("👁 已在新控制台启动 blink_sender.py")
        return proc

    proc = await asyncio.create_subprocess_exec(sys.executable, sender, cwd=here)
    print("👁 已启动 blink_sender.py（非 Windows 无新控制台）")
    return proc


# -------------------------------------------------
# LLM：Ollama
# -------------------------------------------------
def ollama_generate(user_text: str) -> str:
    """
    兼容多种 Ollama 接口：
    1) /api/generate（默认）
    2) /api/chat（新接口，messages）
    3) /v1/chat/completions（OpenAI 兼容接口）
    """
    system_hint = (
        "请在回复中用一个情绪标签标注你的语气，标签格式举例："
        "[emo=happy] 或 [emo=sad] 或 [emo=angry] 或 [emo=surprise] 或 [emo=neutral]。"
        "标签可放在开头或结尾，正文正常回答即可。"
    )

    prompt = system_hint + "\n用户：" + user_text + "\n回复："

    def _post(url: str, payload: dict) -> requests.Response:
        return requests.post(url, json=payload, timeout=180)

    # ① /api/generate
    url1 = OLLAMA_URL  # 你配置的通常是 http://127.0.0.1:11434/api/generate
    try:
        r = _post(url1, {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False})
        if r.status_code == 404:
            raise requests.HTTPError("404", response=r)
        r.raise_for_status()
        return (r.json().get("response") or "").strip()
    except requests.HTTPError as e:
        # 如果不是 404，直接抛出（比如 500/401）
        if getattr(e.response, "status_code", None) not in (404,):
            raise

    # ② /api/chat
    try:
        url2 = url1.replace("/api/generate", "/api/chat")
        r = _post(url2, {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_hint},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
        })
        if r.status_code == 404:
            raise requests.HTTPError("404", response=r)
        r.raise_for_status()
        j = r.json()
        # Ollama /api/chat: {"message":{"role":"assistant","content":"..."}}
        msg = (j.get("message") or {}).get("content")
        if msg:
            return str(msg).strip()
    except requests.HTTPError as e:
        if getattr(e.response, "status_code", None) not in (404,):
            raise

    # ③ OpenAI compatible /v1/chat/completions
    url3 = url1.replace("/api/generate", "/v1/chat/completions")
    r = _post(url3, {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_hint},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.7,
    })
    r.raise_for_status()
    j = r.json()
    # OpenAI style: choices[0].message.content
    return str(j["choices"][0]["message"]["content"]).strip()

# -------------------------------------------------
# 情绪关键词提取 + 清理（TTS 忽略）
# -------------------------------------------------
_EMO_ALIASES = {
    "happy": {"happy", "joy", "smile", "开心", "高兴", "喜", "愉快", "兴奋"},
    "sad": {"sad", "down", "cry", "难过", "伤心", "沮丧", "失落"},
    "angry": {"angry", "mad", "rage", "生气", "愤怒", "恼火"},
    "surprise": {"surprise", "wow", "惊讶", "震惊", "意外"},
    "neutral": {"neutral", "calm", "平静", "正常", "中性"},
}

_TAG_PATTERNS = [
    # [emo=happy] / [emotion:angry] / [happy]
    re.compile(r"\[(?:emo|emotion)\s*[:=]\s*([^\]\s]+)\s*\]", re.IGNORECASE),
    re.compile(r"\[([^\]\s]{2,16})\]"),
    # <sad>
    re.compile(r"<\s*([a-zA-Z]{2,16})\s*>"),
    # #happy#
    re.compile(r"#\s*([a-zA-Z]{2,16}|[^\s#]{1,6})\s*#"),
]

def extract_emotions_and_clean(text: str) -> Tuple[str, str, float, Set[str]]:
    """
    返回：
    - clean_text: 给 TTS 的文本（去除标签）
    - emo: happy/sad/angry/surprise/neutral
    - intensity: 0~1（按命中数粗略估计）
    - tags: 命中的原始标签集合
    """
    tags: Set[str] = set()
    for pat in _TAG_PATTERNS:
        for m in pat.finditer(text):
            tags.add(m.group(1).strip())

    # 计算 emo
    hit = {k: 0 for k in _EMO_ALIASES.keys()}
    for t in tags:
        tl = t.lower()
        for emo, vocab in _EMO_ALIASES.items():
            if tl in {x.lower() for x in vocab} or t in vocab:
                hit[emo] += 1

    emo = max(hit.items(), key=lambda kv: kv[1])[0] if any(v > 0 for v in hit.values()) else "neutral"
    intensity = min(1.0, (hit.get(emo, 0) / 2.0))  # 1次≈0.5，2次≈1.0

    # 清理：移除标签本体（不读出来）
    clean = text
    # 移除所有形如 [...] / <...> / #...# 的短标签
    clean = re.sub(r"\[(?:emo|emotion)\s*[:=]\s*[^\]]+\]", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\[[^\]]{1,24}\]", "", clean)
    clean = re.sub(r"<\s*[a-zA-Z]{2,16}\s*>", "", clean)
    clean = re.sub(r"#\s*[^#]{1,24}\s*#", "", clean)
    # 常见多余空行
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()

    if DEBUG_EMO:
        print("[EMO] tags=", tags, "->", emo, intensity)

    return clean, emo, float(intensity), tags


# -------------------------------------------------
# 表情（exp3）互斥
# -------------------------------------------------
async def set_emotion_expression(vts: VTSClient, emotion: str, intensity: float) -> None:
    if not ENABLE_EXPRESSIONS:
        return

    if emotion not in EMO_EXP_FILES:
        emotion = "neutral"

    # 先关所有
    for _emo, files in EMO_EXP_FILES.items():
        for f in files:
            try:
                await vts.set_expression(f, False, fade_time=0.20)
            except Exception:
                pass

    if emotion == "neutral" or not EMO_EXP_FILES.get(emotion) or intensity < 0.15:
        return

    choices = EMO_EXP_FILES[emotion]
    pick = random.choice(choices) if (USE_RANDOM_EXPRESSION and len(choices) > 1) else choices[0]
    try:
        await vts.set_expression(pick, True, fade_time=0.25)
    except Exception:
        pass


# -------------------------------------------------
# TTS + 音频转换
# -------------------------------------------------
async def tts_to_mp3(text: str, mp3_path: str) -> None:
    await edge_tts.Communicate(text, VOICE).save(mp3_path)


def find_ffmpeg() -> Optional[str]:
    p = shutil.which("ffmpeg")
    if p:
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "ffmpeg.exe")
    if os.path.exists(cand):
        return cand
    return None


async def mp3_to_riff_wav(mp3_path: str, wav_path: str) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg。请把 ffmpeg.exe 放脚本目录或加入 PATH。")

    cmd = [ffmpeg, "-y", "-i", mp3_path, "-ac", "1", "-ar", "48000", "-f", "wav", wav_path]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    await proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg 转 wav 失败")

    with open(wav_path, "rb") as f:
        if not f.read(12).startswith(b"RIFF"):
            raise RuntimeError("wav 不是 RIFF（异常）")

def list_audio_devices():
    """打印所有可用音频输出设备"""
    print("\n=== 可用音频输出设备 ===")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_output_channels'] > 0:
            print(f"[{i:2d}] {dev['name']}  (输出通道: {dev['max_output_channels']})")
    print("========================\n")
    print("提示：在 AUDIO_OUTPUT_DEVICE 中填写设备名称关键词或上面的 [数字]")

# -------------------------------------------------
# 播放音频到指定设备（支持虚拟麦克风）
# -------------------------------------------------
async def play_wav(wav_path: str) -> None:
    """播放 wav 到指定音频输出设备（虚拟麦克风）"""
    if platform.system() != "Windows":
        return

    def _play_to_device():
        try:
            # 读取 wav 文件
            if wavfile is None:
                raise RuntimeError("未安装 scipy，无法读取 wav（请 pip install scipy），将降级使用默认设备播放")
            fs, data = wavfile.read(wav_path)
            if data.ndim == 1:
                data = data.reshape(-1, 1)             # 单声道转 2D

            # 查找目标设备
            device_id = None
            devices = sd.query_devices()

            if isinstance(AUDIO_OUTPUT_DEVICE, int):
                # 直接用索引
                if 0 <= AUDIO_OUTPUT_DEVICE < len(devices):
                    device_id = AUDIO_OUTPUT_DEVICE
            elif isinstance(AUDIO_OUTPUT_DEVICE, str) and AUDIO_OUTPUT_DEVICE.strip():
                # 按名称关键词匹配（不区分大小写）
                name_lower = AUDIO_OUTPUT_DEVICE.lower()
                for i, dev in enumerate(devices):
                    if dev['max_output_channels'] > 0 and name_lower in dev['name'].lower():
                        device_id = i
                        break

            print(f"▶️ 播放到音频设备: {devices[device_id]['name'] if device_id is not None else '系统默认'}")

            # 播放
            sd.play(data, samplerate=fs, device=device_id, blocking=True)
            sd.wait()   # 等待播放完成

        except Exception as e:
            print(f"❌ 播放失败: {e}，尝试使用 winsound 默认设备")
            # 降级使用原来的 winsound
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)

    await asyncio.to_thread(_play_to_device)


# -------------------------------------------------
# 音频分析：音量 + 频率 -> 嘴型参数
# -------------------------------------------------
def _autocorr_pitch_hz(frame: np.ndarray, sr: int, fmin=80.0, fmax=350.0) -> float:
    x = frame.astype(np.float32)
    x = x - np.mean(x)
    if np.allclose(x, 0):
        return 0.0

    win = np.hanning(len(x)).astype(np.float32)
    x = x * win
    corr = np.correlate(x, x, mode="full")[len(x) - 1:]

    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    lag_max = min(lag_max, len(corr) - 1)
    if lag_max <= lag_min:
        return 0.0

    seg = corr[lag_min:lag_max]
    if len(seg) < 3:
        return 0.0
    peak = int(np.argmax(seg)) + lag_min
    if corr[peak] <= 0:
        return 0.0
    return float(sr / peak)


def analyze_wav_to_controls(wav_path: str, frame_ms: int = 40) -> Tuple[List[Tuple[float, float]], float]:
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        hop = max(1, int(sr * frame_ms / 1000))

        rms_list: List[float] = []
        pitch_list: List[float] = []

        while True:
            data = wf.readframes(hop)
            if not data:
                break

            if sw == 2:
                s = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sw == 4:
                s = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                s = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0

            if ch > 1:
                s = s.reshape(-1, ch).mean(axis=1)

            rms = float(np.sqrt(np.mean(s * s)) + 1e-9)
            rms_list.append(rms)

            if rms > 0.01:
                pitch = _autocorr_pitch_hz(s, sr, FREQ_MIN, FREQ_MAX)
            else:
                pitch = 0.0
            pitch_list.append(pitch)

        if not rms_list:
            return [(0.0, 0.0)], frame_ms / 1000

        p95 = float(np.percentile(rms_list, 95)) or 1.0
        vol_norm = [min(1.0, r / p95) for r in rms_list]

        def norm_pitch(p: float) -> float:
            if p <= 0:
                return 0.0
            x = (p - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)
            return float(max(0.0, min(1.0, x)))

        pitch_norm = [norm_pitch(p) for p in pitch_list]

        def smooth(seq: List[float], a: float) -> List[float]:
            out: List[float] = []
            prev = 0.0
            for v in seq:
                prev = a * prev + (1 - a) * v
                out.append(prev)
            return out

        vol_s = smooth(vol_norm, a=0.55)
        pit_s = smooth(pitch_norm, a=0.70)

        controls: List[Tuple[float, float]] = []
        for v, p in zip(vol_s, pit_s):
            mouth = max(0.0, min(1.0, v * VOL_GAIN))
            smile = max(0.0, min(1.0, p * SMILE_GAIN))
            controls.append((mouth, smile))

        return controls, frame_ms / 1000


# -------------------------------------------------
# 眨眼：UDP 接收
# -------------------------------------------------
class _BlinkUDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, q: "asyncio.Queue[float]"):
        self.q = q

    def datagram_received(self, data: bytes, addr):
        try:
            msg = data.decode("utf-8", errors="ignore").strip()
        except Exception:
            return
        if not msg.upper().startswith("BLINK:"):
            return
        try:
            strength = float(msg.split(":", 1)[1])
        except Exception:
            return
        strength = max(0.05, min(1.0, strength))
        if DEBUG_BLINK:
            print(f"[UDP] {addr} {msg} -> {strength:.2f}")
        try:
            self.q.put_nowait(strength)
        except Exception:
            pass


async def start_udp_blink_listener(q: "asyncio.Queue[float]"):
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: _BlinkUDPProtocol(q),
        local_addr=(UDP_HOST, UDP_PORT),
    )
    return transport


def _pick_first(existing: Set[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in existing:
            return c
    low = {p.lower(): p for p in existing}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


async def init_eye_tracking_params(vts: VTSClient) -> None:
    """
    1) 尝试找现成 EyeOpen tracking 参数
    2) 找不到就创建 AiEyeOpenL/R（custom tracking 参数）
    """
    global P_EYE_L_OPEN, P_EYE_R_OPEN, USING_CUSTOM_EYE_PARAMS

    try:
        params = await vts.list_input_parameters()
    except Exception as e:
        if DEBUG_BLINK:
            print("[BLINK] InputParameterListRequest failed:", e)
        params = []

    exist = set(params)

    left = _pick_first(exist, EYE_L_CANDIDATES)
    right = _pick_first(exist, EYE_R_CANDIDATES)

    if left and right:
        P_EYE_L_OPEN, P_EYE_R_OPEN = left, right
        USING_CUSTOM_EYE_PARAMS = False
        print(f"👁 使用现成 tracking 眨眼参数：L={P_EYE_L_OPEN} R={P_EYE_R_OPEN}")
        return

    # 创建 custom tracking 参数（若已存在，VTS 可能返回错误但不影响后续使用）
    print("⚠️ 没找到现成 EyeOpen tracking 参数，创建 custom tracking 参数 AiEyeOpenL/AiEyeOpenR ...")
    for name in (CUSTOM_EYE_L, CUSTOM_EYE_R):
        try:
            resp = await vts.create_custom_parameter(
                name,
                default_value=EYE_OPEN_MAX,
                min_value=0.0,
                max_value=1.0,
            )
            if DEBUG_BLINK:
                print("[BLINK][CreateCustom]", name, resp)
        except Exception as e:
            if DEBUG_BLINK:
                print("[BLINK][CreateCustom] exception", name, e)

    P_EYE_L_OPEN, P_EYE_R_OPEN = CUSTOM_EYE_L, CUSTOM_EYE_R
    USING_CUSTOM_EYE_PARAMS = True

    print(f"👁 使用 custom tracking 参数：L={P_EYE_L_OPEN} R={P_EYE_R_OPEN}")
    print("需要你在 VTS 里做一次映射（只需一次）：")
    print("设置(齿轮) -> Model -> 选择你的模型 -> VTS Parameter Setup")
    print(f"  INPUT: {CUSTOM_EYE_L} -> OUTPUT: 你的左眼 Live2D 参数（常见 ParamEyeLOpen）")
    print(f"  INPUT: {CUSTOM_EYE_R} -> OUTPUT: 你的右眼 Live2D 参数（常见 ParamEyeROpen）")
    print("输入范围 [0,1]。如果睁闭反了：勾 Invert 或把 INVERT_EYE=True ")

async def init_hand_params(vts: VTSClient) -> None:
    """
    初始化手部 tracking 输入参数：
    - 优先使用 VTS 已存在的输入参数
    - 若找不到则创建 custom 参数 AiHandLX/AiHandLY/AiHandRX/AiHandRY（-1..1）以及 AiHandWave（0..1）
    你需要在 VTS 里做一次映射（只需一次）。
    """
    global P_HAND_LX, P_HAND_LY, P_HAND_RX, P_HAND_RY, P_HAND_WAVE

    try:
        params = await vts.list_input_parameters()
    except Exception:
        params = []
    exist = set(params)

    P_HAND_LX = _pick_first(exist, HAND_CANDIDATES_LX)
    P_HAND_LY = _pick_first(exist, HAND_CANDIDATES_LY)
    P_HAND_RX = _pick_first(exist, HAND_CANDIDATES_RX)
    P_HAND_RY = _pick_first(exist, HAND_CANDIDATES_RY)
    P_HAND_WAVE = _pick_first(exist, HAND_CANDIDATES_WAVE)

    if any([P_HAND_LX, P_HAND_LY, P_HAND_RX, P_HAND_RY, P_HAND_WAVE]):
        if DEBUG_HAND:
            print("[HAND] Using existing tracking params:",
                  (P_HAND_LX, P_HAND_LY, P_HAND_RX, P_HAND_RY, P_HAND_WAVE))
        return

    print("⚠️ 没找到现成手部 tracking 输入参数，创建 custom tracking 参数 AiHandL*/AiHandR*/AiHandWave ...")

    async def _create(name: str, minv: float, maxv: float, defaultv: float) -> None:
        try:
            await vts.create_custom_parameter(
                name, default_value=float(defaultv), min_value=float(minv), max_value=float(maxv)
            )
        except Exception:
            pass

    await _create(CUSTOM_HAND_LX, -1.0, 1.0, 0.0)
    await _create(CUSTOM_HAND_LY, -1.0, 1.0, 0.0)
    await _create(CUSTOM_HAND_RX, -1.0, 1.0, 0.0)
    await _create(CUSTOM_HAND_RY, -1.0, 1.0, 0.0)
    await _create(CUSTOM_HAND_WAVE, 0.0, 1.0, 0.0)

    P_HAND_LX, P_HAND_LY, P_HAND_RX, P_HAND_RY, P_HAND_WAVE = (
        CUSTOM_HAND_LX, CUSTOM_HAND_LY, CUSTOM_HAND_RX, CUSTOM_HAND_RY, CUSTOM_HAND_WAVE
    )

    print(f"🤚 使用 custom hand tracking 参数：L=({P_HAND_LX},{P_HAND_LY}) R=({P_HAND_RX},{P_HAND_RY}) wave={P_HAND_WAVE}")
    print("需要你在 VTS 里做一次映射（只需一次）：")
    print("设置(齿轮) -> Model -> 选择你的模型 -> VTS Parameter Setup")
    print(f"  INPUT: {CUSTOM_HAND_LX} -> OUTPUT: 左手/左臂 X（例如 ParamArmLX/ParamHandLX 等）")
    print(f"  INPUT: {CUSTOM_HAND_LY} -> OUTPUT: 左手/左臂 Y")
    print(f"  INPUT: {CUSTOM_HAND_RX} -> OUTPUT: 右手/右臂 X")
    print(f"  INPUT: {CUSTOM_HAND_RY} -> OUTPUT: 右手/右臂 Y")
    print(f"  INPUT: {CUSTOM_HAND_WAVE} -> OUTPUT: 挥手/强调参数（可选）")
    print("手部输入建议范围 [-1,1]；挥手输入 [0,1]；方向反了可勾 Invert。")


async def init_motion_params(vts: VTSClient) -> None:
    """
    初始化“待机随机头部/眼球”可注入参数：

    优先使用 VTS 已存在的 tracking 输入参数（InputParameterListRequest 能看到的）。
    若找不到（很多模型/配置确实没有 FaceAngleX/EyeX 之类的输入），则自动创建
    自定义 tracking 参数：
      - AiHeadX / AiHeadY / AiHeadZ   （范围 -10..10）
      - AiGazeX / AiGazeY             （范围 -1..1）

    你需要在 VTS 里把这些 INPUT 映射到模型的 Live2D 参数（只需一次）：
      AiHeadX -> ParamAngleX（或模型对应的头部X）
      AiHeadY -> ParamAngleY
      AiHeadZ -> ParamAngleZ
      AiGazeX -> ParamEyeBallX（或模型对应的眼球X）
      AiGazeY -> ParamEyeBallY
    """
    global P_HEAD_X, P_HEAD_Y, P_HEAD_Z, P_GAZE_X, P_GAZE_Y

    try:
        params = await vts.list_input_parameters()
    except Exception:
        params = []
    exist = set(params)

    # 1) 先尝试找现成的 tracking 输入参数
    P_HEAD_X = _pick_first(exist, HEAD_CANDIDATES_X)
    P_HEAD_Y = _pick_first(exist, HEAD_CANDIDATES_Y)
    P_HEAD_Z = _pick_first(exist, HEAD_CANDIDATES_Z)
    P_GAZE_X = _pick_first(exist, EYE_CANDIDATES_X)
    P_GAZE_Y = _pick_first(exist, EYE_CANDIDATES_Y)

    if any([P_HEAD_X, P_HEAD_Y, P_HEAD_Z, P_GAZE_X, P_GAZE_Y]):
        if DEBUG_MOTION:
            print("[MOTION] Using existing tracking params:",
                  "head=", (P_HEAD_X, P_HEAD_Y, P_HEAD_Z),
                  "gaze=", (P_GAZE_X, P_GAZE_Y))
        return

    # 2) 没有现成参数：创建自定义 tracking 参数（若已存在，VTS 会报错但可忽略）
    print("⚠️ 没找到现成头/眼 tracking 输入参数，创建 custom tracking 参数 AiHeadX/Y/Z + AiGazeX/Y ...")

    async def _create(name: str, minv: float, maxv: float, defaultv: float = 0.0) -> None:
        try:
            await vts.create_custom_parameter(
                name,
                default_value=float(defaultv),
                min_value=float(minv),
                max_value=float(maxv),
            )
        except Exception:
            # 已存在/创建失败都不致命：后续仍尝试使用这个名字
            pass

    await _create(CUSTOM_HEAD_X, -10.0, 10.0, 0.0)
    await _create(CUSTOM_HEAD_Y, -10.0, 10.0, 0.0)
    await _create(CUSTOM_HEAD_Z, -10.0, 10.0, 0.0)
    await _create(CUSTOM_GAZE_X, -1.0, 1.0, 0.0)
    await _create(CUSTOM_GAZE_Y, -1.0, 1.0, 0.0)

    P_HEAD_X, P_HEAD_Y, P_HEAD_Z = CUSTOM_HEAD_X, CUSTOM_HEAD_Y, CUSTOM_HEAD_Z
    P_GAZE_X, P_GAZE_Y = CUSTOM_GAZE_X, CUSTOM_GAZE_Y

    print(f"🧠 使用 custom motion tracking 参数：head=({P_HEAD_X},{P_HEAD_Y},{P_HEAD_Z}) gaze=({P_GAZE_X},{P_GAZE_Y})")
    print("需要你在 VTS 里做一次映射（只需一次）：")
    print("设置(齿轮) -> Model -> 选择你的模型 -> VTS Parameter Setup")
    print(f"  INPUT: {CUSTOM_HEAD_X} -> OUTPUT: 头部左右（常见 ParamAngleX）")
    print(f"  INPUT: {CUSTOM_HEAD_Y} -> OUTPUT: 头部上下（常见 ParamAngleY）")
    print(f"  INPUT: {CUSTOM_HEAD_Z} -> OUTPUT: 头部歪头（常见 ParamAngleZ）")
    print(f"  INPUT: {CUSTOM_GAZE_X} -> OUTPUT: 眼球X（常见 ParamEyeBallX）")
    print(f"  INPUT: {CUSTOM_GAZE_Y} -> OUTPUT: 眼球Y（常见 ParamEyeBallY）")
    print("头部输入建议范围 [-10,10]；眼球输入建议范围 [-1,1]；方向反了可勾 Invert。")


def _eye_map(norm_0_1: float) -> float:
    """把 0~1 的归一化值映射到 [EYE_OPEN_MIN, EYE_OPEN_MAX]，避免瞪眼。"""
    x = max(0.0, min(1.0, float(norm_0_1)))
    return EYE_OPEN_MIN + x * (EYE_OPEN_MAX - EYE_OPEN_MIN)



async def do_blink(vts: VTSClient, strength: float) -> None:
    global _last_blink_ts

    if not P_EYE_L_OPEN or not P_EYE_R_OPEN:
        if DEBUG_BLINK:
            print("[BLINK] EyeOpen tracking params not ready, skip")
        return

    async with blink_lock:
        now = time.perf_counter()
        if now - _last_blink_ts < BLINK_COOLDOWN_SEC:
            return
        _last_blink_ts = now

        # strength 越大，闭得越狠：close_norm 越小
        close_norm = max(0.0, 1.0 - float(strength))  # 0=闭, 1=开（归一化）
        val_close = _eye_map(close_norm)
        if INVERT_EYE:
            val_close = 1.0 - val_close

        try:
            await vts.inject({P_EYE_L_OPEN: val_close, P_EYE_R_OPEN: val_close}, mode="Set")
        except Exception:
            return

        await asyncio.sleep(0.05)

        # 回到自然睁眼（映射后就是 EYE_OPEN_MAX）
        for x in BLINK_OPEN_FRAMES:
            val_open = _eye_map(x)
            if INVERT_EYE:
                val_open = 1.0 - val_open
            try:
                await vts.inject({P_EYE_L_OPEN: val_open, P_EYE_R_OPEN: val_open}, mode="Set")
            except Exception:
                break
            await asyncio.sleep(0.04)



class State:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.emotion = "neutral"
        self.intensity = 0.0
        self.is_speaking = False

    async def set_emotion(self, emo: str, intensity: float):
        async with self.lock:
            self.emotion = emo
            self.intensity = float(max(0.0, min(1.0, intensity)))

    async def set_speaking(self, s: bool):
        async with self.lock:
            self.is_speaking = bool(s)

    async def get(self):
        async with self.lock:
            return self.emotion, self.intensity, self.is_speaking


async def blink_loop(vts: VTSClient, st: State, blink_q: "asyncio.Queue[float]"):
    # 确保启动时先睁眼
    if P_EYE_L_OPEN and P_EYE_R_OPEN:
        open_val = _eye_map(1.0)
        if INVERT_EYE:
            open_val = 1.0 - open_val
        await vts.inject({P_EYE_L_OPEN: open_val, P_EYE_R_OPEN: open_val}, mode="Set")

    while True:
        emo, inten, speaking = await st.get()

        base_min, base_max = 2.5, 5.5
        speed_up = 0.0
        if emo in ("angry", "surprise"):
            speed_up += 0.7 * inten
        if speaking:
            speed_up += 0.35
        interval = random.uniform(base_min, base_max) * (1.0 / (1.0 + speed_up))
        interval = max(0.6, interval)

        strength: Optional[float] = None
        if ENABLE_UDP_BLINK:
            try:
                strength = await asyncio.wait_for(blink_q.get(), timeout=interval)
            except asyncio.TimeoutError:
                strength = None
        else:
            await asyncio.sleep(interval)

        if strength is not None:
            if BLINK_QUEUE_FLUSH:
                last = strength
                while True:
                    try:
                        last = blink_q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                strength = last
            await do_blink(vts, strength)
            continue

        if not ENABLE_NATURAL_BLINK:
            continue

        nat_strength = random.uniform(0.45, 0.60) + 0.10 * inten + (0.05 if speaking else 0.0)
        nat_strength = max(0.35, min(0.95, nat_strength))
        await do_blink(vts, nat_strength)



# -------------------------------------------------
# 待机眼睛保持（防止 custom 参数回到默认值 1.0 导致“瞪眼”）
# -------------------------------------------------
async def idle_eye_keeper(vts: VTSClient, st: State):
    """
    有些情况下（尤其是 custom tracking 参数），如果你不持续注入，
    VTS 可能会把输入参数缓慢回到默认值（比如 1.0），导致突然“瞪眼”。
    这里在非眨眼期间，周期性把眼睛保持在 EYE_OPEN_MAX（或 INVERT 后的等效值）。
    """
    if not P_EYE_L_OPEN or not P_EYE_R_OPEN:
        return

    while True:
        if blink_lock.locked():
            await asyncio.sleep(0.10)
            continue
        # 说话/待机都保持自然开眼，不写到 1.0
        open_val = _eye_map(1.0)
        if INVERT_EYE:
            open_val = 1.0 - open_val
        try:
            await vts.inject({P_EYE_L_OPEN: open_val, P_EYE_R_OPEN: open_val}, mode="Set")
        except Exception:
            pass
        await asyncio.sleep(0.35)

# -------------------------------------------------
# 待机随机头部/眼球转动（说话时暂停，低频注入，避免抢占）
# -------------------------------------------------
def _rand(rng):
    return random.uniform(float(rng[0]), float(rng[1]))

def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

async def idle_motion_loop(vts: VTSClient, st: State):
    """更像真人的待机头部/眼球运动：
    - 眼球先动、头部后跟随（gaze lead head）
    - 注视停留（dwell）+ 微小眼跳（micro-saccade）
    - 呼吸式微摆（breathing sway）
    """
    if not ENABLE_IDLE_MOTION:
        return
    if not any([P_HEAD_X, P_HEAD_Y, P_HEAD_Z, P_GAZE_X, P_GAZE_Y]):
        return

    ids = [p for p in [P_HEAD_X, P_HEAD_Y, P_HEAD_Z, P_GAZE_X, P_GAZE_Y] if p]
    cur = {k: 0.0 for k in ids}

    def _absmax(rng):
        return max(abs(float(rng[0])), abs(float(rng[1])), 1e-6)

    eye_x_max = _absmax(EYE_X_RANGE)
    eye_y_max = _absmax(EYE_Y_RANGE)
    head_x_max = _absmax(HEAD_X_RANGE)
    head_y_max = _absmax(HEAD_Y_RANGE)

    while True:
        # 说话时暂停（避免影响嘴型/眨眼）
        _, _, speaking = await st.get()
        if speaking:
            await asyncio.sleep(0.25)
            continue

        # 眨眼期间让位
        if blink_lock.locked():
            await asyncio.sleep(0.12)
            continue

        # 呼吸式微摆：极小幅度、低频
        t_now = time.perf_counter()
        breath = float(np.sin(t_now * 2.0 * np.pi * 0.18))  # ~5.5s/周期
        bx = breath * 0.22
        by = breath * 0.16

        # 先定“注视点”（眼球目标）
        gx = _rand(EYE_X_RANGE) if P_GAZE_X else 0.0
        gy = _rand(EYE_Y_RANGE) if P_GAZE_Y else 0.0

        # 头部跟随比例：越大越“爱扭头”
        follow = random.uniform(0.35, 0.55)

        tgt = {}
        if P_GAZE_X: tgt[P_GAZE_X] = gx
        if P_GAZE_Y: tgt[P_GAZE_Y] = gy

        # 头部：跟随眼睛 + 少量独立变化 + 呼吸
        if P_HEAD_X:
            tgt[P_HEAD_X] = (gx / eye_x_max) * head_x_max * follow + _rand((-0.35, 0.35)) + bx
        if P_HEAD_Y:
            tgt[P_HEAD_Y] = (gy / eye_y_max) * head_y_max * follow + _rand((-0.28, 0.28)) + by
        if P_HEAD_Z:
            tgt[P_HEAD_Z] = _rand(HEAD_Z_RANGE) * 0.35  # 轻微歪头

        # 平滑过渡到目标
        ease = random.uniform(float(MOTION_EASE_SEC[0]), float(MOTION_EASE_SEC[1]))
        interval = random.uniform(float(MOTION_INTERVAL_SEC[0]), float(MOTION_INTERVAL_SEC[1]))
        steps = max(1, int(ease * MOTION_HZ))
        dt = ease / steps

        start = {k: cur.get(k, 0.0) for k in ids}

        for i in range(steps):
            _, _, speaking2 = await st.get()
            if speaking2 or blink_lock.locked():
                break

            t01 = _smoothstep((i + 1) / steps)
            payload = {}
            for pid in ids:
                target = float(tgt.get(pid, start[pid]))
                val = float(start[pid] + (target - start[pid]) * t01)
                cur[pid] = val
                payload[pid] = val

            if payload:
                try:
                    await vts.inject(payload, mode="Set")
                except Exception:
                    pass
            await asyncio.sleep(dt)

        # 注视停留期间：做几次微小眼跳（micro-saccade）
        end_t = time.perf_counter() + max(0.3, interval)
        while time.perf_counter() < end_t:
            _, _, speaking2 = await st.get()
            if speaking2 or blink_lock.locked():
                break

            await asyncio.sleep(random.uniform(0.25, 0.55))

            # 低概率触发，避免太抽搐
            if random.random() > 0.65:
                continue

            dx = random.uniform(-0.045, 0.045)
            dy = random.uniform(-0.030, 0.030)

            payload = {}
            if P_GAZE_X:
                payload[P_GAZE_X] = float(cur.get(P_GAZE_X, 0.0) + dx)
            if P_GAZE_Y:
                payload[P_GAZE_Y] = float(cur.get(P_GAZE_Y, 0.0) + dy)

            if payload:
                try:
                    await vts.inject(payload, mode="Set")
                except Exception:
                    pass

                await asyncio.sleep(random.uniform(0.04, 0.08))

                payload2 = {}
                if P_GAZE_X: payload2[P_GAZE_X] = float(cur.get(P_GAZE_X, 0.0))
                if P_GAZE_Y: payload2[P_GAZE_Y] = float(cur.get(P_GAZE_Y, 0.0))
                try:
                    await vts.inject(payload2, mode="Set")
                except Exception:
                    pass


# -------------------------------------------------
# 待机/说话 手部运动（独立低频注入，避免抢占）
# -------------------------------------------------
async def idle_hand_loop(vts: VTSClient, st: State):
    if not ENABLE_HAND_MOTION:
        return
    if not any([P_HAND_LX, P_HAND_LY, P_HAND_RX, P_HAND_RY, P_HAND_WAVE]):
        return

    def rand_range(rng):
        return random.uniform(float(rng[0]), float(rng[1]))

    def smoothstep(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    cur = {p: 0.0 for p in [P_HAND_LX, P_HAND_LY, P_HAND_RX, P_HAND_RY, P_HAND_WAVE] if p}
    next_wave_at = time.perf_counter() + random.uniform(*WAVE_COOLDOWN_SEC)

    while True:
        emo, inten, speaking = await st.get()

        if blink_lock.locked():
            await asyncio.sleep(0.10)
            continue

        now = time.perf_counter()

        # 说话时偶尔挥手/强调（需要映射 AiHandWave 才生效）
        if speaking and P_HAND_WAVE and now >= next_wave_at:
            dur = max(0.35, float(WAVE_DURATION_SEC))
            steps = max(1, int(dur * HAND_HZ))
            dt = dur / steps

            for i in range(steps):
                _, _, speaking2 = await st.get()
                if not speaking2:
                    break
                t = i / max(1, steps - 1)
                w = (t * 2.0) if t < 0.5 else (2.0 - t * 2.0)  # 三角波 0..1..0
                w = max(0.0, min(1.0, w))
                w *= max(0.25, min(1.0, SPEAK_GESTURE_GAIN + 0.25 * inten))
                try:
                    await vts.inject({P_HAND_WAVE: w}, mode="Set")
                except Exception:
                    pass
                await asyncio.sleep(dt)

            try:
                await vts.inject({P_HAND_WAVE: 0.0}, mode="Set")
            except Exception:
                pass

            next_wave_at = time.perf_counter() + random.uniform(*WAVE_COOLDOWN_SEC)
            await asyncio.sleep(0.2)
            continue

        # 轻微摆动（说话时稍大）
        scale = 1.0 + (0.35 if speaking else 0.0) + 0.20 * float(inten)

        tgt = {}
        if P_HAND_LX: tgt[P_HAND_LX] = rand_range(HAND_X_RANGE) * scale
        if P_HAND_LY: tgt[P_HAND_LY] = rand_range(HAND_Y_RANGE) * scale
        if P_HAND_RX: tgt[P_HAND_RX] = rand_range(HAND_X_RANGE) * scale
        if P_HAND_RY: tgt[P_HAND_RY] = rand_range(HAND_Y_RANGE) * scale

        ease = random.uniform(float(HAND_EASE_SEC[0]), float(HAND_EASE_SEC[1]))
        interval = random.uniform(float(HAND_INTERVAL_SEC[0]), float(HAND_INTERVAL_SEC[1]))
        steps = max(1, int(ease * HAND_HZ))
        dt = ease / steps

        for i in range(steps):
            t = smoothstep((i + 1) / steps)
            payload = {}
            for pid, target in tgt.items():
                cur[pid] = cur[pid] + (target - cur[pid]) * t
                payload[pid] = cur[pid]
            if payload:
                try:
                    await vts.inject(payload, mode="Set")
                except Exception:
                    pass
            await asyncio.sleep(dt)

        await asyncio.sleep(max(0.25, interval))

# -------------------------------------------------
# 嘴型驱动（VTS Voice params）
# -------------------------------------------------
async def drive_voice_params(vts: VTSClient, controls: List[Tuple[float, float]], dt: float, st: State):
    await st.set_speaking(True)
    try:
        start = time.perf_counter()
        i = 0
        n = len(controls)

        while i < n:
            target_t = start + i * dt
            now = time.perf_counter()

            # 落后就跳帧追上
            if now > target_t + 0.5 * dt:
                behind = int((now - target_t) // dt)
                i += max(1, behind)
                if i >= n:
                    break

            mouth, smile = controls[i]
            await vts.inject({P_MOUTH_OPEN: mouth, P_MOUTH_SMILE: smile}, mode="Set")

            now2 = time.perf_counter()
            sleep_for = (start + (i + 1) * dt) - now2
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            i += 1

        await vts.inject({P_MOUTH_OPEN: 0.0, P_MOUTH_SMILE: 0.0}, mode="Set")
    finally:
        await st.set_speaking(False)


# -------------------------------------------------
# 主流程
# -------------------------------------------------
async def main():
    # 显示可用音频设备（方便设置虚拟麦克风）
    list_audio_devices()
    print("🔌 Connecting to VTube Studio...")
    vts = await connect_vts()
    # 第二条 WS 连接：专门用于待机随机动作，避免抢占嘴型/眨眼的注入锁
    vts_motion = await connect_vts()
    print("🎉 Connected & authed.")

    # 初始化眨眼 tracking 参数
    await init_eye_tracking_params(vts)

    await init_motion_params(vts)
    await init_motion_params(vts_motion)
    await init_hand_params(vts_motion)

    st = State()
    blink_q: "asyncio.Queue[float]" = asyncio.Queue(maxsize=64)

    udp_transport = None
    if ENABLE_UDP_BLINK:
        udp_transport = await start_udp_blink_listener(blink_q)
        print(f"👁 UDP blink listener on {UDP_HOST}:{UDP_PORT} (期待 BLINK:0.xx)")

    if AUTO_START_BLINK_SENDER and ENABLE_UDP_BLINK:
        await start_blink_sender_new_console()

    # 后台任务
    tasks = [
        asyncio.create_task(blink_loop(vts, st, blink_q)),
        asyncio.create_task(idle_eye_keeper(vts, st)),
        asyncio.create_task(idle_motion_loop(vts_motion, st)),
        asyncio.create_task(idle_hand_loop(vts_motion, st)),
    ]

    # 启动自检：尝试一次眨眼（如果你还没做映射，注入会成功但模型可能不动）
    await asyncio.sleep(1.0)
    await do_blink(vts, 0.55)

    print("\n🎤 Ready. 输入 quit 退出。\n")
    try:
        while True:
            user = (await asyncio.to_thread(input, "你：")).strip()

            # 退出
            if user.lower() in ["quit", "exit", "q", "退出"]:
                print("👋 已退出")
                break

            # 运行时切换音频输出设备：device <索引|关键词>
            if user.lower().startswith("device "):
                new_dev = user[7:].strip()
                global AUDIO_OUTPUT_DEVICE
                try:
                    AUDIO_OUTPUT_DEVICE = int(new_dev)  # 数字 -> 设备索引
                except ValueError:
                    AUDIO_OUTPUT_DEVICE = new_dev       # 字符串 -> 名称关键词
                print(f"🎧 音频输出设备已切换为：{AUDIO_OUTPUT_DEVICE}")
                continue
            ai_raw = ollama_generate(user)
            print("她：", ai_raw)

            # 2) 提取关键词/标签 + 清理给 TTS 的文本
            tts_text, emo, inten, _tags = extract_emotions_and_clean(ai_raw)
            await st.set_emotion(emo, inten)

            # 3) 表情联动
            await set_emotion_expression(vts, emo, inten)

            # 4) TTS + wav
            if not tts_text.strip():
                # 兜底：万一全是标签
                tts_text = "嗯。"
            await tts_to_mp3(tts_text, TMP_MP3)
            await mp3_to_riff_wav(TMP_MP3, TMP_WAV)

            # 5) 预分析 -> controls
            controls, dt = analyze_wav_to_controls(TMP_WAV, frame_ms=FRAME_MS)

            # 6) 播放 + 注入口型（并行）
            await asyncio.gather(
                play_wav(TMP_WAV),
                drive_voice_params(vts, controls, dt, st),
            )
    finally:
        for t in tasks:
            t.cancel()
        if udp_transport is not None:
            udp_transport.close()
        await vts.ws.close()
        try:
            await vts_motion.ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
