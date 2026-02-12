"""AI VTuber runtime with Ollama + Live2D(VTS) + TTS pipeline.

该脚本负责以下核心流程：
1) 从 Ollama 获取文本回复并提取情绪标签。
2) 使用 Edge TTS 合成音频并播放。
3) 分析音频驱动口型参数。
4) 通过 VTube Studio API 驱动表情、眨眼、头眼和手部动作。
"""

# ===== 可选：切换到其他 LLM 供应商（示例，默认注释）=====
# 1) OpenAI：
#    - 安装依赖：pip install openai
#    - 设置环境变量：
#      set OPENAI_API_KEY=你的key        (Windows CMD)
#      $env:OPENAI_API_KEY="你的key"      (PowerShell)
#    - 可参考如下函数替换 ollama_generate：
#
#      # from openai import OpenAI
#      # def openai_generate(user_text: str) -> str:
#      #     client = OpenAI()
#      #     resp = client.chat.completions.create(
#      #         model="gpt-4o-mini",
#      #         messages=[
#      #             {"role": "system", "content": "请在回复中带上[emo=happy/sad/angry/surprise/neutral]标签"},
#      #             {"role": "user", "content": user_text},
#      #         ],
#      #         temperature=0.7,
#      #     )
#      #     return (resp.choices[0].message.content or "").strip()
#
# 2) xAI（兼容 OpenAI SDK 的接口）：
#    - 安装依赖：pip install openai
#    - 设置环境变量：
#      set XAI_API_KEY=你的key           (Windows CMD)
#      $env:XAI_API_KEY="你的key"         (PowerShell)
#    - 可参考如下函数替换 ollama_generate：
#
#      # from openai import OpenAI
#      # def xai_generate(user_text: str) -> str:
#      #     client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
#      #     resp = client.chat.completions.create(
#      #         model="grok-2-latest",
#      #         messages=[
#      #             {"role": "system", "content": "请在回复中带上[emo=happy/sad/angry/surprise/neutral]标签"},
#      #             {"role": "user", "content": user_text},
#      #         ],
#      #         temperature=0.7,
#      #     )
#      #     return (resp.choices[0].message.content or "").strip()
#
# 3) 接入方式：
#    - 保留 extract_emotions_and_clean() 不变。
#    - 在 main() 中把 ai_raw = ollama_generate(user) 改为你上面的 *_generate(user)。
# =====================================================

import asyncio
import json
import os
import platform
import random
import re
import shutil
import sys
import time
import wave
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import requests
import websockets
import edge_tts
import sounddevice as sd

try:
    from scipy.io import wavfile
except Exception:
    wavfile = None

if platform.system() == "Windows":
    import winsound


## 基础连接配置
VTS_WS_URL = "ws://127.0.0.1:8001"
API_NAME, API_VERSION = "VTubeStudioPublicAPI", "1.0"
PLUGIN_NAME, PLUGIN_AUTHOR = "AI_VTuber_Full", "User"
TOKEN_FILE = "vts_token_ai_vtuber.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "vtuber:latest"

VOICE = "zh-CN-XiaoxiaoNeural"
TMP_MP3, TMP_WAV = "tts.mp3", "tts.wav"


## 音频输出配置：可用 None / 设备索引 / 设备关键字
AUDIO_OUTPUT_DEVICE: Optional[object] = "Voicemeeter Input"


P_MOUTH_OPEN = "VoiceVolumePlusMouthOpen"
P_MOUTH_SMILE = "VoiceFrequencyPlusMouthSmile"
MOUTH_INJECT_PRIMARY_MODE = "Set"
ENABLE_MOUTH_OPEN_PARAM = True
ENABLE_MOUTH_SMILE_PARAM = True


EYE_L_CANDIDATES = ["EyeOpenLeft","EyeOpenL","EyeLeftOpen","EyeOpen_L","EyeBlinkLeft","EyeBlinkL","EyeL","EyeOpen"]
EYE_R_CANDIDATES = ["EyeOpenRight","EyeOpenR","EyeRightOpen","EyeOpen_R","EyeBlinkRight","EyeBlinkR","EyeR","EyeOpen"]
CUSTOM_EYE_L, CUSTOM_EYE_R = "AiEyeOpenL", "AiEyeOpenR"


HEAD_CANDIDATES_X = ["FaceAngleX","HeadAngleX","AngleX","RotationX"]
HEAD_CANDIDATES_Y = ["FaceAngleY","HeadAngleY","AngleY","RotationY"]
HEAD_CANDIDATES_Z = ["FaceAngleZ","HeadAngleZ","AngleZ","RotationZ"]
GAZE_CANDIDATES_X = ["EyeX","EyeBallX","GazeX","LookX"]
GAZE_CANDIDATES_Y = ["EyeY","EyeBallY","GazeY","LookY"]
CUSTOM_HEAD_X, CUSTOM_HEAD_Y, CUSTOM_HEAD_Z = "AiHeadX","AiHeadY","AiHeadZ"
CUSTOM_GAZE_X, CUSTOM_GAZE_Y = "AiGazeX","AiGazeY"


HAND_CANDIDATES_LX = ["HandLX","LeftHandX","ArmLX","LeftArmX","LArmX","LHandX"]
HAND_CANDIDATES_LY = ["HandLY","LeftHandY","ArmLY","LeftArmY","LArmY","LHandY"]
HAND_CANDIDATES_RX = ["HandRX","RightHandX","ArmRX","RightArmX","RArmX","RHandX"]
HAND_CANDIDATES_RY = ["HandRY","RightHandY","ArmRY","RightArmY","RArmY","RHandY"]
HAND_CANDIDATES_WAVE = ["HandWave","Wave","Gesture","HandGesture"]
CUSTOM_HAND_LX, CUSTOM_HAND_LY, CUSTOM_HAND_RX, CUSTOM_HAND_RY, CUSTOM_HAND_WAVE =\
    "AiHandLX","AiHandLY","AiHandRX","AiHandRY","AiHandWave"


INVERT_EYE = False
EYE_OPEN_MIN, EYE_OPEN_MAX = 0.05, 0.65
BLINK_COOLDOWN_SEC = 0.35
BLINK_OPEN_FRAMES = (0.25, 0.55, 0.80, 1.0)
BLINK_QUEUE_FLUSH = True


ENABLE_UDP_BLINK = True
UDP_HOST, UDP_PORT = "127.0.0.1", 49721
ENABLE_NATURAL_BLINK = True
AUTO_START_BLINK_SENDER = True
BLINK_SENDER_PATH = "blink_sender.py"


FRAME_MS = 25
VOL_GAIN, SMILE_GAIN = 1.20, 1.00
FREQ_MIN, FREQ_MAX = 80.0, 350.0


MOUTH_MIN_OPEN = 0.12
RMS_FLOOR = 0.002
PITCH_RMS_GATE = 0.006


ENABLE_EXPRESSIONS = True
EMO_EXP_FILES = {
    "happy": ["xinxin.exp3.json", "bq3.exp3.json"],
    "sad": ["ku.exp3.json", "st.exp3.json"],
    "angry": ["ga.exp3.json"],
    "surprise": ["sq.exp3.json"],
    "neutral": [],
}
USE_RANDOM_EXPRESSION = True
AUTO_RESET_EXPRESSION_AFTER_REPLY = True


ENABLE_IDLE_MOTION = True
MOTION_HZ = 20
MOTION_INTERVAL_SEC = (3.0, 6.0)
MOTION_EASE_SEC = (0.60, 1.40)
HEAD_X_RANGE, HEAD_Y_RANGE, HEAD_Z_RANGE = (-7.0, 7.0), (-5.0, 5.0), (-4.0, 4.0)
EYE_X_RANGE, EYE_Y_RANGE = (-0.45, 0.45), (-0.30, 0.30)
SPEAK_MOTION_GAIN = 0.72
SPEAK_MOTION_HZ = 26
SPEAK_NOD_PROB = 0.33


ENABLE_HAND_MOTION = True
HAND_HZ = 10
HAND_INTERVAL_SEC = (2.2, 4.8)
HAND_EASE_SEC = (0.45, 1.05)
HAND_X_RANGE, HAND_Y_RANGE = (-0.35, 0.35), (-0.20, 0.20)
SPEAK_GESTURE_GAIN = 0.55
WAVE_DURATION_SEC = 0.75
WAVE_COOLDOWN_SEC = (2.0, 5.0)


def load_token() -> Optional[str]:
    """Load cached VTS auth token from disk."""
    if not os.path.exists(TOKEN_FILE):
        return None
    try:
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("token")
    except Exception:
        return None

def save_token(token: str) -> None:
    """Persist VTS auth token to disk."""
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        json.dump({"token": token}, f, ensure_ascii=False, indent=2)

def find_ffmpeg() -> Optional[str]:
    """Find ffmpeg from PATH or local script directory."""
    p = shutil.which("ffmpeg")
    if p:
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "ffmpeg.exe")
    return cand if os.path.exists(cand) else None

def list_audio_devices() -> None:
    """Print available audio output devices for runtime selection."""
    print("\n=== 可用音频输出设备 ===")
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_output_channels", 0) > 0:
            print(f"[{i:2d}] {dev['name']}  (输出通道:{dev['max_output_channels']})")
    print("========================\n提示：AUDIO_OUTPUT_DEVICE 设索引或名称关键词；运行中输入 device <索引|关键词>\n")

def _pick_first(existing: Set[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in existing:
            return c
    low = {p.lower(): p for p in existing}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

def _eye_map(norm_0_1: float) -> float:
    x = max(0.0, min(1.0, float(norm_0_1)))
    return EYE_OPEN_MIN + x * (EYE_OPEN_MAX - EYE_OPEN_MIN)

def resolve_output_device_id(device_pref: Optional[object]) -> Optional[int]:
    """Resolve user-configured output device to a sounddevice index."""
    devices = sd.query_devices()
    if isinstance(device_pref, int):
        return device_pref if 0 <= device_pref < len(devices) else None
    if isinstance(device_pref, str) and device_pref.strip():
        key = device_pref.lower()
        for idx, dev in enumerate(devices):
            if dev.get("max_output_channels", 0) > 0 and key in dev["name"].lower():
                return idx
    return None


def is_vts_api_error(resp: dict) -> bool:
    """Return True when VTS response indicates an API-level error."""
    mt = str(resp.get("messageType", ""))
    return mt.lower() == "apierror"


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
            resp = await self.rpc({
                "apiName": API_NAME, "apiVersion": API_VERSION,
                "requestID": "token",
                "messageType": "AuthenticationTokenRequest",
                "data": {"pluginName": PLUGIN_NAME, "pluginDeveloper": PLUGIN_AUTHOR},
            })
            token = (resp.get("data") or {}).get("authenticationToken")
            if not token:
                return False
            save_token(token)
            print("token saved. VTS 弹窗请点 Allow/允许")

        resp = await self.rpc({
            "apiName": API_NAME, "apiVersion": API_VERSION,
            "requestID": "auth",
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": PLUGIN_NAME,
                "pluginDeveloper": PLUGIN_AUTHOR,
                "authenticationToken": token,
            },
        })
        return bool((resp.get("data") or {}).get("authenticated", False))

    async def inject(self, params: Dict[str, float], mode: str = "Set") -> dict:
        return await self.rpc({
            "apiName": API_NAME, "apiVersion": API_VERSION,
            "requestID": f"inject-{int(time.time()*1000)}-{random.randint(100,999)}",
            "messageType": "InjectParameterDataRequest",
            "data": {
                "mode": mode,
                "parameterValues": [{"id": k, "value": float(v)} for k, v in params.items()],
            },
        })

    async def list_input_parameters(self) -> List[str]:
        resp = await self.rpc({
            "apiName": API_NAME, "apiVersion": API_VERSION,
            "requestID": "ilist",
            "messageType": "InputParameterListRequest",
            "data": {},
        })
        out: List[str] = []
        for p in (resp.get("data", {}).get("parameters") or []):
            pid = p.get("name") or p.get("parameterName") or p.get("id")
            if pid:
                out.append(pid)
        return out

    async def create_custom_parameter(self, name: str, default: float, minv: float, maxv: float) -> None:
        try:
            await self.rpc({
                "apiName": API_NAME, "apiVersion": API_VERSION,
                "requestID": f"pcreate-{name}",
                "messageType": "ParameterCreationRequest",
                "data": {
                    "parameterName": name,
                    "explanation": f"Created by {PLUGIN_NAME}",
                    "min": float(minv),
                    "max": float(maxv),
                    "defaultValue": float(default),
                },
            })
        except Exception:
            pass

    async def set_expression(self, expression_file: str, active: bool, fade: float = 0.25) -> None:
        try:
            await self.rpc({
                "apiName": API_NAME, "apiVersion": API_VERSION,
                "requestID": f"expr-{int(time.time()*1000)}-{random.randint(100,999)}",
                "messageType": "ExpressionActivationRequest",
                "data": {"expressionFile": expression_file, "active": bool(active), "fadeTime": float(fade)},
            })
        except Exception:
            pass

async def connect_vts() -> VTSClient:
    ws = await websockets.connect(VTS_WS_URL, compression=None, ping_interval=None)
    vts = VTSClient(ws)
    if not await vts.authenticate():
        await ws.close()
        raise RuntimeError("VTS authenticate failed（请确认 VTS 弹窗已 Allow，并启用插件权限）")
    return vts


def ollama_generate(user_text: str) -> str:
    """Call Ollama with fallback endpoints and return plain text response."""
    # 优先使用 /api/generate，失败后自动回退到兼容接口。
    system_hint = (
        "请在回复中用一个情绪标签标注你的语气，格式如：[emo=happy]/[emo=sad]/[emo=angry]/[emo=surprise]/[emo=neutral]。"
        "标签可放开头或结尾，正文正常回答。"
    )

    def _post(url: str, payload: dict) -> requests.Response:
        return requests.post(url, json=payload, timeout=180)


    try:
        r = _post(OLLAMA_URL, {"model": OLLAMA_MODEL, "prompt": system_hint + "\n用户：" + user_text + "\n回复：", "stream": False})
        if r.status_code == 404:
            raise requests.HTTPError("404", response=r)
        r.raise_for_status()
        return (r.json().get("response") or "").strip()
    except requests.HTTPError as e:
        if getattr(e.response, "status_code", None) not in (404,):
            raise


    try:
        url2 = OLLAMA_URL.replace("/api/generate", "/api/chat")
        r = _post(url2, {"model": OLLAMA_MODEL, "messages": [{"role":"system","content":system_hint},{"role":"user","content":user_text}], "stream": False})
        if r.status_code == 404:
            raise requests.HTTPError("404", response=r)
        r.raise_for_status()
        msg = (r.json().get("message") or {}).get("content")
        if msg:
            return str(msg).strip()
    except requests.HTTPError as e:
        if getattr(e.response, "status_code", None) not in (404,):
            raise


    url3 = OLLAMA_URL.replace("/api/generate", "/v1/chat/completions")
    r = _post(url3, {"model": OLLAMA_MODEL, "messages": [{"role":"system","content":system_hint},{"role":"user","content":user_text}], "temperature": 0.7})
    r.raise_for_status()
    return str(r.json()["choices"][0]["message"]["content"]).strip()


_EMO_ALIASES = {
    "happy": {"happy","joy","smile","开心","高兴","喜","愉快","兴奋"},
    "sad": {"sad","down","cry","难过","伤心","沮丧","失落"},
    "angry": {"angry","mad","rage","生气","愤怒","恼火"},
    "surprise": {"surprise","wow","惊讶","震惊","意外"},
    "neutral": {"neutral","calm","平静","正常","中性"},
}
_EMO_ALIASES_LOWER = {emo: {v.lower() for v in vocab} for emo, vocab in _EMO_ALIASES.items()}
_TAG_PATTERNS = [
    re.compile(r"\[(?:emo|emotion)\s*[:=]\s*([^\]\s]+)\s*\]", re.IGNORECASE),
    re.compile(r"\[([^\]\s]{2,16})\]"),
    re.compile(r"<\s*([a-zA-Z]{2,16})\s*>"),
    re.compile(r"#\s*([a-zA-Z]{2,16}|[^\s#]{1,6})\s*#"),
]

def extract_emotions_and_clean(text: str) -> Tuple[str, str, float]:
    """从 LLM 输出里提取情绪标签，并返回 (可用于TTS的文本, emotion, intensity)。
    - 如果模型没给标签，会用关键词/符号做一个保守兜底
    - intensity 过低时抬到一个可见值，避免“表情不变”
    """
    # 先采集标签，再根据别名表映射到统一情绪。
    tags: Set[str] = set()
    for pat in _TAG_PATTERNS:
        for m in pat.finditer(text):
            tags.add(m.group(1).strip())

    hit = {k: 0 for k in _EMO_ALIASES}
    for t in tags:
        tl = t.lower()
        for emo, vocab in _EMO_ALIASES.items():
            if tl in _EMO_ALIASES_LOWER[emo] or t in vocab:
                hit[emo] += 1

    emo = max(hit.items(), key=lambda kv: kv[1])[0] if any(v > 0 for v in hit.values()) else "neutral"
    intensity = min(1.0, hit.get(emo, 0) / 2.0)

    clean = text
    clean = re.sub(r"\[(?:emo|emotion)\s*[:=]\s*[^\]]+\]", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\[[^\]]{1,24}\]", "", clean)
    clean = re.sub(r"<\s*[a-zA-Z]{2,16}\s*>", "", clean)
    clean = re.sub(r"#\s*[^#]{1,24}\s*#", "", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()

    if emo == "neutral":
        c = clean
        if any(x in c for x in ("哈哈", "hh", "开心", "高兴", "耶", "太好了", "好耶", "可爱")):
            emo = "happy"
        elif any(x in c for x in ("生气", "愤怒", "气死", "可恶", "别闹", "烦", "讨厌")):
            emo = "angry"
        elif any(x in c for x in ("难过", "伤心", "呜呜", "哭", "沮丧", "失落", "委屈")):
            emo = "sad"
        elif any(x in c for x in ("!?","？！","!?", "惊", "诶", "欸", "哇", "啊？", "真的吗", "不会吧")) or ("!" in c and "？" in c):
            emo = "surprise"

    if emo != "neutral" and intensity < 0.35:
        intensity = 0.35

    return clean, emo, float(intensity)

async def set_emotion_expression(vts: VTSClient, emotion: str, intensity: float) -> None:
    if not ENABLE_EXPRESSIONS:
        return
    if emotion not in EMO_EXP_FILES:
        emotion = "neutral"


    for files in EMO_EXP_FILES.values():
        for f in files:
            await vts.set_expression(f, False, fade=0.20)

    if emotion == "neutral" or not EMO_EXP_FILES.get(emotion) or intensity < 0.02:
        return

    choices = EMO_EXP_FILES[emotion]
    pick = random.choice(choices) if (USE_RANDOM_EXPRESSION and len(choices) > 1) else choices[0]
    print(f"emotion={emotion} intensity={intensity:.2f} expr={pick}")
    await vts.set_expression(pick, True, fade=0.25)


async def tts_to_mp3(text: str, mp3_path: str) -> None:
    await edge_tts.Communicate(text, VOICE).save(mp3_path)

async def mp3_to_wav(mp3_path: str, wav_path: str) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg。请把 ffmpeg.exe 放脚本目录或加入 PATH。")
    cmd = [ffmpeg, "-y", "-i", mp3_path, "-ac", "1", "-ar", "48000", "-f", "wav", wav_path]
    p = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await p.wait()
    if p.returncode != 0:
        raise RuntimeError("ffmpeg 转 wav 失败")
    with open(wav_path, "rb") as f:
        if not f.read(12).startswith(b"RIFF"):
            raise RuntimeError("wav 不是 RIFF（异常）")

async def play_wav(wav_path: str) -> None:
    """Play synthesized WAV on Windows with fallback to default output."""
    if platform.system() != "Windows":
        return

    def _play():
        try:
            if wavfile is None:
                raise RuntimeError("未安装 scipy（pip install scipy），将降级到默认播放")
            fs, data = wavfile.read(wav_path)
            if getattr(data, "ndim", 1) == 1:
                data = data.reshape(-1, 1)

            devices = sd.query_devices()
            device_id = resolve_output_device_id(AUDIO_OUTPUT_DEVICE)

            print(f"▶️ 播放到音频设备: {devices[device_id]['name'] if device_id is not None else '系统默认'}")
            sd.play(data, samplerate=fs, device=device_id, blocking=True)
            sd.wait()
        except Exception as e:
            print(f"播放失败: {e}，降级 winsound 默认设备")
            try:
                winsound.PlaySound(wav_path, winsound.SND_FILENAME)
            except Exception:
                pass

    await asyncio.to_thread(_play)


def _autocorr_pitch_hz(frame: np.ndarray, sr: int, fmin=80.0, fmax=350.0) -> float:
    x = frame.astype(np.float32)
    x -= np.mean(x)
    if np.allclose(x, 0):
        return 0.0
    x *= np.hanning(len(x)).astype(np.float32)
    corr = np.correlate(x, x, mode="full")[len(x)-1:]
    lag_min = int(sr / fmax)
    lag_max = min(int(sr / fmin), len(corr)-1)
    if lag_max <= lag_min:
        return 0.0
    seg = corr[lag_min:lag_max]
    if len(seg) < 3:
        return 0.0
    peak = int(np.argmax(seg)) + lag_min
    return float(sr / peak) if corr[peak] > 0 else 0.0

def analyze_wav_to_controls(wav_path: str, frame_ms: int = 40) -> Tuple[List[Tuple[float, float]], float]:
    with wave.open(wav_path, "rb") as wf:
        sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        hop = max(1, int(sr * frame_ms / 1000))
        rms_list: List[float] = []
        pitch_list: List[float] = []

        while True:
            buf = wf.readframes(hop)
            if not buf:
                break
            if sw == 2:
                s = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
            elif sw == 4:
                s = np.frombuffer(buf, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                s = (np.frombuffer(buf, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            if ch > 1:
                s = s.reshape(-1, ch).mean(axis=1)

            rms = float(np.sqrt(np.mean(s*s)) + 1e-9)
            rms = max(rms, RMS_FLOOR)
            rms_list.append(rms)
            pitch_list.append(_autocorr_pitch_hz(s, sr, FREQ_MIN, FREQ_MAX) if rms > PITCH_RMS_GATE else 0.0)

        if not rms_list:
            return [(0.0, 0.0)], frame_ms / 1000

        p95 = float(np.percentile(rms_list, 95)) or 1.0
        vol_norm = [min(1.0, r / p95) for r in rms_list]

        def norm_pitch(p: float) -> float:
            if p <= 0:
                return 0.0
            return float(max(0.0, min(1.0, (p - FREQ_MIN) / (FREQ_MAX - FREQ_MIN))))

        pit_norm = [norm_pitch(p) for p in pitch_list]

        def smooth(seq: List[float], a: float) -> List[float]:
            out, prev = [], 0.0
            for v in seq:
                prev = a * prev + (1 - a) * v
                out.append(prev)
            return out

        vol_s = smooth(vol_norm, 0.55)
        pit_s = smooth(pit_norm, 0.70)

        controls = []
        for v, p in zip(vol_s, pit_s):
            mouth = max(MOUTH_MIN_OPEN, min(1.0, v * VOL_GAIN))
            smile = max(0.0, min(1.0, p * SMILE_GAIN))
            controls.append((mouth, smile))
        return controls, frame_ms / 1000


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
        try:
            self.q.put_nowait(strength)
        except Exception:
            pass

async def start_udp_blink_listener(q: "asyncio.Queue[float]"):
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(lambda: _BlinkUDPProtocol(q), local_addr=(UDP_HOST, UDP_PORT))
    return transport

async def start_blink_sender_new_console() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    sender = BLINK_SENDER_PATH if os.path.isabs(BLINK_SENDER_PATH) else os.path.join(here, BLINK_SENDER_PATH)
    if not os.path.exists(sender):
        return
    if platform.system() == "Windows":
        CREATE_NEW_CONSOLE = 0x00000010
        await asyncio.create_subprocess_exec(sys.executable, sender, cwd=here, creationflags=CREATE_NEW_CONSOLE)
    else:
        await asyncio.create_subprocess_exec(sys.executable, sender, cwd=here)


P_EYE_L_OPEN: Optional[str] = None
P_EYE_R_OPEN: Optional[str] = None
P_HEAD_X: Optional[str] = None
P_HEAD_Y: Optional[str] = None
P_HEAD_Z: Optional[str] = None
P_GAZE_X: Optional[str] = None
P_GAZE_Y: Optional[str] = None
P_HAND_LX: Optional[str] = None
P_HAND_LY: Optional[str] = None
P_HAND_RX: Optional[str] = None
P_HAND_RY: Optional[str] = None
P_HAND_WAVE: Optional[str] = None

async def init_eye_params(vts: VTSClient) -> None:
    global P_EYE_L_OPEN, P_EYE_R_OPEN
    try:
        exist = set(await vts.list_input_parameters())
    except Exception:
        exist = set()
    left = _pick_first(exist, EYE_L_CANDIDATES)
    right = _pick_first(exist, EYE_R_CANDIDATES)
    if left and right:
        P_EYE_L_OPEN, P_EYE_R_OPEN = left, right
        print(f"Eye params: L={left} R={right}")
        return

    print("未找到现成 EyeOpen，创建 AiEyeOpenL/R（需在 VTS 映射一次）")
    await vts.create_custom_parameter(CUSTOM_EYE_L, default=EYE_OPEN_MAX, minv=0.0, maxv=1.0)
    await vts.create_custom_parameter(CUSTOM_EYE_R, default=EYE_OPEN_MAX, minv=0.0, maxv=1.0)
    P_EYE_L_OPEN, P_EYE_R_OPEN = CUSTOM_EYE_L, CUSTOM_EYE_R
    print("设置(齿轮)->Model->VTS Parameter Setup：")
    print(f"  INPUT {CUSTOM_EYE_L} -> OUTPUT 左眼参数(如 ParamEyeLOpen)")
    print(f"  INPUT {CUSTOM_EYE_R} -> OUTPUT 右眼参数(如 ParamEyeROpen)")


async def init_mouth_params(vts: VTSClient) -> None:
    global P_MOUTH_OPEN, P_MOUTH_SMILE, ENABLE_MOUTH_OPEN_PARAM, ENABLE_MOUTH_SMILE_PARAM
    try:
        exist = await vts.list_input_parameters()
    except Exception:
        exist = []

    actual_name_map = {p.casefold(): p for p in exist}
    P_MOUTH_OPEN = actual_name_map.get("voicevolumeplusmouthopen", "VoiceVolumePlusMouthOpen")
    P_MOUTH_SMILE = actual_name_map.get("voicefrequencyplusmouthsmile", "VoiceFrequencyPlusMouthSmile")

    if actual_name_map:
        ENABLE_MOUTH_OPEN_PARAM = "voicevolumeplusmouthopen" in actual_name_map
        ENABLE_MOUTH_SMILE_PARAM = "voicefrequencyplusmouthsmile" in actual_name_map
    else:
        ENABLE_MOUTH_OPEN_PARAM = True
        ENABLE_MOUTH_SMILE_PARAM = True

    print(
        f"Mouth params: Open={P_MOUTH_OPEN} enabled={ENABLE_MOUTH_OPEN_PARAM} "
        f"Smile={P_MOUTH_SMILE} enabled={ENABLE_MOUTH_SMILE_PARAM}"
    )

    if not actual_name_map:
        print("警告：未获取到 VTS 输入参数列表，将按内置嘴型参数名直接注入。")
        return

    missing = []
    if not ENABLE_MOUTH_OPEN_PARAM:
        missing.append("VoiceVolumePlusMouthOpen")
    if not ENABLE_MOUTH_SMILE_PARAM:
        missing.append("VoiceFrequencyPlusMouthSmile")
    if missing:
        print("警告：未在 VTS 输入参数中找到内置嘴型参数：" + ", ".join(missing))
        print("将只注入已存在的内置参数，避免整包注入失败导致口型不动。")
        print("请确认 VTube Studio 已启用麦克风跟踪，且模型支持嘴型跟踪参数。")

async def init_motion_params(vts: VTSClient) -> None:
    global P_HEAD_X, P_HEAD_Y, P_HEAD_Z, P_GAZE_X, P_GAZE_Y
    try:
        exist = set(await vts.list_input_parameters())
    except Exception:
        exist = set()

    P_HEAD_X = _pick_first(exist, HEAD_CANDIDATES_X)
    P_HEAD_Y = _pick_first(exist, HEAD_CANDIDATES_Y)
    P_HEAD_Z = _pick_first(exist, HEAD_CANDIDATES_Z)
    P_GAZE_X = _pick_first(exist, GAZE_CANDIDATES_X)
    P_GAZE_Y = _pick_first(exist, GAZE_CANDIDATES_Y)

    if any([P_HEAD_X, P_HEAD_Y, P_HEAD_Z, P_GAZE_X, P_GAZE_Y]):
        return

    print("未找到现成头/眼输入，创建 AiHead*/AiGaze*（需在 VTS 映射一次）")
    await vts.create_custom_parameter(CUSTOM_HEAD_X, default=0.0, minv=-10.0, maxv=10.0)
    await vts.create_custom_parameter(CUSTOM_HEAD_Y, default=0.0, minv=-10.0, maxv=10.0)
    await vts.create_custom_parameter(CUSTOM_HEAD_Z, default=0.0, minv=-10.0, maxv=10.0)
    await vts.create_custom_parameter(CUSTOM_GAZE_X, default=0.0, minv=-1.0, maxv=1.0)
    await vts.create_custom_parameter(CUSTOM_GAZE_Y, default=0.0, minv=-1.0, maxv=1.0)

    P_HEAD_X, P_HEAD_Y, P_HEAD_Z = CUSTOM_HEAD_X, CUSTOM_HEAD_Y, CUSTOM_HEAD_Z
    P_GAZE_X, P_GAZE_Y = CUSTOM_GAZE_X, CUSTOM_GAZE_Y
    print("设置(齿轮)->Model->VTS Parameter Setup：")
    print(f"  {CUSTOM_HEAD_X}->{ 'ParamAngleX' }  {CUSTOM_HEAD_Y}->{ 'ParamAngleY' }  {CUSTOM_HEAD_Z}->{ 'ParamAngleZ' }")
    print(f"  {CUSTOM_GAZE_X}->{ 'ParamEyeBallX' } {CUSTOM_GAZE_Y}->{ 'ParamEyeBallY' }")

async def init_hand_params(vts: VTSClient) -> None:
    global P_HAND_LX, P_HAND_LY, P_HAND_RX, P_HAND_RY, P_HAND_WAVE
    try:
        exist = set(await vts.list_input_parameters())
    except Exception:
        exist = set()

    P_HAND_LX = _pick_first(exist, HAND_CANDIDATES_LX)
    P_HAND_LY = _pick_first(exist, HAND_CANDIDATES_LY)
    P_HAND_RX = _pick_first(exist, HAND_CANDIDATES_RX)
    P_HAND_RY = _pick_first(exist, HAND_CANDIDATES_RY)
    P_HAND_WAVE = _pick_first(exist, HAND_CANDIDATES_WAVE)

    if any([P_HAND_LX, P_HAND_LY, P_HAND_RX, P_HAND_RY, P_HAND_WAVE]):
        return

    print("未找到现成手部输入，创建 AiHand*（需在 VTS 映射一次）")
    for name, mn, mx, dv in [
        (CUSTOM_HAND_LX, -1.0, 1.0, 0.0),
        (CUSTOM_HAND_LY, -1.0, 1.0, 0.0),
        (CUSTOM_HAND_RX, -1.0, 1.0, 0.0),
        (CUSTOM_HAND_RY, -1.0, 1.0, 0.0),
        (CUSTOM_HAND_WAVE, 0.0, 1.0, 0.0),
    ]:
        await vts.create_custom_parameter(name, default=dv, minv=mn, maxv=mx)

    P_HAND_LX, P_HAND_LY, P_HAND_RX, P_HAND_RY, P_HAND_WAVE =\
        CUSTOM_HAND_LX, CUSTOM_HAND_LY, CUSTOM_HAND_RX, CUSTOM_HAND_RY, CUSTOM_HAND_WAVE


class State:
    """Thread-safe shared state for emotion and speaking status."""
    # 状态对象被多个协程共享，因此读写都需要加锁。
    def __init__(self):
        self._lock = asyncio.Lock()
        self.emo = "neutral"
        self.intensity = 0.0
        self.speaking = False

    async def set(self, emo: Optional[str] = None, intensity: Optional[float] = None, speaking: Optional[bool] = None):
        async with self._lock:
            if emo is not None:
                self.emo = emo
            if intensity is not None:
                self.intensity = float(max(0.0, min(1.0, intensity)))
            if speaking is not None:
                self.speaking = bool(speaking)

    async def get(self):
        async with self._lock:
            return self.emo, self.intensity, self.speaking

blink_lock = asyncio.Lock()
_last_blink_ts = 0.0


async def do_blink(vts: VTSClient, strength: float) -> None:
    """Perform one blink animation with cooldown control."""
    global _last_blink_ts
    if not P_EYE_L_OPEN or not P_EYE_R_OPEN:
        return
    async with blink_lock:
        now = time.perf_counter()
        if now - _last_blink_ts < BLINK_COOLDOWN_SEC:
            return
        _last_blink_ts = now

        close_norm = max(0.0, 1.0 - float(strength))
        val = _eye_map(close_norm)
        if INVERT_EYE:
            val = 1.0 - val
        try:
            await vts.inject({P_EYE_L_OPEN: val, P_EYE_R_OPEN: val}, mode="Set")
        except Exception:
            return

        await asyncio.sleep(0.05)
        for x in BLINK_OPEN_FRAMES:
            val2 = _eye_map(x)
            if INVERT_EYE:
                val2 = 1.0 - val2
            try:
                await vts.inject({P_EYE_L_OPEN: val2, P_EYE_R_OPEN: val2}, mode="Set")
            except Exception:
                break
            await asyncio.sleep(0.04)

async def blink_loop(vts: VTSClient, st: State, q: "asyncio.Queue[float]"):

    if P_EYE_L_OPEN and P_EYE_R_OPEN:
        v = _eye_map(1.0)
        if INVERT_EYE:
            v = 1.0 - v
        try:
            await vts.inject({P_EYE_L_OPEN: v, P_EYE_R_OPEN: v}, mode="Set")
        except Exception:
            pass

    while True:
        emo, inten, speaking = await st.get()
        base = random.uniform(2.5, 5.5)
        speed = (0.7 * inten if emo in ("angry", "surprise") else 0.0) + (0.35 if speaking else 0.0)
        interval = max(0.6, base * (1.0 / (1.0 + speed)))

        strength: Optional[float] = None
        if ENABLE_UDP_BLINK:
            try:
                strength = await asyncio.wait_for(q.get(), timeout=interval)
            except asyncio.TimeoutError:
                strength = None
        else:
            await asyncio.sleep(interval)

        if strength is not None:
            if BLINK_QUEUE_FLUSH:
                last = strength
                while True:
                    try:
                        last = q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                strength = last
            await do_blink(vts, strength)
            continue

        if not ENABLE_NATURAL_BLINK:
            continue

        nat = random.uniform(0.45, 0.60) + 0.10 * inten + (0.05 if speaking else 0.0)
        nat = max(0.35, min(0.95, nat))
        await do_blink(vts, nat)

async def idle_eye_keeper(vts: VTSClient):
    if not P_EYE_L_OPEN or not P_EYE_R_OPEN:
        return
    while True:
        if blink_lock.locked():
            await asyncio.sleep(0.10)
            continue
        v = _eye_map(1.0)
        if INVERT_EYE:
            v = 1.0 - v
        try:
            await vts.inject({P_EYE_L_OPEN: v, P_EYE_R_OPEN: v}, mode="Set")
        except Exception:
            pass
        await asyncio.sleep(0.35)


async def idle_motion_loop(vts: VTSClient, st: State):
    if not ENABLE_IDLE_MOTION:
        return
    if not any([P_HEAD_X, P_HEAD_Y, P_HEAD_Z, P_GAZE_X, P_GAZE_Y]):
        return

    ids = [p for p in [P_HEAD_X, P_HEAD_Y, P_HEAD_Z, P_GAZE_X, P_GAZE_Y] if p]
    cur = {k: 0.0 for k in ids}

    def absmax(rng):
        return max(abs(float(rng[0])), abs(float(rng[1])), 1e-6)

    eye_x_max, eye_y_max = absmax(EYE_X_RANGE), absmax(EYE_Y_RANGE)
    head_x_max, head_y_max = absmax(HEAD_X_RANGE), absmax(HEAD_Y_RANGE)

    while True:
        emo, inten, speaking = await st.get()

        gain = SPEAK_MOTION_GAIN if speaking else 1.3
        hz = SPEAK_MOTION_HZ if speaking else MOTION_HZ

        if blink_lock.locked():
            await asyncio.sleep(0.12)
            continue

        t_now = time.perf_counter()
        breath = float(np.sin(t_now * 2.0 * np.pi * 0.18))
        bx = breath * 0.40 * gain
        by = breath * 0.30 * gain

        gx = (random.uniform(*EYE_X_RANGE) * gain * 1.15) if P_GAZE_X else 0.0
        gy = (random.uniform(*EYE_Y_RANGE) * gain * 1.15) if P_GAZE_Y else 0.0
        follow = random.uniform(0.55, 0.85)

        tgt = {}
        if P_GAZE_X: tgt[P_GAZE_X] = gx
        if P_GAZE_Y: tgt[P_GAZE_Y] = gy
        if P_HEAD_X: tgt[P_HEAD_X] = (gx / eye_x_max) * head_x_max * follow + random.uniform(-0.35, 0.35) * gain + bx
        if P_HEAD_Y: tgt[P_HEAD_Y] = (gy / eye_y_max) * head_y_max * follow + random.uniform(-0.28, 0.28) * gain + by
        if P_HEAD_Z: tgt[P_HEAD_Z] = random.uniform(*HEAD_Z_RANGE) * 0.35 * gain

        if speaking and random.random() < SPEAK_NOD_PROB:
            tgt[P_HEAD_Y] = float(tgt.get(P_HEAD_Y, 0.0) + random.uniform(0.6, 1.2) * gain)

        ease = random.uniform(*MOTION_EASE_SEC)
        interval = random.uniform(*MOTION_INTERVAL_SEC)
        steps = max(1, int(ease * hz))
        dt = ease / steps

        start = {k: cur.get(k, 0.0) for k in ids}

        for i in range(steps):
            _, _, speaking2 = await st.get()
            if blink_lock.locked():
                break

            t01 = _smoothstep((i + 1) / steps)
            payload = {}

            for pid in ids:
                val = start[pid] + (tgt.get(pid, start[pid]) - start[pid]) * t01
                cur[pid] = val
                payload[pid] = val

            try:
                await vts.inject(payload, mode="Set")
            except:
                pass

            await asyncio.sleep(dt)

        end_t = time.perf_counter() + max(0.4, interval)

        while time.perf_counter() < end_t:
            if blink_lock.locked():
                break

            await asyncio.sleep(random.uniform(0.25, 0.55))

            if random.random() > (0.55 if speaking else 0.65):
                continue

            dx = random.uniform(-0.045, 0.045) * gain
            dy = random.uniform(-0.030, 0.030) * gain

            payload = {}

            if P_GAZE_X: payload[P_GAZE_X] = cur.get(P_GAZE_X, 0.0) + dx
            if P_GAZE_Y: payload[P_GAZE_Y] = cur.get(P_GAZE_Y, 0.0) + dy

            try:
                await vts.inject(payload, mode="Set")
            except:
                pass

            await asyncio.sleep(random.uniform(0.04, 0.08))

            payload2 = {}
            if P_GAZE_X: payload2[P_GAZE_X] = cur.get(P_GAZE_X, 0.0)
            if P_GAZE_Y: payload2[P_GAZE_Y] = cur.get(P_GAZE_Y, 0.0)

            try:
                await vts.inject(payload2, mode="Set")
            except:
                pass


async def idle_hand_loop(vts: VTSClient, st: State):
    if not ENABLE_HAND_MOTION:
        return
    if not any([P_HAND_LX, P_HAND_LY, P_HAND_RX, P_HAND_RY, P_HAND_WAVE]):
        return

    cur = {p: 0.0 for p in [P_HAND_LX, P_HAND_LY, P_HAND_RX, P_HAND_RY, P_HAND_WAVE] if p}
    next_wave_at = time.perf_counter() + random.uniform(*WAVE_COOLDOWN_SEC)

    while True:
        emo, inten, speaking = await st.get()
        if blink_lock.locked():
            await asyncio.sleep(0.10)
            continue

        now = time.perf_counter()

        if speaking and P_HAND_WAVE and now >= next_wave_at:
            dur = max(0.35, float(WAVE_DURATION_SEC))
            steps = max(1, int(dur * HAND_HZ))
            dt = dur / steps
            for i in range(steps):
                _, _, speaking2 = await st.get()
                if not speaking2:
                    break
                t = i / max(1, steps - 1)
                w = (t * 2.0) if t < 0.5 else (2.0 - t * 2.0)
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

        scale = 1.0 + (0.35 if speaking else 0.0) + 0.20 * float(inten)
        tgt = {}
        if P_HAND_LX: tgt[P_HAND_LX] = random.uniform(*HAND_X_RANGE) * scale
        if P_HAND_LY: tgt[P_HAND_LY] = random.uniform(*HAND_Y_RANGE) * scale
        if P_HAND_RX: tgt[P_HAND_RX] = random.uniform(*HAND_X_RANGE) * scale
        if P_HAND_RY: tgt[P_HAND_RY] = random.uniform(*HAND_Y_RANGE) * scale

        ease = random.uniform(*HAND_EASE_SEC)
        interval = random.uniform(*HAND_INTERVAL_SEC)
        steps = max(1, int(ease * HAND_HZ))
        dt = ease / steps

        for i in range(steps):
            t = _smoothstep((i + 1) / steps)
            payload = {}
            for pid, target in tgt.items():
                cur[pid] = float(cur[pid] + (target - cur[pid]) * t)
                payload[pid] = cur[pid]
            if payload:
                try:
                    await vts.inject(payload, mode="Set")
                except Exception:
                    pass
            await asyncio.sleep(dt)

        await asyncio.sleep(max(0.25, interval))


async def drive_voice_params(vts: VTSClient, controls: List[Tuple[float, float]], dt: float, st: State):
    global MOUTH_INJECT_PRIMARY_MODE
    await st.set(speaking=True)
    has_warned_mouth_error = False
    has_warned_mouth_exception = False

    async def inject_mouth(mouth_value: float, smile_value: float) -> None:
        nonlocal has_warned_mouth_error, has_warned_mouth_exception
        global MOUTH_INJECT_PRIMARY_MODE
        payload = {}
        if ENABLE_MOUTH_OPEN_PARAM:
            payload[P_MOUTH_OPEN] = mouth_value
        if ENABLE_MOUTH_SMILE_PARAM:
            payload[P_MOUTH_SMILE] = smile_value
        if not payload:
            return

        primary = MOUTH_INJECT_PRIMARY_MODE
        secondary = "Add" if primary == "Set" else "Set"

        try:
            resp = await vts.inject(payload, mode=primary)
        except Exception as ex:
            if not has_warned_mouth_exception:
                has_warned_mouth_exception = True
                print(f"mouth inject raised exception: {ex}")
            return

        if not is_vts_api_error(resp):
            return

        try:
            resp2 = await vts.inject(payload, mode=secondary)
        except Exception:
            return

        if not is_vts_api_error(resp2):
            MOUTH_INJECT_PRIMARY_MODE = secondary
            print(f"mouth inject mode switched to {secondary}")
            return

        if not has_warned_mouth_error:
            has_warned_mouth_error = True
            print("mouth inject failed in both Set/Add mode, check VTS parameter mapping")

    try:
        start = time.perf_counter()
        i, n = 0, len(controls)
        while i < n:
            target_t = start + i * dt
            now = time.perf_counter()
            if now > target_t + 0.5 * dt:
                i += max(1, int((now - target_t) // dt))
                if i >= n:
                    break
            mouth, smile = controls[i]
            await inject_mouth(mouth, smile)
            sleep_for = (start + (i + 1) * dt) - time.perf_counter()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            i += 1
        await inject_mouth(0.0, 0.0)
    finally:
        await st.set(speaking=False)


async def main():
    """Run interactive VTuber loop and orchestrate all background tasks."""
    # 主流程：初始化连接 -> 启动后台任务 -> 处理用户输入。
    list_audio_devices()
    print("Connecting to VTube Studio...")
    vts = await connect_vts()
    vts_motion = await connect_vts()
    print("Connected & authed.")

    await init_eye_params(vts)
    await init_mouth_params(vts)
    await init_motion_params(vts_motion)
    await init_hand_params(vts_motion)

    st = State()
    blink_q: "asyncio.Queue[float]" = asyncio.Queue(maxsize=64)

    udp_transport = None
    if ENABLE_UDP_BLINK:
        udp_transport = await start_udp_blink_listener(blink_q)
        print(f"UDP blink listener on {UDP_HOST}:{UDP_PORT} (期待 BLINK:0.xx)")
        if AUTO_START_BLINK_SENDER:
            await start_blink_sender_new_console()

    tasks = [
        asyncio.create_task(blink_loop(vts, st, blink_q)),
        asyncio.create_task(idle_eye_keeper(vts)),
        asyncio.create_task(idle_motion_loop(vts_motion, st)),
        asyncio.create_task(idle_hand_loop(vts_motion, st)),
    ]

    await asyncio.sleep(1.0)
    await do_blink(vts, 0.55)

    print("\nReady. 输入 quit 退出。\n")
    try:
        while True:
            user = (await asyncio.to_thread(input, "你：")).strip()
            if not user:
                continue

            if user.lower() in ("quit", "exit", "q", "退出"):
                print("已退出")
                break

            if user.lower().startswith("device "):
                new_dev = user[7:].strip()
                global AUDIO_OUTPUT_DEVICE
                try:
                    AUDIO_OUTPUT_DEVICE = int(new_dev)
                except ValueError:
                    AUDIO_OUTPUT_DEVICE = new_dev
                print(f"音频输出设备已切换为：{AUDIO_OUTPUT_DEVICE}")
                continue

            ai_raw = ollama_generate(user)
            print("她：", ai_raw)

            tts_text, emo, inten = extract_emotions_and_clean(ai_raw)
            await st.set(emo=emo, intensity=inten)
            await set_emotion_expression(vts, emo, inten)

            if not tts_text.strip():
                tts_text = "嗯。"

            await tts_to_mp3(tts_text, TMP_MP3)
            await mp3_to_wav(TMP_MP3, TMP_WAV)

            controls, dt = analyze_wav_to_controls(TMP_WAV, frame_ms=FRAME_MS)
            await asyncio.gather(
                play_wav(TMP_WAV),
                drive_voice_params(vts, controls, dt, st),
            )

            if AUTO_RESET_EXPRESSION_AFTER_REPLY:
                await st.set(emo="neutral", intensity=0.0)
                await set_emotion_expression(vts, "neutral", 0.0)
    finally:
        for t in tasks:
            t.cancel()
        if udp_transport is not None:
            udp_transport.close()
        try:
            await vts.ws.close()
        except Exception:
            pass
        try:
            await vts_motion.ws.close()
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())
