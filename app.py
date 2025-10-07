import logging
import os
import uuid
import warnings
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)
from pydub import AudioSegment

from tts_generate import generate_tts, prepare_speaker_wav

# Flask application ---------------------------------------------------------

app = Flask(__name__, template_folder='.', static_folder='.')
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB upload limit

# Logging & warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")


# Paths & directories -------------------------------------------------------

BASE_DIR = r"C:\Users\yasin\Desktop\ai"
CORPUS_DIR = os.path.join(BASE_DIR, "sps-corpus-1.0-2025-09-05-tr")
AUDIO_CORPUS_DIR = os.path.join(CORPUS_DIR, "audios")
CORPUS_FILE = os.path.join(CORPUS_DIR, "ss-corpus-tr.tsv")
REPORTED_FILE = os.path.join(CORPUS_DIR, "ss-reported-audios-tr.tsv")

CONVERTER_DIR = os.path.join(BASE_DIR, "converter")
SESCLON_DIR = os.path.join(BASE_DIR, "sesclon")
TTS_OUTPUT_DIR = os.path.join(BASE_DIR, "seslendirme")

for directory in (CONVERTER_DIR, SESCLON_DIR, TTS_OUTPUT_DIR):
    os.makedirs(directory, exist_ok=True)


# Optional dependencies -----------------------------------------------------

try:
    from faster_whisper import WhisperModel

    WHISPER_AVAILABLE = True
except Exception as exc:  # pragma: no cover - handled at runtime
    logger.warning("faster-whisper not available: %s", exc)
    WhisperModel = None  # type: ignore
    WHISPER_AVAILABLE = False


if TYPE_CHECKING:
    from faster_whisper import WhisperModel as WhisperModelType
else:
    WhisperModelType = Any

SPEAKER_DIARIZATION_AVAILABLE = False
preprocess_audio = segment_audio = extract_embeddings = load_embedding_model = None  # type: ignore
cluster_embeddings = None  # type: ignore

try:
    import torchaudio

    from clustering.speaker_clustering import cluster_embeddings
    from embedding.speaker_embedding import load_embedding_model, extract_embeddings
    from preprocessing.audio_preprocessing import preprocess_audio, segment_audio

    SPEAKER_DIARIZATION_AVAILABLE = True
    logger.info("Speaker diarization modules loaded.")
except Exception as exc:  # pragma: no cover - optional dependency
    logger.warning("Speaker diarization unavailable: %s", exc)
    torchaudio = None  # type: ignore


# Global state --------------------------------------------------------------

whisper_model: Optional[WhisperModelType] = None
embedding_model = None
selected_wav: Optional[str] = None


# Utility helpers -----------------------------------------------------------

def ensure_initialized() -> None:
    """Lazy initialisation for Whisper / TTS speaker embedding model."""
    global whisper_model, embedding_model, selected_wav

    if selected_wav is None:
        try:
            selected_wav = prepare_speaker_wav()
            if selected_wav:
                logger.info("Speaker WAV prepared at %s", selected_wav)
        except Exception as exc:
            logger.warning("Failed to prepare speaker WAV: %s", exc)
            selected_wav = None

    if WHISPER_AVAILABLE and whisper_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        logger.info("Loading Whisper model (large-v3) on %s (%s)", device, compute_type)
        if WhisperModel is not None:
            whisper_model = WhisperModel("large-v3", device=device, compute_type=compute_type)
        else:
            logger.error("WhisperModel is not available")
            whisper_model = None

    if SPEAKER_DIARIZATION_AVAILABLE and embedding_model is None and callable(load_embedding_model):
        try:
            embedding_model = load_embedding_model()
            logger.info("Speaker embedding model loaded.")
        except Exception as exc:
            logger.warning("Failed to load speaker embedding model: %s", exc)
            embedding_model = None


def cleanup_files(file_paths: List[Optional[str]]) -> None:
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError as exc:
                logger.debug("Failed to remove %s: %s", path, exc)


def seconds_to_timestamp(value: float) -> str:
    minutes = int(value // 60)
    seconds = int(value % 60)
    return f"{minutes:02d}:{seconds:02d}"


# Diarization utilities -----------------------------------------------------

def perform_speaker_diarization(audio_path: str, n_speakers: int = 2) -> Optional[List[Dict[str, Any]]]:
    if not SPEAKER_DIARIZATION_AVAILABLE:
        return None

    try:
        if (
            preprocess_audio is None
            or segment_audio is None
            or load_embedding_model is None
            or extract_embeddings is None
        ):
            logger.warning("Speaker diarization helpers unavailable.")
            return None

        preproc = preprocess_audio
        segment_fn = segment_audio
        load_embed = load_embedding_model
        extract_embed = extract_embeddings

        waveform, sample_rate = preproc(audio_path)
        segment_length_sec = 0.25
        segments = segment_fn(waveform, sample_rate, segment_length=segment_length_sec)
        filtered_segments = [seg for seg in segments if seg.shape[-1] > 0]

        if not filtered_segments:
            return None

        global embedding_model
        if embedding_model is None:
            embedding_model = load_embed()
        if embedding_model is None:
            return None

        embeddings = extract_embed(embedding_model, filtered_segments)
        if not embeddings:
            return None

        labels = None
        if cluster_embeddings is not None:
            try:
                labels = cluster_embeddings(embeddings, method="kmeans", n_clusters=n_speakers)
            except Exception as exc:
                logger.warning("Clustering failed: %s", exc)

        if not labels:
            labels = [idx % n_speakers for idx in range(len(embeddings))]

        diar_segments = []
        for idx, label in enumerate(labels):
            start = idx * segment_length_sec
            end = (idx + 1) * segment_length_sec
            diar_segments.append(
                {
                    "speaker": f"Speaker {int(label) % n_speakers + 1}",
                    "start": float(start),
                    "end": float(end),
                }
            )

        return diar_segments
    except Exception as exc:
        logger.warning("Speaker diarization failed: %s", exc)
        return None


def is_overlap(seg1: Dict[str, Any], seg2: Dict[str, Any], threshold: float = 0.0) -> bool:
    overlap_start = max(seg1["start"], seg2["start"])
    overlap_end = min(seg1["end"], seg2["end"])
    overlap_duration = max(0.0, overlap_end - overlap_start)
    min_duration = min(seg1["end"] - seg1["start"], seg2["end"] - seg2["start"])
    if min_duration <= 0:
        return False
    return overlap_duration / min_duration > threshold


def assign_overlap_labels(whisper_segments: List[Any], diarization_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not whisper_segments:
        return []

    formatted_whisper_segments: List[Dict[str, Any]] = []
    for idx, seg in enumerate(whisper_segments):
        start = float(getattr(seg, "start", idx * 1.0))
        end = float(getattr(seg, "end", start + 1.0))
        text = str(getattr(seg, "text", "") or "")
        words_data = getattr(seg, "words", None)

        formatted_words = []
        if isinstance(words_data, list):
            for word in words_data:
                w_start = getattr(word, "start", None) or getattr(word, "start_time", None)
                w_end = getattr(word, "end", None) or getattr(word, "end_time", None)
                w_text = str(getattr(word, "word", "") or "").strip()
                if w_start is None or w_end is None or not w_text:
                    continue
                formatted_words.append({"start": float(w_start), "end": float(w_end), "word": w_text})

        formatted_whisper_segments.append(
            {
                "start": start,
                "end": end,
                "text": text.strip(),
                "words": formatted_words,
            }
        )

    diar_segments = [
        {
            "speaker": str(seg.get("speaker", "Speaker 1")),
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
        }
        for seg in diarization_segments
        if float(seg.get("end", 0.0)) > float(seg.get("start", 0.0))
    ]

    if not diar_segments:
        return [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker": "Speaker 1",
            }
            for seg in formatted_whisper_segments
            if seg["text"]
        ]

    diar_segments.sort(key=lambda d: d["start"])
    speaker_fallback = diar_segments[0]["speaker"]

    def choose_diar_index(start: float, end: float) -> int:
        best_idx = 0
        best_overlap = 0.0
        for idx, diar in enumerate(diar_segments):
            overlap_start = max(start, diar["start"])
            overlap_end = min(end, diar["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_idx = idx
                best_overlap = overlap
        if best_overlap == 0.0:
            mid = (start + end) / 2.0
            best_idx = min(
                range(len(diar_segments)),
                key=lambda i: abs(mid - ((diar_segments[i]["start"] + diar_segments[i]["end"]) / 2.0)),
            )
        return best_idx

    diar_text: Dict[int, Dict[str, Any]] = {}

    for seg in formatted_whisper_segments:
        words = seg["words"]
        if words:
            for word in words:
                idx = choose_diar_index(word["start"], word["end"])
                entry = diar_text.setdefault(
                    idx,
                    {
                        "speaker": diar_segments[idx]["speaker"],
                        "parts": [],
                        "start": word["start"],
                        "end": word["end"],
                    },
                )
                entry["parts"].append(word["word"])
                entry["start"] = min(entry["start"], word["start"])
                entry["end"] = max(entry["end"], word["end"])
        elif seg["text"]:
            idx = choose_diar_index(seg["start"], seg["end"])
            entry = diar_text.setdefault(
                idx,
                {
                    "speaker": diar_segments[idx]["speaker"],
                    "parts": [],
                    "start": seg["start"],
                    "end": seg["end"],
                },
            )
            entry["parts"].append(seg["text"])
            entry["start"] = min(entry["start"], seg["start"])
            entry["end"] = max(entry["end"], seg["end"])

    labeled_segments: List[Dict[str, Any]] = []
    for idx, diar in enumerate(diar_segments):
        entry = diar_text.get(
            idx,
            {
                "speaker": diar["speaker"],
                "parts": [],
                "start": diar["start"],
                "end": diar["end"],
            },
        )
        text = " ".join(part for part in entry["parts"] if part).strip()
        if not text:
            continue
        start = max(entry["start"], diar["start"])
        end = min(entry["end"], diar["end"])
        if end <= start:
            start, end = diar["start"], diar["end"]
        labeled_segments.append(
            {
                "start": float(start),
                "end": float(end),
                "text": text,
                "speaker": entry.get("speaker", speaker_fallback),
            }
        )

    labeled_segments.sort(key=lambda s: s["start"])

    merged_segments: List[Dict[str, Any]] = []
    if labeled_segments:
        current = labeled_segments[0].copy()
        for seg in labeled_segments[1:]:
            if current["speaker"] == seg["speaker"] and seg["start"] - current["end"] <= 0.2:
                current["end"] = max(current["end"], seg["end"])
                current["text"] = (current["text"].rstrip() + " " + seg["text"].lstrip()).strip()
            else:
                merged_segments.append(current)
                current = seg.copy()
        merged_segments.append(current)

    final_segments: List[Dict[str, Any]] = []
    for i, seg in enumerate(merged_segments):
        overlapping = any(
            i != j and is_overlap(seg, other, threshold=0.3) for j, other in enumerate(merged_segments)
        )
        if overlapping:
            seg = seg.copy()
            seg["speaker"] = f"{seg['speaker']} (OVERLAP)"
        final_segments.append(seg)

    return final_segments


def match_speaker_segments(whisper_segments: List[Any], diarization_segments: List[Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    formatted_whisper_segments: List[Dict[str, Any]] = []
    for i, seg in enumerate(whisper_segments):
        start = float(getattr(seg, "start", i * 1.0))
        end = float(getattr(seg, "end", start + 1.0))
        text = str(getattr(seg, "text", "") or "")
        formatted_whisper_segments.append({"start": start, "end": end, "text": text})

    formatted_diarization_labels: List[str] = []
    if isinstance(diarization_segments, list):
        for item in diarization_segments:
            if isinstance(item, tuple) and len(item) == 2:
                _, label = item
                try:
                    speaker_label = f"Speaker {int(label) + 1}"
                except (ValueError, TypeError):
                    speaker_label = "Speaker 1"
                formatted_diarization_labels.append(speaker_label)
            elif isinstance(item, (int, float)):
                formatted_diarization_labels.append(f"Speaker {int(item) + 1}")
            elif isinstance(item, dict) and "speaker" in item:
                formatted_diarization_labels.append(str(item["speaker"]))
            else:
                formatted_diarization_labels.append("Speaker 1")

    for i, seg in enumerate(formatted_whisper_segments):
        speaker_label = (
            formatted_diarization_labels[i % len(formatted_diarization_labels)]
            if formatted_diarization_labels
            else "Speaker 1"
        )
        results.append(
            {
                "speaker": speaker_label,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
        )

    merged_results: List[Dict[str, Any]] = []
    if results:
        current = results[0].copy()
        for seg in results[1:]:
            if current["speaker"] == seg["speaker"] and seg["start"] - current["end"] <= 1.0:
                current["end"] = seg["end"]
                current["text"] = (current["text"].rstrip() + " " + seg["text"].lstrip()).strip()
            else:
                merged_results.append(current)
                current = seg.copy()
        merged_results.append(current)
        results = merged_results

    return results


def summarize_speakers(labeled_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not labeled_segments:
        return []

    summary: Dict[str, Dict[str, Any]] = {}
    for seg in sorted(labeled_segments, key=lambda s: s["start"]):
        speaker = str(seg["speaker"]).replace(" (OVERLAP)", "")
        entry = summary.setdefault(speaker, {"total_speech_time": 0.0, "segments": []})
        duration = max(0.0, float(seg["end"]) - float(seg["start"]))
        entry["total_speech_time"] += duration
        entry["segments"].append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": str(seg["text"]),
            }
        )

    ordered = sorted(summary.items(), key=lambda item: item[1]["total_speech_time"], reverse=True)
    results: List[Dict[str, Any]] = []
    for speaker, data in ordered:
        segments_sorted = sorted(data["segments"], key=lambda x: x["start"])
        results.append(
            {
                "speaker": speaker,
                "total_speech_time": round(data["total_speech_time"], 2),
                "segments": segments_sorted,
            }
        )
    if len(results) > 2:
        results = results[:2]
    return results


# Transcription core --------------------------------------------------------

def build_transcription_text(labeled_segments: List[Dict[str, Any]]) -> str:
    if not labeled_segments:
        return ""
    lines: List[str] = []
    current_speaker: Optional[str] = None
    for seg in sorted(labeled_segments, key=lambda s: s["start"]):
        speaker = str(seg["speaker"])
        if speaker != current_speaker:
            lines.append(f"\n{speaker}:")
            current_speaker = speaker
        start_ts = seconds_to_timestamp(seg["start"])
        end_ts = seconds_to_timestamp(seg["end"])
        lines.append(f"  [{start_ts}-{end_ts}] {seg['text']}")
    return "\n".join(lines).strip()


def detailed_segments(labeled_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker": str(seg["speaker"]),
            "text": str(seg["text"]),
        }
        for seg in sorted(labeled_segments, key=lambda s: s["start"])
    ]


def transcribe_file(audio_path: str) -> Dict[str, Any]:
    ensure_initialized()
    if not WHISPER_AVAILABLE or whisper_model is None:
        raise RuntimeError("Whisper model is not available.")

    try:
        segments, info = whisper_model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=True,
        )
        segments_list = list(segments)

        diarization = perform_speaker_diarization(audio_path, n_speakers=2)
        if diarization:
            labeled_segments = assign_overlap_labels(segments_list, diarization)
        else:
            labeled_segments = match_speaker_segments(segments_list, diarization or [])

        text = build_transcription_text(labeled_segments)
        speakers = summarize_speakers(labeled_segments)
        details = detailed_segments(labeled_segments)

        return {
            "text": text,
            "speakers": speakers,
            "detailed_transcription": details,
            "language": getattr(info, "language", "unknown"),
        }
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        raise


# Routes --------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/convert", methods=["POST"])
def convert_text_to_speech():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"success": False, "error": "Metin gerekli."}), 400

    ensure_initialized()
    output_filename = f"tts_{uuid.uuid4().hex}.wav"
    output_path = os.path.join(TTS_OUTPUT_DIR, output_filename)

    success = generate_tts(text, output_path)
    if not success or not os.path.exists(output_path):
        cleanup_files([output_path])
        return jsonify({"success": False, "error": "Ses oluşturulamadı."}), 500

    return jsonify({"success": True, "filename": output_filename})


@app.route("/audio/<path:filename>")
def serve_audio(filename: str):
    file_path = os.path.join(TTS_OUTPUT_DIR, filename)
    logger.info(f"Attempting to serve audio file: {file_path}")
    logger.info(f"TTS_OUTPUT_DIR: {TTS_OUTPUT_DIR}")
    logger.info(f"Filename: {filename}")
    logger.info(f"File exists: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        logger.error(f"Audio file not found: {file_path}")
        return jsonify({"success": False, "error": "Dosya bulunamadı."}), 404
    
    logger.info(f"Serving audio file: {file_path}")
    return send_file(file_path, as_attachment=False, mimetype='audio/wav')


# Statik dosyalar için route
@app.route('/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory('.', filename)
    except FileNotFoundError:
        return jsonify({"success": False, "error": "Dosya bulunamadı."}), 404


def save_uploaded_file(field_name: str) -> Optional[str]:
    file = request.files.get(field_name)
    if file is None:
        return None
    filename = file.filename or ""
    if filename == "":
        return None
    _, ext = os.path.splitext(filename)
    temp_path = os.path.join(CONVERTER_DIR, f"upload_{uuid.uuid4().hex}{ext.lower()}")
    file.save(temp_path)
    return temp_path


def convert_to_wav(input_path: str) -> str:
    ext = os.path.splitext(input_path)[1].lower()
    output_path = os.path.join(CONVERTER_DIR, f"converted_{uuid.uuid4().hex}.wav")
    if ext == ".wav":
        os.replace(input_path, output_path)
        return output_path
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")
    os.remove(input_path)
    return output_path


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    temp_path = save_uploaded_file("audio")
    if temp_path is None:
        return jsonify({"success": False, "error": "Ses dosyası gerekli."}), 400

    try:
        wav_path = convert_to_wav(temp_path)
        result = transcribe_file(wav_path)
        return jsonify({"success": True, **result})
    except Exception as exc:
        logger.exception("Transcription failed: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        wav_path_local = locals().get("wav_path")
        cleanup_files(
            [
                temp_path,
                wav_path_local if isinstance(wav_path_local, str) else None,
            ]
        )


@app.route("/transcribe_video", methods=["POST"])
def transcribe_video():
    temp_path = save_uploaded_file("audio")
    if temp_path is None:
        return jsonify({"success": False, "error": "Video dosyası gerekli."}), 400

    wav_path = None
    try:
        wav_path = os.path.join(CONVERTER_DIR, f"video_{uuid.uuid4().hex}.wav")
        audio = AudioSegment.from_file(temp_path)
        audio.export(wav_path, format="wav")
        result = transcribe_file(wav_path)
        return jsonify({"success": True, **result})
    except Exception as exc:
        logger.exception("Video transcription failed: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        cleanup_files([temp_path, wav_path])


# Main ----------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)