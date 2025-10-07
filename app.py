import logging
import os
import statistics
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
        segment_length_sec = 0.5
        segments = segment_fn(waveform, sample_rate, segment_length=segment_length_sec)
        total_duration = float(waveform.shape[-1]) / float(sample_rate) if sample_rate else 0.0

        segment_meta: List[Dict[str, Any]] = []
        energies: List[float] = []
        for idx, seg in enumerate(segments):
            if seg.shape[-1] <= 0:
                continue
            start = idx * segment_length_sec
            end = min(start + segment_length_sec, total_duration)
            # Ortalama mutlak genliği enerji temsilcisi olarak kullan.
            energy = float(seg.abs().mean().item()) if hasattr(seg, "abs") else 0.0
            segment_meta.append({
                "index": idx,
                "segment": seg,
                "start": float(start),
                "end": float(end),
                "energy": energy,
            })
            energies.append(energy)

        if not segment_meta:
            return None

        # Sessiz (çok düşük enerjili) segmentleri filtreleyerek diarizasyonun
        # gürültüye karşı daha kararlı olmasını sağla.
        positive_energies = [energy for energy in energies if energy > 0.0]
        if positive_energies:
            median_energy = statistics.median(positive_energies)
            energy_threshold = max(median_energy * 0.05, 5e-5)
        else:
            energy_threshold = 0.0

        guard_indices = set(range(min(3, len(segment_meta))))
        guard_indices.update(
            range(max(0, len(segment_meta) - 3), len(segment_meta))
        )
        significant_indices = {
            meta["index"]
            for meta in segment_meta
            if meta["energy"] >= energy_threshold
        }
        significant_indices.update(guard_indices)

        if not significant_indices:
            significant_indices = {meta["index"] for meta in segment_meta}

        significant_order = sorted(significant_indices)
        significant_segments = [segment_meta[idx] for idx in significant_order]

        global embedding_model
        if embedding_model is None:
            embedding_model = load_embed()
        if embedding_model is None:
            return None

        embeddings = extract_embed(
            embedding_model, [meta["segment"] for meta in significant_segments]
        )
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

        # Tek bir segmentlik sapmaları yumuşatarak konuşmacı geçişlerini
        # daha akıcı hale getir.
        if len(labels) >= 3:
            smoothed_labels = labels[:]
            for idx in range(1, len(labels) - 1):
                if smoothed_labels[idx - 1] == smoothed_labels[idx + 1] != smoothed_labels[idx]:
                    smoothed_labels[idx] = smoothed_labels[idx - 1]
            labels = smoothed_labels

        label_by_index: List[Optional[int]] = [None] * len(segment_meta)
        for seg_label, seg_idx in zip(labels, significant_order):
            if 0 <= seg_idx < len(label_by_index):
                label_by_index[seg_idx] = int(seg_label)

        last_seen: Optional[int] = None
        for idx in range(len(label_by_index)):
            if label_by_index[idx] is None:
                label_by_index[idx] = last_seen
            else:
                last_seen = label_by_index[idx]

        last_seen = None
        for idx in range(len(label_by_index) - 1, -1, -1):
            if label_by_index[idx] is None:
                label_by_index[idx] = last_seen
            else:
                last_seen = label_by_index[idx]

        default_label = next(
            (val for val in label_by_index if val is not None),
            0,
        )
        label_by_index = [
            (val if val is not None else default_label) for val in label_by_index
        ]

        diar_segments: List[Dict[str, Any]] = []
        for meta, raw_label in zip(segment_meta, label_by_index):
            label = int(raw_label) if raw_label is not None else 0
            diar_segments.append(
                {
                    "speaker": f"Speaker {label % n_speakers + 1}",
                    "start": meta["start"],
                    "end": meta["end"],
                }
            )

        diar_segments.sort(key=lambda item: item["start"])

        merged_segments: List[Dict[str, Any]] = []
        for seg in diar_segments:
            if not merged_segments:
                merged_segments.append(seg.copy())
                continue
            prev = merged_segments[-1]
            if (
                prev["speaker"] == seg["speaker"]
                and seg["start"] - prev["end"] <= 0.1
            ):
                prev["end"] = max(prev["end"], seg["end"])
            else:
                merged_segments.append(seg.copy())

        # Çok kısa (ör. <0.12s) segmentleri filtrele; bunlar genelde yanlış pozitiflerdir.
        stable_segments = [
            seg for seg in merged_segments if (seg["end"] - seg["start"]) >= 0.12
        ]

        return stable_segments or merged_segments
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


def assign_overlap_labels(
    whisper_segments: List[Any], diarization_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not whisper_segments:
        return []

    formatted_whisper_segments: List[Dict[str, Any]] = []
    for idx, seg in enumerate(whisper_segments):
        start = float(getattr(seg, "start", idx * 1.0))
        end = float(getattr(seg, "end", start + 1.0))
        text = str(getattr(seg, "text", "") or "")
        words_data = getattr(seg, "words", None)

        formatted_words: List[Dict[str, Any]] = []
        if isinstance(words_data, list):
            for word in words_data:
                w_start = getattr(word, "start", None) or getattr(word, "start_time", None)
                w_end = getattr(word, "end", None) or getattr(word, "end_time", None)
                w_text = str(getattr(word, "word", "") or "").strip()
                if w_start is None or w_end is None or not w_text:
                    continue
                formatted_words.append(
                    {
                        "start": float(w_start),
                        "end": float(w_end),
                        "word": w_text,
                    }
                )

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
    default_speaker = diar_segments[0]["speaker"]

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

    word_entries: List[Dict[str, Any]] = []
    for seg in formatted_whisper_segments:
        words = seg["words"]
        if words:
            for word in words:
                idx = choose_diar_index(word["start"], word["end"])
                diar = diar_segments[idx]
                word_entries.append(
                    {
                        "speaker": diar["speaker"],
                        "start": float(word["start"]),
                        "end": float(word["end"]),
                        "text": word["word"],
                    }
                )
        elif seg["text"]:
            idx = choose_diar_index(seg["start"], seg["end"])
            diar = diar_segments[idx]
            word_entries.append(
                {
                    "speaker": diar["speaker"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
            )

    if not word_entries:
        return []

    word_entries.sort(key=lambda item: (item["start"], item["end"]))

    def diar_overlap_map(segment: Dict[str, Any]) -> Dict[str, float]:
        coverage: Dict[str, float] = {}
        for diar in diar_segments:
            overlap_start = max(segment["start"], diar["start"])
            overlap_end = min(segment["end"], diar["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap <= 0.0:
                continue
            speaker = diar["speaker"]
            coverage[speaker] = coverage.get(speaker, 0.0) + overlap
        return coverage

    grouped_segments: List[Dict[str, Any]] = []
    for entry in word_entries:
        text = entry["text"].strip()
        if not text:
            continue
        if not grouped_segments:
            grouped_segments.append(entry.copy())
            grouped_segments[-1]["text"] = text
            continue

        prev = grouped_segments[-1]
        gap = entry["start"] - prev["end"]
        if entry["speaker"] == prev["speaker"] and gap <= 0.35:
            prev["end"] = max(prev["end"], entry["end"])
            prev["text"] = (prev["text"].rstrip() + " " + text.lstrip()).strip()
        else:
            grouped_segments.append({**entry, "text": text})

    if not grouped_segments:
        return []

    smoothed_segments = [seg.copy() for seg in grouped_segments]

    def segment_duration(segment: Dict[str, Any]) -> float:
        return max(0.0, float(segment["end"]) - float(segment["start"]))

    segment_stats: List[Dict[str, Any]] = []
    for seg in grouped_segments:
        coverage = diar_overlap_map(seg)
        duration = segment_duration(seg)
        if coverage and duration > 0.0:
            dominant_speaker, dominant_value = max(
                coverage.items(), key=lambda item: item[1]
            )
            dominance = dominant_value / duration if duration > 0 else 0.0
        else:
            dominant_speaker = None
            dominance = 0.0
        segment_stats.append(
            {
                "coverage": coverage,
                "dominant": dominant_speaker,
                "dominance": dominance,
                "duration": duration,
            }
        )

    for idx in range(len(smoothed_segments)):
        seg = smoothed_segments[idx]
        stats = segment_stats[idx]
        coverage = stats["coverage"]
        duration = stats["duration"]
        dominant_speaker = stats["dominant"]
        dominance = stats["dominance"]

        prev_seg = smoothed_segments[idx - 1] if idx > 0 else None
        next_seg = smoothed_segments[idx + 1] if idx + 1 < len(smoothed_segments) else None

        if duration >= 1.2:
            if (
                dominant_speaker
                and seg["speaker"] != dominant_speaker
                and dominance >= 0.55
            ):
                seg["speaker"] = dominant_speaker
            continue

        if (
            dominant_speaker
            and seg["speaker"] != dominant_speaker
            and dominance >= 0.55
        ):
            seg["speaker"] = dominant_speaker
            continue

        prev_overlap = coverage.get(prev_seg["speaker"], 0.0) if prev_seg else 0.0
        next_overlap = coverage.get(next_seg["speaker"], 0.0) if next_seg else 0.0

        if dominance < 0.45 and duration > 0.0:
            candidate_speaker: Optional[str] = None
            if (
                prev_seg
                and next_seg
                and prev_seg["speaker"] == next_seg["speaker"]
                != seg["speaker"]
                and prev_overlap + next_overlap >= duration * 0.3
            ):
                candidate_speaker = prev_seg["speaker"]
            elif prev_seg and prev_overlap >= duration * 0.25:
                candidate_speaker = prev_seg["speaker"]
            elif next_seg and next_overlap >= duration * 0.25:
                candidate_speaker = next_seg["speaker"]

            if candidate_speaker:
                seg["speaker"] = candidate_speaker
                continue

        reassigned_speaker: Optional[str] = None
        if (
            prev_seg
            and next_seg
            and prev_seg["speaker"] == next_seg["speaker"] != seg["speaker"]
            and (duration <= 0.7 or len(seg["text"].split()) <= 4)
        ):
            reassigned_speaker = prev_seg["speaker"]
        else:
            candidates: List[Dict[str, Any]] = []
            if prev_seg and prev_seg["speaker"] != seg["speaker"]:
                candidates.append(prev_seg)
            if next_seg and next_seg["speaker"] != seg["speaker"]:
                candidates.append(next_seg)
            if candidates and duration <= 0.45:
                closest = min(
                    candidates,
                    key=lambda item: abs(
                        (seg["start"] + seg["end"]) / 2.0
                        - (item["start"] + item["end"]) / 2.0
                    ),
                )
                reassigned_speaker = closest["speaker"]

        if reassigned_speaker:
            seg["speaker"] = reassigned_speaker

    merged_segments: List[Dict[str, Any]] = []
    for seg in smoothed_segments:
        if not merged_segments:
            merged_segments.append(seg.copy())
            continue

        prev = merged_segments[-1]
        gap = seg["start"] - prev["end"]
        if seg["speaker"] == prev["speaker"] and gap <= 0.5:
            prev["end"] = max(prev["end"], seg["end"])
            prev["text"] = (prev["text"].rstrip() + " " + seg["text"].lstrip()).strip()
        else:
            merged_segments.append(seg.copy())

    final_segments: List[Dict[str, Any]] = []
    for i, seg in enumerate(merged_segments):
        overlapping = any(
            i != j and is_overlap(seg, other, threshold=0.3) for j, other in enumerate(merged_segments)
        )
        if overlapping:
            seg = seg.copy()
            seg["speaker"] = f"{seg['speaker']} (OVERLAP)"
        final_segments.append(seg)

    if final_segments:
        return final_segments

    return [
        {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "speaker": default_speaker,
        }
        for seg in formatted_whisper_segments
        if seg["text"]
    ]


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