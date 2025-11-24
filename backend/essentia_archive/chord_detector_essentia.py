import math
import logging
import os
import tempfile
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
import numpy as np
import librosa
import scipy.ndimage

from domain.models import DetectedChord
from domain.music_theory import NOTES, get_chord_notes, frequency_to_note, base

# --- Core Functions ---

def extract_notes_per_second(file: str, sr: int = 22050) -> Tuple[Dict[int, Dict[str, Any]], float]:
    y, sr_used = librosa.load(file, sr=sr)
    dur = len(y)/sr_used
    total = int(math.ceil(dur))
    res: Dict[int, Dict[str, Any]] = {}
    for s in range(total):
        ini, fin = s*sr_used, min((s+1)*sr_used, len(y))
        seg = y[ini:fin]
        if seg.size == 0: continue
        pitches, mags = librosa.piptrack(y=seg, sr=sr_used, threshold=0.05)
        freqs: List[float] = []
        for t in range(pitches.shape[1]):
            idx = np.argsort(mags[:, t])[-5:]
            for i in idx:
                f = pitches[i, t]
                if f > 0 and mags[i, t] > 0.1:
                    freqs.append(float(f))
        if freqs:
            fm = float(np.median(freqs))
            res[s] = {'note': frequency_to_note(fm), 'frequency': fm, 'confidence': len(freqs)}
        else:
            res[s] = {'note': 'Silence', 'frequency': 0.0, 'confidence': 0}
    return res, dur

def compact_changes(notes_sec: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    prev: Optional[str] = None
    for s in sorted(notes_sec.keys()):
        n = notes_sec[s]['note']
        if n != prev:
            out.append({'second': s, **notes_sec[s]})
            prev = n
    return out

def detect_chords(changes: List[Dict[str, Any]], window: int = 8) -> List[DetectedChord]:
    res: List[DetectedChord] = []
    i = 0
    while i < len(changes):
        t0 = changes[i]['second']
        coll: List[str] = []
        j = i
        while j < len(changes) and changes[j]['second'] <= t0 + window:
            b = base(changes[j]['note'])
            if b != 'Silence' and b not in coll:
                coll.append(b)
            j += 1
        best: Optional[Tuple[str, float]] = None
        # Try common chord types with PyChord
        common_chords = ['C', 'Cm', 'D', 'Dm', 'E', 'Em', 'F', 'Fm', 'G', 'Gm', 'A', 'Am', 'B', 'Bm']
        for symb in common_chords:
            notes_ac = get_chord_notes(symb)
            if not notes_ac: continue
            inter = set(notes_ac).intersection(coll)
            if not inter: continue
            score = len(inter)/len(notes_ac)
            if best is None or score > best[1]:
                best = (symb, score)
        t1 = changes[j-1]['second'] if j-1 < len(changes) else t0 + window
        if best and (not res or res[-1].symbol != best[0]):
            res.append(DetectedChord(best[0], float(t0), float(t1), coll.copy(), best[1]*100.0))
        i = j
    return res

# --- Alternative detection based on chromas (major/minor) ---
def _triad_templates() -> Dict[str, np.ndarray]:
    # Vector of 12 pitch classes C,C#,D,...,B
    templates = {}
    pcs = NOTES  # order already defined
    for i, root in enumerate(pcs):
        # Major: R, M3 (+4), P5 (+7)
        maj = np.zeros(12); maj[i]=1; maj[(i+4)%12]=0.9; maj[(i+7)%12]=0.85
        minv = np.zeros(12); minv[i]=1; minv[(i+3)%12]=0.9; minv[(i+7)%12]=0.85
        templates[root] = maj
        templates[root+'m'] = minv
    # Normalize
    for k,v in templates.items():
        s = v.sum()
        if s>0: templates[k] = v/s
    return templates

def detect_chords_chroma(audio_file: str,
                            sr: int = 22050,
                            hop_length: int = 4096,
                            min_conf: float = 0.25,
                            smooth: int = 3) -> List[DetectedChord]:
    """Detects chords using CQT chromas + simple triad template.
    Returns list of DetectedChord with continuous times.
    """
    try:
        y, sr_used = librosa.load(audio_file, sr=sr)
    except Exception as e:
        print('❌ Error loading audio for chromas:', e)
        return []
    # Separate harmonic component for stability
    try:
        y_harm, _ = librosa.effects.hpss(y)
    except Exception:
        y_harm = y
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr_used, hop_length=hop_length)
    # Normalize columns
    col_sums = chroma.sum(axis=0)
    col_sums[col_sums==0] = 1
    chroma_norm = chroma / col_sums
    templates = _triad_templates()
    symbols = list(templates.keys())
    mat_templates = np.stack([templates[s] for s in symbols], axis=0)  # (Nchords,12)
    # Similarity (dot product) -> since both normalized
    sims = mat_templates @ chroma_norm  # (Nchords, frames)
    best_idx = np.argmax(sims, axis=0)
    best_score = sims[best_idx, np.arange(sims.shape[1])]
    best_symbol = [symbols[i] for i in best_idx]
    # Nullify below threshold
    for k,sc in enumerate(best_score):
        if sc < min_conf:
            best_symbol[k] = 'N'
    # Simple smoothing (sliding mode of size 'smooth')
    if smooth > 1:
        half = smooth//2
        for i in range(len(best_symbol)):
            win = best_symbol[max(0,i-half):min(len(best_symbol), i+half+1)]
            if win:
                # mode
                cnt = Counter([w for w in win if w!='N'])
                if cnt:
                    best_symbol[i] = cnt.most_common(1)[0][0]
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr_used, hop_length=hop_length)
    res: List[DetectedChord] = []
    cur_sym = None
    cur_start = None
    cur_score_acc = []
    for t,sym,sc in zip(times, best_symbol, best_score):
        if sym == 'N':
            if cur_sym is not None and cur_start is not None:
                fin = t
                avg_score = float(np.mean(cur_score_acc))*100.0 if cur_score_acc else 0.0
                notes = [cur_sym[0]] if cur_sym else []
                res.append(DetectedChord(cur_sym, float(cur_start), float(fin), notes, avg_score))
                cur_sym = None
                cur_start = None
                cur_score_acc = []
            continue
        if sym != cur_sym:
            # close previous
            if cur_sym is not None and cur_start is not None:
                fin = t
                avg_score = float(np.mean(cur_score_acc))*100.0 if cur_score_acc else 0.0
                notes = [cur_sym[0]] if cur_sym else []
                res.append(DetectedChord(cur_sym, float(cur_start), float(fin), notes, avg_score))
            cur_sym = sym
            cur_start = t
            cur_score_acc = [sc]
        else:
            cur_score_acc.append(sc)
    # close final
    if cur_sym is not None and cur_start is not None:
        fin = float(times[-1])
        avg_score = float(np.mean(cur_score_acc))*100.0 if cur_score_acc else 0.0
        notes = [cur_sym[0]] if cur_sym else []
        res.append(DetectedChord(cur_sym, float(cur_start), float(fin), notes, avg_score))
    return res

# --- Advanced method: chromas + Viterbi (simplified HPCP) ---
def _create_extended_templates() -> Tuple[List[str], np.ndarray]:
    # Templates for: major, minor, dom7, maj7, m7, sus4 (optional)
    # 12-D representation. Weights: root=1, third=0.95, fifth=0.9, seventh=0.85, fourth(sus)=0.9
    types = {
        '': [0,4,7],          # major
        'm': [0,3,7],         # minor
        '7': [0,4,7,10],      # dom 7
        'maj7': [0,4,7,11],   # maj 7
        'm7': [0,3,7,10],     # min 7
        'sus4': [0,5,7],      # sus4
    }
    names = []
    mats = []
    for r in range(12):
        for suf, intervals in types.items():
            v = np.zeros(12, dtype=float)
            for k, iv in enumerate(intervals):
                semitone = (r + iv) % 12
                # weight by role
                if k == 0:
                    w = 1.0
                elif iv in (3,4): # third
                    w = 0.95
                elif iv in (5,): # fourth (sus)
                    w = 0.9
                elif iv in (7,): # fifth
                    w = 0.9
                else: # seventh
                    w = 0.85
                v[semitone] = w
            # small floor for notes outside chord
            v[v == 0] = 0.08
            v /= np.linalg.norm(v) + 1e-9
            name = NOTES[r] + suf
            names.append(name)
            mats.append(v)
    return names, np.vstack(mats)  # (N,12)

def detect_chords_hpcp_viterbi(audio_file: str,
                                  sr: int = 22050,
                                  hop_length: int = 2048,
                                  gamma: float = 12.0,
                                  stay_cost: float = 0.1,
                                  neighbor_cost: float = 0.3,
                                  change_cost: float = 1.0,
                                  min_dur: float = 0.4) -> List[DetectedChord]:
    """Chord detection using chromas + Viterbi.
    gamma: softmax sharpness factor for emissions
    Costs (stay/neighbor/change) control transition smoothness.
    """
    try:
        y, sr_used = librosa.load(audio_file, sr=sr)
    except Exception as e:
        print('❌ Error loading audio (hpcp):', e)
        return []
    try:
        y_h, _ = librosa.effects.hpss(y)
    except Exception:
        y_h = y
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr_used, hop_length=hop_length)
    # Normalize columns L2
    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True)+1e-9)
    names, templates = _create_extended_templates()  # (M,12)
    # Emissions: cosine similarity -> softmax row-wise
    sims = templates @ chroma_norm  # (M,F)
    # Scale and softmax column by column
    emis = np.exp(gamma * (sims - np.max(sims, axis=0, keepdims=True)))
    emis /= emis.sum(axis=0, keepdims=True)
    F = emis.shape[1]
    M = emis.shape[0]
    # Transition cost matrix (lower = better). Convert to negative log-prob.
    trans_cost = np.full((M,M), change_cost, dtype=float)
    # Same root but different suffix: medium penalty
    for i, ni in enumerate(names):
        root_i = ni.rstrip('m7ajsuor')  # simplistic; extract first note (# included)
        if len(root_i) >=2 and root_i[1] == '#':
            root_i = root_i[:2]
        else:
            root_i = root_i[0]
        for j, nj in enumerate(names):
            root_j = nj.rstrip('m7ajsuor')
            if len(root_j) >=2 and root_j[1] == '#':
                root_j = root_j[:2]
            else:
                root_j = root_j[0]
            if i == j:
                trans_cost[i,j] = stay_cost
            elif root_i == root_j:
                trans_cost[i,j] = neighbor_cost
    # Convert to pseudo log-prob: p ~ exp(-cost)
    trans_log = -trans_cost
    # Viterbi
    log_emis = np.log(emis + 1e-12)
    dp = np.zeros((M,F)) - 1e9
    back = np.zeros((M,F), dtype=np.int32)
    dp[:,0] = log_emis[:,0]
    for f in range(1,F):
        prev = dp[:,f-1][:,None] + trans_log  # (M,M)
        best_prev = np.argmax(prev, axis=0)
        dp[:,f] = prev[best_prev, range(M)] + log_emis[:,f]
        back[:,f] = best_prev
    path = np.zeros(F, dtype=np.int32)
    path[-1] = np.argmax(dp[:, -1])
    for f in range(F-2, -1, -1):
        path[f] = back[path[f+1], f+1]
    times = librosa.frames_to_time(np.arange(F), sr=sr_used, hop_length=hop_length)
    # Group segments
    res: List[DetectedChord] = []
    cur = path[0]
    start_t = times[0]
    scores_seg = [float(np.exp(dp[cur,0]))]
    for idx in range(1,F):
        p = path[idx]
        if p != cur:
            end_t = times[idx]
            dur = end_t - start_t
            if dur >= min_dur:
                name = names[cur]
                avg_score = np.mean(scores_seg)
                notes = [name]  # simplified
                res.append(DetectedChord(name, float(start_t), float(end_t), notes, float(avg_score*100)))
            cur = p
            start_t = times[idx]
            scores_seg = [float(np.exp(dp[cur,idx]))]
        else:
            scores_seg.append(float(np.exp(dp[cur,idx])))
    # last
    end_t = times[-1]
    if end_t - start_t >= min_dur:
        name = names[cur]
        avg_score = np.mean(scores_seg)
        notes = [name]
        res.append(DetectedChord(name, float(start_t), float(end_t), notes, float(avg_score*100)))
    return res

# --- Method focused on bass root ---
def _extract_root(name: str) -> str:
    # Root = first note (possible #)
    if len(name) >=2 and name[1] == '#':
        return name[:2]
    return name[0]

def detect_chords_bass(audio_file: str,
                          sr: int = 22050,
                          hop_length: int = 2048,
                          min_seg: float = 0.35) -> List[DetectedChord]:
    """Detects chords emphasizing the bass root:
    1. pyin for low f0
    2. Group consecutive roots.
    3. For each segment calculate mean chromas and select quality (major, minor, 7, maj7, m7, sus4) based on similarity to templates with that root.
    """
    try:
        y, sr_used = librosa.load(audio_file, sr=sr)
    except Exception as e:
        print('❌ Error loading audio (bass):', e)
        return []
    try:
        y_h, _ = librosa.effects.hpss(y)
    except Exception:
        y_h = y
    # Try pyin with smaller hop for higher resolution and avoid transition error
    pyin_hop = max(256, hop_length//4)
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y_h,
            fmin=float(librosa.note_to_hz('E2')),
            fmax=float(librosa.note_to_hz('E5')),
            sr=sr_used,
            hop_length=pyin_hop
        )
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr_used, hop_length=pyin_hop)
    except Exception:
        # Fallback: use STFT to estimate approximate fundamental
        st_hop = 512
        S = librosa.stft(y_h, n_fft=2048, hop_length=st_hop)
        mag = np.abs(S)
        freqs_st = librosa.fft_frequencies(sr=sr_used, n_fft=2048)
        low_mask = freqs_st <= 600
        mag_low = mag[low_mask]
        freqs_low = freqs_st[low_mask]
        peak_idx = np.argmax(mag_low, axis=0)
        f0 = freqs_low[peak_idx]
        voiced_flag = np.ones_like(f0, dtype=bool)
        voiced_prob = np.ones_like(f0, dtype=float)
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr_used, hop_length=st_hop)
    # Map to pitch class root
    roots: List[Optional[str]] = []
    for val, vf in zip(f0, voiced_flag):
        if vf and val is not None and val > 0:
            try:
                note = librosa.hz_to_note(val, octave=False)
                # note e.g. 'C#'
                root = note if note in NOTES else note[0]
                roots.append(root)
            except Exception:
                roots.append(None)
        else:
            roots.append(None)
    # If pyin failed largely, fallback to low CQT energies
    if sum(r is not None for r in roots) < len(roots)*0.1:
        fmin = librosa.note_to_hz('E1')
        n_bins = 36
        cqt = librosa.cqt(y_h, sr=sr_used, fmin=fmin, n_bins=n_bins, bins_per_octave=12, hop_length=hop_length)
        freqs = librosa.cqt_frequencies(n_bins, fmin=float(fmin), bins_per_octave=12)
        midi = librosa.hz_to_midi(freqs)
        pcs = (midi.astype(int)) % 12
        mag = np.abs(cqt)
        # Energies per pitch class per frame
        pc_energy = np.zeros((12, mag.shape[1]))
        for i_pc in range(12):
            pc_energy[i_pc] = mag[pcs == i_pc].sum(axis=0) if np.any(pcs==i_pc) else 0
        dominant = np.argmax(pc_energy, axis=0)
        roots = [NOTES[d] for d in dominant]
    # Group segments
    segments = []  # (root, start_idx, end_idx)
    cur_root = None
    start_idx = 0
    for idx, r in enumerate(roots):
        if r != cur_root:
            if cur_root is not None:
                segments.append((cur_root, start_idx, idx))
            cur_root = r
            start_idx = idx
    if cur_root is not None:
        segments.append((cur_root, start_idx, len(roots)))
    # Global Chroma
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr_used, hop_length=hop_length)
    names_ext, templates_ext = _create_extended_templates()
    # Pre-filter templates by root
    by_root: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for name, vec in zip(names_ext, templates_ext):
        rt = _extract_root(name)
        by_root.setdefault(rt, []).append((name, vec))
    results: List[DetectedChord] = []
    for root, a, b in segments:
        if root is None:
            continue
        t0 = times[a]
        t1 = times[min(b, len(times)-1)] if b < len(times) else times[-1]
        if t1 - t0 < min_seg:
            continue
        chroma_seg = chroma[:, a:b]
        if chroma_seg.size == 0:
            continue
        avg = chroma_seg.mean(axis=1)
        norm = avg / (np.linalg.norm(avg)+1e-9)
        cand_list = by_root.get(root, [])
        if not cand_list:
            continue
        best_name = None
        best_score = -1.0
        for nm, vec in cand_list:
            sc = float(np.dot(vec, norm))
            if sc > best_score:
                best_score = sc
                best_name = nm
        if best_name is None:
            continue
        results.append(DetectedChord(best_name, float(t0), float(t1), [root], best_score*100.0))
    # Merge adjacent equal segments
    fusion: List[DetectedChord] = []
    for ac in results:
        if fusion and fusion[-1].symbol == ac.symbol and ac.start - fusion[-1].end < 0.15:
            fusion[-1].end = ac.end
        else:
            fusion.append(ac)
    return fusion

# --- Beat-based detection (root + quality by heuristics) ---
def detect_chords_beat(audio_file: str,
                          sr: int = 22050,
                          use_power: bool = True,
                          third_ratio: float = 0.18,
                          seventh_ratio: float = 0.15,
                          min_dur: float = 0.25,
                          min_delta: float = 0.15,
                          hold_time: float = 0.5,
                          energy_change_thresh: float = 0.28,
                          force_every_beat: bool = False) -> List[DetectedChord]:
    """Detects chords by beat interval.
    Heuristics:
      - Root: pitch class of highest chromatic energy + bass band reinforcement.
      - Third: compares M3 (4 semitones) vs m3 (3 semitones); if both weak and power enabled -> X5.
      - Seventh: if 10 (b7) or 11 (maj7) exceed threshold -> adds 7/m7/maj7.
      - sus4: if third weak and fourth strong (> third_ratio) -> sus4.
    Parameters:
      third_ratio: minimum relative energy to accept a third.
      seventh_ratio: minimum relative energy to accept a seventh.
      min_dur: minimum segment duration to be emitted.
      min_delta: fusion of changes too close.
      force_every_beat: if True, emits chord on every beat even if equal.
    """
    try:
        y, sr_used = librosa.load(audio_file, sr=sr)
    except Exception as e:
        print('❌ Error loading audio (beat):', e)
        return []
    try:
        y_h, _ = librosa.effects.hpss(y)
    except Exception:
        y_h = y
    # Chromas (CQT for better harmonic stability)
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr_used)
    # Bass energy (spectrum up to ~300 Hz) for root weighting
    S = np.abs(librosa.stft(y_h, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr_used, n_fft=2048)
    low_mask = freqs < 310
    S_low = S[low_mask]
    # Beat tracking
    tempo, beats = librosa.beat.beat_track(y=y_h, sr=sr_used, trim=False)
    if len(beats) < 2:
        # fallback: use frames every 0.5s
        hop_time = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr_used)
        beat_frames = np.arange(0, len(hop_time), max(1, int(0.5/(hop_time[1]-hop_time[0]))))
    else:
        beat_frames = beats
    beat_times = librosa.frames_to_time(beat_frames, sr=sr_used)
    if len(beat_times) < 2:
        return []
    # Function to name chord from root and energies
    def build_chord(root_idx: int, pc_vec: np.ndarray) -> str:
        # Energies normalized
        v = pc_vec / (pc_vec.sum()+1e-9)
        def rel(semi):
            return v[(root_idx + semi) % 12]
        maj3 = rel(4); min3 = rel(3); fifth = rel(7); fourth = rel(5)
        b7 = rel(10); maj7 = rel(11)
        name_root = NOTES[root_idx]
        quality = ''
        is_power = False
        # Decide third / sus / power
        if maj3 < third_ratio and min3 < third_ratio:
            if fourth > third_ratio and fourth > maj3 and fourth > min3:
                quality = 'sus4'
            elif use_power and fifth > max(maj3, min3) and fifth > 0.1:
                is_power = True
                quality = '5'
        else:
            if maj3 >= min3 and maj3 >= third_ratio:
                quality = ''  # major
            elif min3 > maj3 and min3 >= third_ratio:
                quality = 'm'
        # Seventh
        if not is_power:
            if quality in ('','m'):
                if b7 >= seventh_ratio and b7 > maj7:
                    quality += '7' if quality == '' else '7'
                elif maj7 >= seventh_ratio and maj7 > b7:
                    quality += 'maj7' if quality == '' else 'maj7'
                elif quality == 'm' and b7 >= seventh_ratio:
                    # m7 already covered above but for clarity
                    if not quality.endswith('7'):
                        quality += '7'
        return name_root + quality
    events: List[DetectedChord] = []
    prev_vec = None
    seg_start = None
    seg_chord = None
    seg_root = None
    last_commit_t = None
    for i in range(len(beat_frames)-1):
        a = beat_frames[i]; b = beat_frames[i+1]
        if b <= a: continue
        chroma_seg = chroma[:, a:b]
        if chroma_seg.size == 0: continue
        pc_sum = chroma_seg.sum(axis=1)
        # root weighting with bass
        a_low = int(a * 512 / 2048); b_low = int(b * 512 / 2048) + 1
        spec_low_seg = S_low[:, a_low:b_low]
        if spec_low_seg.size > 0:
            freq_bins = freqs[low_mask]
            midi_vals = librosa.hz_to_midi(freq_bins)
            pcs_low = (np.round(midi_vals).astype(int)) % 12
            low_vec = np.zeros(12)
            mag_low = spec_low_seg.mean(axis=1)
            for pc_i, val in zip(pcs_low, mag_low):
                if 0 <= pc_i < 12:
                    low_vec[pc_i] += val
            pc_sum = pc_sum + 0.8*low_vec
        root_idx = int(np.argmax(pc_sum))
        chord_name = build_chord(root_idx, pc_sum)
        t0 = beat_times[i]; t1 = beat_times[i+1]
        mid_t = (t0 + t1)/2
        vec_norm = pc_sum / (np.linalg.norm(pc_sum)+1e-9)
        if seg_chord is None:
            seg_chord = chord_name; seg_start = t0; seg_root = NOTES[root_idx]; prev_vec = vec_norm; last_commit_t = t0
            continue
        # Decide whether to change
        sim = float(np.dot(prev_vec, vec_norm)) if prev_vec is not None else 0
        changed = (chord_name != seg_chord)
        need_change = False
        if changed:
            # energy condition and temporal hold
            if (mid_t - (last_commit_t or seg_start)) >= hold_time and (1 - sim) >= energy_change_thresh:
                need_change = True
            elif force_every_beat:
                need_change = True
        if need_change:
            # close current segment
            if seg_chord is not None and seg_start is not None and seg_root is not None and t0 - seg_start >= min_dur:
                events.append(DetectedChord(seg_chord, float(seg_start), float(t0), [seg_root], 0.0))
                last_commit_t = t0
            else:
                # if too short force merge to new (discard)
                pass
            seg_chord = chord_name; seg_start = t0; seg_root = NOTES[root_idx]; prev_vec = vec_norm
        else:
            # prolong
            prev_vec = (prev_vec + vec_norm)/2 if prev_vec is not None else vec_norm
            continue
    # Close last
    if seg_chord and seg_start is not None and seg_root is not None and (beat_times[-1] - seg_start) >= min_dur:
        events.append(DetectedChord(seg_chord, float(seg_start), float(beat_times[-1]), [seg_root], 0.0))
    # Merge adjacent identical separated by small gaps
    fusion: List[DetectedChord] = []
    for ac in events:
        if fusion and ac.symbol == fusion[-1].symbol and ac.start - fusion[-1].end < min_delta:
            fusion[-1].end = ac.end
        else:
            fusion.append(ac)
    return fusion

# --- Hybrid method (bass + chromatic novelty) ---
def detect_chords_hybrid(audio_file: str,
                             sr: int = 22050,
                             hop_length: int = 2048,
                             novelty_th: float = 0.35,
                             min_chord_dur: float = 0.5,
                             force_change_beats: int = 0) -> List[DetectedChord]:
    """Hybrid: root by bass energy + quality by chromas with internal splits by novelty.
    novelty_th: chromatic flux threshold to subdivide a segment (0-1 approx, depends on scale).
    min_chord_dur: minimum resulting chord duration.
    force_change_beats: forces a re-check every N root segments even if no novelty.
    """
    try:
        y, sr_used = librosa.load(audio_file, sr=sr)
    except Exception as e:
        print('❌ Error loading audio (hybrid):', e)
        return []
    try:
        y_h, _ = librosa.effects.hpss(y)
    except Exception:
        y_h = y
    # Chroma and Bass STFT
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr_used, hop_length=hop_length)
    S = np.abs(librosa.stft(y_h, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr_used, n_fft=2048)
    low_mask = freqs < 400
    S_low = S[low_mask]
    freq_low = freqs[low_mask]
    midi_low = librosa.hz_to_midi(freq_low)
    pcs_low = (np.round(midi_low).astype(int)) % 12
    # Root by frame (low energy aggregation)
    low_energy_pc = np.zeros((12, S_low.shape[1]))
    for pc in range(12):
        mask = pcs_low == pc
        if np.any(mask):
            low_energy_pc[pc] = S_low[mask].sum(axis=0)
    root_idx_frames = np.argmax(low_energy_pc, axis=0)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr_used, hop_length=hop_length)
    # Novelty (chroma flux) normalized
    diff = np.diff(chroma, axis=1)
    flux = np.maximum(diff, 0).sum(axis=0)
    if flux.max() > 0:
        flux = flux / flux.max()
    # Segment by root
    segments = []  # (start_frame, end_frame, root_idx)
    cur_root = root_idx_frames[0]
    start_f = 0
    for f in range(1, len(root_idx_frames)):
        if root_idx_frames[f] != cur_root:
            segments.append((start_f, f, cur_root))
            cur_root = root_idx_frames[f]
            start_f = f
    segments.append((start_f, len(root_idx_frames), cur_root))
    # Subdivide segments by internal novelty
    refined = []
    for (a,b,rroot) in segments:
        last = a
        # iterate over flux indices (flux has len = frames-1)
        for f in range(a+1, b-1):
            if flux[f-1] >= novelty_th and (f - last) * (times[1]-times[0]) >= min_chord_dur/2:
                refined.append((last, f, rroot))
                last = f
        refined.append((last, b, rroot))
    # Force changes every N segments if requested
    if force_change_beats > 1:
        forced = []
        for idx,(a,b,r) in enumerate(refined):
            forced.append((a,b,r))
            if (idx+1) % force_change_beats == 0:
                # insert artificial split if enough dur
                mid = (a+b)//2
                if b - a > 3 and mid - a > 1 and b - mid > 1:
                    forced[-1] = (a, mid, r)
                    forced.append((mid, b, r))
        refined = forced
    # Quality classification
    results: List[DetectedChord] = []
    third_ratio = 0.15
    seventh_ratio = 0.13
    def build_quality(root_idx: int, pc_vec: np.ndarray) -> str:
        v = pc_vec / (pc_vec.sum()+1e-9)
        def rel(semi): return v[(root_idx+semi)%12]
        maj3, min3, fifth, fourth = rel(4), rel(3), rel(7), rel(5)
        b7, maj7 = rel(10), rel(11)
        name_root = NOTES[root_idx]
        # Power
        if maj3 < third_ratio and min3 < third_ratio and fifth > max(maj3,min3) and fifth > 0.12:
            return name_root + '5'
        # Sus4
        if fourth > maj3 and fourth > min3 and fourth > third_ratio:
            base = 'sus4'
        else:
            if maj3 >= min3 and maj3 >= third_ratio: base = ''
            elif min3 > maj3 and min3 >= third_ratio: base = 'm'
            else: base = '5' if fifth > 0.1 else ''
        # Seventh
        if base.endswith('5'):
            return name_root + base
        if b7 >= seventh_ratio and b7 > maj7:
            if base == 'm': return name_root + 'm7'
            if base == 'sus4': return name_root + '7sus4'
            return name_root + '7'
        if maj7 >= seventh_ratio and maj7 > b7:
            if base == 'm': return name_root + 'm(maj7)'
            if base == 'sus4': return name_root + 'maj7sus4'
            return name_root + 'maj7'
        return name_root + base
    frame_dur = times[1]-times[0] if len(times) > 1 else 1.0/sr_used
    for (a,b,rroot) in refined:
        t0 = times[a]; t1 = times[b-1] if b-1 < len(times) else times[-1]
        if t1 - t0 < min_chord_dur:
            continue
        pc_vec = chroma[:, a:b].sum(axis=1)
        chord = build_quality(rroot, pc_vec)
        sim_score = float(np.max(pc_vec))/ (pc_vec.sum()+1e-9) * 100.0
        results.append(DetectedChord(chord, float(t0), float(t1), [NOTES[rroot]], sim_score))
    # Merge adjacent equal
    fusion: List[DetectedChord] = []
    for ac in results:
        if fusion and fusion[-1].symbol == ac.symbol and ac.start - fusion[-1].end < 0.1:
            fusion[-1].end = ac.end
        else:
            fusion.append(ac)
    return fusion

def identify_chord_from_notes(notes: List[int]) -> Optional[str]:
    """
    Identifies chord from a set of MIDI notes (0-11).
    """
    # Normalize to pitch class (0-11)
    notes = sorted(set(n % 12 for n in notes))
    
    if len(notes) < 3:
        return None
    
    # Take lowest note as root
    root = notes[0]
    relative_notes = [(n - root) % 12 for n in notes]
    
    # Common chord patterns
    patterns = {
        (0, 4, 7): '',        # Major
        (0, 3, 7): 'm',       # Minor
        (0, 4, 7, 10): '7',   # Dom 7
        (0, 3, 7, 10): 'm7',  # Min 7
        (0, 4, 7, 11): 'maj7',# Maj 7
    }
    
    # Find matching pattern
    for pattern, suffix in patterns.items():
        if all(note in relative_notes for note in pattern):
            root_name = NOTES[root]
            return f"{root_name}{suffix}"
    
    # If no match, return root as major default
    return NOTES[root]

def detect_chords_from_midi(audio_file: str) -> List[DetectedChord]:
    """
    Detects chords by first converting audio to MIDI with Basic Pitch (Spotify).
    More precise than spectral analysis because it transcribes explicit notes.
    """
    try:
        print(f"🎹 Converting audio to MIDI with Basic Pitch...")
        
        from basic_pitch.inference import predict_and_save
        from basic_pitch import ICASSP_2022_MODEL_PATH
        import mido
        from collections import defaultdict
        
        # Create temp dir for MIDI
        output_dir = os.path.join(tempfile.gettempdir(), "basic_pitch_midi")
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert audio to MIDI
        predict_and_save(
            [audio_file],
            output_dir,
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=ICASSP_2022_MODEL_PATH
        )
        
        # Find generated MIDI file
        filename = os.path.splitext(os.path.basename(audio_file))[0]
        midi_file = os.path.join(output_dir, f"{filename}_basic_pitch.mid")
        
        if not os.path.exists(midi_file):
            print(f"⚠️ MIDI file not generated at: {midi_file}")
            return []
        
        print(f"✅ MIDI generated. Analyzing chords...")
        
        # Read MIDI file
        mid = mido.MidiFile(midi_file)
        
        # Analyze active notes at each moment
        detected_chords = []
        active_notes = defaultdict(list)  # time -> list of notes
        ticks_per_beat = mid.ticks_per_beat
        
        for track in mid.tracks:
            time_in_track = 0
            for msg in track:
                time_in_track += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    time_sec = mido.tick2second(time_in_track, ticks_per_beat, 500000)
                    note = msg.note % 12  # 0-11 (C, C#, D, etc.)
                    active_notes[round(time_sec, 1)].append(note)
                    
        # Group notes into chords
        times_sorted = sorted(active_notes.keys())
        for i, t in enumerate(times_sorted):
            notes = set(active_notes[t])
            if len(notes) >= 3:  # At least 3 notes for a chord
                chord = identify_chord_from_notes(list(notes))
                if chord:
                    detected_chords.append(DetectedChord(
                        symbol=chord,
                        start=t,
                        end=t+0.5, # approximation
                        notes=[],
                        percentage=90.0
                    ))
        
        # Merge/Clean up detected chords (simple logic)
        # ... (omitted for brevity, basic pitch is experimental here)
        
        print(f"✅ Detected {len(detected_chords)} chords from MIDI")
        return detected_chords
        
    except Exception as e:
        print(f"❌ Error in MIDI detection: {e}")
        import traceback
        traceback.print_exc()
        return []


# --- Essentia-based chord detection (C++ core, professional MIR) ---

# Genre-specific parameter presets for optimal chord detection
GENRE_PARAMS = {
    'rock': {
        'hpcp_size': 12,
        'magnitudeThreshold': 0.00002,  # Increased to ignore more noise
        'maxPeaks': 150,  # Capture more harmonics from distorted guitars
        'windowSize': 1.6,  # Reduced from 1.8 to catch faster changes
        'weightType': 'none',  # Don't weight harmonics (distortion creates unusual ratios)
        'min_duration': 0.4,  # Reduced from 0.5 to allow faster chords
        'bandPreset': False
    },
    'electronic': {
        'hpcp_size': 12,
        'magnitudeThreshold': 0.0001,  # Stricter (synths are clean)
        'maxPeaks': 60,  # Fewer peaks (clean synth tones)
        'windowSize': 2.5,  # Slower changes (typical 4/4 disco patterns)
        'weightType': 'cosine',
        'min_duration': 0.8,  # Longer sustained chords in electronic music
        'bandPreset': False
    },
    'acoustic': {
        'hpcp_size': 12,
        'magnitudeThreshold': 0.000001,  # Very sensitive for soft picking
        'maxPeaks': 100,
        'windowSize': 2.0,
        'weightType': 'cosine',
        'min_duration': 0.5,
        'bandPreset': False
    },
    'default': {
        'hpcp_size': 12,
        'magnitudeThreshold': 0.000001,
        'maxPeaks': 100,
        'windowSize': 2.0,
        'weightType': 'cosine',
        'min_duration': 0.5,
        'bandPreset': False
    }
}

def _detect_music_style(audio: np.ndarray, sr: int) -> str:
    """
    Detects music style to choose optimal chord detection parameters.
    
    Uses spectral features to classify:
    - 'rock': Distorted guitars (bright, noisy/flat spectrum, dynamic)
    - 'electronic': Synth-based (bright, clean spectrum, steady)
    - 'acoustic': Clean guitars (warm, moderate dynamics)
    - 'default': Fallback for unclear classification
    
    Args:
        audio: Audio signal
        sr: Sample rate
        
    Returns:
        Style identifier string
    """
    try:
        import essentia.standard as es
        
        # Extract spectral features
        spectral_centroid = es.SpectralCentroidTime()
        spectral_flatness = es.Flatness()  # High = noisy/distorted, Low = tonal/clean
        rms_energy = es.RMS()
        
        centroids = []
        flatnesses = []
        energies = []
        
        # Analyze frames
        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
            if len(frame) < 2048:
                continue
            windowed = es.Windowing(type='hann')(frame)
            spectrum = es.Spectrum()(windowed)
            centroids.append(spectral_centroid(spectrum))
            flatnesses.append(spectral_flatness(spectrum))
            energies.append(rms_energy(frame))
        
        if not centroids or not energies:
            return 'default'
        
        # Calculate statistics
        avg_centroid = float(np.mean(centroids))
        avg_flatness = float(np.mean(flatnesses))
        avg_energy = float(np.mean(energies))
        energy_variance = float(np.var(energies))
        
        # Debug logging
        print(f"   📊 Style detection: centroid={avg_centroid:.0f}Hz, flatness={avg_flatness:.3f}, energy_var={energy_variance:.4f}")
        
        # IMPROVED Classification heuristics
        # Rock/Punk/Grunge: Multiple conditions to catch different rock styles
        # 1. Noisy spectrum (high flatness from distortion)
        # 2. Very bright (loud guitars/cymbals) + some dynamics
        # 3. Moderately noisy + bright (borderline distortion)
        if (avg_flatness > 0.115 or  # Lowered from 0.125 to catch borderline grunge (e.g. 0.118)
            avg_centroid > 4500 or  # Very bright = loud rock/punk
            (avg_centroid > 2500 and energy_variance > 0.005)):
            return 'rock'
        
        # Electronic: Bright + clean (low flatness) + very steady
        # Synths are tonal (low flatness) and very consistent
        elif avg_centroid > 2000 and avg_flatness < 0.1 and energy_variance < 0.003:
            return 'electronic'
        
        # Acoustic: Warm (lower centroid) and clean
        elif avg_centroid < 2000 and avg_flatness < 0.12:
            return 'acoustic'
        
        # Default for ambiguous cases
        return 'default'
        
    except Exception as e:
        print(f"⚠️ Style detection failed: {e}, using default parameters")
        return 'default'

def detect_chords_essentia(audio_file: str) -> List[DetectedChord]:
    """
    Detects chords using Essentia's professional-grade chord detection.
    Uses HPCP (Harmonic Pitch Class Profiles) and ChordsDetection algorithm.
    
    Args:
        audio_file: Path to audio file
        
    Returns:
        List of DetectedChord objects
        
    Note:
        Very fast (C++ core) and accurate. Developed by Music Technology Group (MTG).
    """
    try:
        import essentia
        import essentia.standard as es
        print("🎼 Using Essentia ChordsDetection (MTG)...")
    except ImportError as e:
        print(f"❌ Essentia not installed: {e}")
        print("⚠️ Falling back to Librosa method...")
        return detect_chords_chroma_improved(audio_file)
    
    try:
        # Load audio with optimal sample rate for chord detection
        loader = es.MonoLoader(filename=audio_file, sampleRate=44100)
        audio = loader()
        
        if len(audio) == 0:
            print("⚠️ Essentia loaded empty audio, falling back to Librosa.")
            return detect_chords_chroma_improved(audio_file)
        
        # Detect music style for adaptive parameters
        sampleRate = 44100
        style = _detect_music_style(audio, sampleRate)
        params = GENRE_PARAMS[style]
        print(f"🎵 Detected style: {style.upper()} - using optimized parameters")
        
        # Processing parameters
        frameSize = 4096
        hopSize = 2048  # ~46ms temporal resolution
        
        # Frame processing
        windowing = es.Windowing(type='blackmanharris62')
        spectrum = es.Spectrum()
        
        # SpectralPeaks with adaptive parameters
        spectral_peaks = es.SpectralPeaks(
            orderBy='magnitude', 
            magnitudeThreshold=params['magnitudeThreshold'],
            minFrequency=50,
            maxFrequency=5000, 
            maxPeaks=params['maxPeaks']
        )
        
        # HPCP with adaptive parameters
        hpcp = es.HPCP(
            size=params['hpcp_size'],
            referenceFrequency=440,
            harmonics=8,
            bandPreset=params['bandPreset'],
            minFrequency=50,
            maxFrequency=5000,
            weightType=params['weightType'],
            nonLinear=False,
            windowSize=1.0
        )
        
        print(f"🎸 Processing {audio_file} with Essentia (optimized for strings)...")
        
        # Compute HPCPs for all frames
        hpcps = []
        for frame in es.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
            frame_windowed = windowing(frame)
            frame_spectrum = spectrum(frame_windowed)
            frequencies, magnitudes = spectral_peaks(frame_spectrum)
            hpcp_frame = hpcp(frequencies, magnitudes)
            hpcps.append(hpcp_frame)
        
        if len(hpcps) == 0:
            print("⚠️ No HPCPs computed, trying Librosa fallback...")
            return detect_chords_chroma_improved(audio_file)
        
        # Convert to numpy array (required by ChordsDetection)
        import numpy as np
        hpcps_matrix = np.array(hpcps, dtype='float32')
        
        # ChordsDetection with adaptive window size
        chords_detector = es.ChordsDetection(
            sampleRate=sampleRate, 
            hopSize=hopSize, 
            windowSize=params['windowSize']
        )
        
        # Get chords with native start/end times (more accurate than manual calculation)
        chords_raw, strengths_raw = chords_detector(hpcps_matrix)

        # Build initial chord list (one per frame)
        frame_chords: List[tuple] = []
        frame_duration = hopSize / sampleRate
        
        for i, (chord_label, strength) in enumerate(zip(chords_raw, strengths_raw)):
            if not chord_label or chord_label == "N":  # Skip noise/silence
                continue
            
            start_time = i * frame_duration
            end_time = start_time + frame_duration
            normalized_label = _normalize_essentia_chord(chord_label)
            root_note = chord_label.split(':')[0] if ':' in chord_label else chord_label[0]
            confidence = float(strength) * 100 if strength is not None else 90.0
            
            frame_chords.append((normalized_label, start_time, end_time, root_note, confidence))
        
        if not frame_chords:
            print("⚠️ Essentia returned no chords, falling back to Librosa.")
            return detect_chords_chroma_improved(audio_file)
        
        # MERGE consecutive identical chords to reduce granularity
        merged: List[DetectedChord] = []
        current_symbol = frame_chords[0][0]
        current_start = frame_chords[0][1]
        current_end = frame_chords[0][2]
        current_root = frame_chords[0][3]
        confidence_sum = frame_chords[0][4]
        count = 1
        
        for symbol, start, end, root, conf in frame_chords[1:]:
            if symbol == current_symbol:
                # Same chord - extend the end time
                current_end = end
                confidence_sum += conf
                count += 1
            else:
                # Different chord - save current and start new
                avg_confidence = confidence_sum / count
                merged.append(DetectedChord(
                    symbol=current_symbol,
                    start=current_start,
                    end=current_end,
                    notes=[current_root],
                    percentage=avg_confidence
                ))
                # Start new chord
                current_symbol = symbol
                current_start = start
                current_end = end
                current_root = root
                confidence_sum = conf
                count = 1
        
        # Add last chord
        avg_confidence = confidence_sum / count
        merged.append(DetectedChord(
            symbol=current_symbol,
            start=current_start,
            end=current_end,
            notes=[current_root],
            percentage=avg_confidence
        ))
        
        # Filter out very short chords using genre-specific threshold
        min_duration = params['min_duration']
        filtered = [chord for chord in merged if (chord.end - chord.start) >= min_duration]
        
        # SECOND PASS: Merge duplicates separated by short different chords
        # E.g., C (2s) -> D (0.3s) -> C (3s) becomes C (5.3s)
        result: List[DetectedChord] = []
        i = 0
        while i < len(filtered):
            current = filtered[i]
            
            # Look ahead for same chord after a short interruption
            j = i + 1
            while j < len(filtered) - 1:
                # Check if chord at j+1 is same as current
                if filtered[j + 1].symbol == current.symbol:
                    # Merge if interruption is short (< 1s)
                    interruption = filtered[j]
                    if (interruption.end - interruption.start) < 1.0:
                        # Extend current chord to end of j+1
                        current = DetectedChord(
                            symbol=current.symbol,
                            start=current.start,
                            end=filtered[j + 1].end,
                            notes=current.notes,
                            percentage=(current.percentage + filtered[j + 1].percentage) / 2
                        )
                        j += 2  # Skip both interruption and merged chord
                        continue
                break
            
            result.append(current)
            i = j if j > i + 1 else i + 1
        
        if not result:
            print("⚠️ All chords filtered out (too short), falling back to Librosa.")
            return detect_chords_chroma_improved(audio_file)
        
        print(f"✅ Essentia detected {len(result)} chord segments (merged from {len(frame_chords)} frames)")
        return result
        
    except Exception as e:
        print(f"❌ Error in Essentia chord detection: {e}")
        print("⚠️ Falling back to Librosa method...")
        import traceback
        traceback.print_exc()
        return detect_chords_chroma_improved(audio_file)

def _essentia_key_to_chord(key: str, scale: str) -> str:
    """
    Converts Essentia key/scale to chord symbol.
    Essentia returns key (A, B, C, etc.) and scale (major/minor).
    """
    if scale == "major":
        return key
    elif scale == "minor":
        return f"{key}m"
    else:
        return key

def _normalize_essentia_chord(label: str) -> str:
    """
    Convert Essentia chord label (e.g., "C:maj", "A:min", "D:7", "N")
    to the project's chord symbol format.
    """
    if label == "N":
        return "N"
    if ':' in label:
        root, quality = label.split(':', 1)
    else:
        root, quality = label, ''
    quality_map = {
        'maj': '',
        'min': 'm',
        'm': 'm',
        '7': '7',
        'maj7': 'maj7',
        'min7': 'm7',
        'm7': 'm7',
        'dim': 'dim',
        'aug': 'aug',
        'sus4': 'sus4',
    }
    mapped = quality_map.get(quality, quality)
    return f"{root}{mapped}"


def detect_chords_chroma_improved(audio_file: str, sr: int = 22050) -> List[DetectedChord]:
    """
    Improved chord detection using Chroma CQT and temporal smoothing.
    Includes prior audio cleaning (Demucs or HPSS).
    """
    try:
        # 1. Use provided audio directly
        # Assumes caller (generate_video.py) already handled separation if needed.
        audio_to_process = audio_file
        
        # Determine if HPSS cleaning is needed
        # If file has "Instrumental" or "other" in name, assume it's a clean stem
        if "Instrumental" in audio_file or "other" in audio_file:
            print("✨ Using pre-separated audio (stem).")
            use_hpss = False
        else:
            print("⚠️ Using standard audio (applying HPSS cleaning).")
            use_hpss = True
            
        y, sr_used = librosa.load(audio_to_process, sr=sr)
        
        # 2. Additional cleaning if not using Demucs
        if use_hpss:
            from services.audio_extractor import clean_audio_for_chords
            y_clean = clean_audio_for_chords(y, sr_used)
        else:
            y_clean = y # Already 'other', should not have percussion
        
        # 3. Calculate CQT Chroma (better resolution for music)
        # Larger hop length for smoothing (approx 0.1s)
        hop_length = 2048 
        chroma = librosa.feature.chroma_cqt(y=y_clean, sr=sr_used, hop_length=hop_length, n_chroma=12)
        
        # 4. Temporal smoothing (median filter)
        # Smooth over ~0.5 seconds (5 frames approx if hop=2048 @ 22050 -> ~0.1s/frame)
        chroma_smooth = scipy.ndimage.median_filter(chroma, size=(1, 9))
        
        # 4. Matching with templates
        # Simple Major and Minor templates
        templates = {}
        for i, note in enumerate(NOTES):
            # Major: 0, 4, 7
            vec = np.zeros(12)
            vec[i] = 1.0
            vec[(i+4)%12] = 0.8
            vec[(i+7)%12] = 0.8
            templates[note] = vec / np.linalg.norm(vec)
            
            # Minor: 0, 3, 7
            vec = np.zeros(12)
            vec[i] = 1.0
            vec[(i+3)%12] = 0.8
            vec[(i+7)%12] = 0.8
            templates[note+'m'] = vec / np.linalg.norm(vec)
            
        # Calculate similarity
        best_score = -1
        best_chords = []
        
        chord_names = list(templates.keys())
        template_mat = np.array([templates[c] for c in chord_names]) # (24, 12)
        
        # Normalize chroma
        chroma_norm = chroma_smooth / (np.linalg.norm(chroma_smooth, axis=0) + 1e-9)
        
        # Dot product (Cosine Similarity)
        similarity = np.dot(template_mat, chroma_norm) # (24, T)
        
        # Get best chord per frame
        max_idx = np.argmax(similarity, axis=0)
        max_val = np.max(similarity, axis=0)
        
        # 5. Convert to temporal events
        times = librosa.frames_to_time(np.arange(similarity.shape[1]), sr=sr_used, hop_length=hop_length)
        
        detected_chords = []
        current_chord = None
        start_time = 0.0
        
        # Confidence threshold
        MIN_CONFIDENCE = 0.6
        MIN_DURATION = 0.5 # seconds
        
        for t, idx, val in zip(times, max_idx, max_val):
            chord = chord_names[idx]
            
            if val < MIN_CONFIDENCE:
                chord = None # Silence or uncertainty
            
            if chord != current_chord:
                # Close previous
                if current_chord is not None:
                    duration = t - start_time
                    if duration >= MIN_DURATION:
                        detected_chords.append(DetectedChord(
                            symbol=current_chord,
                            start=start_time,
                            end=t,
                            notes=[], # We don't calculate individual notes here
                            percentage=float(val)*100
                        ))
                
                current_chord = chord
                start_time = t
                
        # Close last
        if current_chord is not None:
            detected_chords.append(DetectedChord(
                symbol=current_chord,
                start=start_time,
                end=times[-1],
                notes=[],
                percentage=0.0
            ))
            
        return detected_chords
        
    except Exception as e:
        print(f"❌ Error in improved detection: {e}")
        return []
