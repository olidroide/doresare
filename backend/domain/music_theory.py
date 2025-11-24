import math
from typing import Dict, List
from pychord import Chord

NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
A4 = 440.0

# Simple ASCII diagrams (G C E A tuning). Each pattern: G C E A
DIAGRAMS: Dict[str, List[str]] = {
    'C':  ["Pattern: 0 0 0 3","G C E A","     ─┬─","G: 0","C: 0","E: 0","A: 3"],
    'Cm': ["Pattern: 0 3 3 3","G C E A","     ─┬─","G: 0","C: 3","E: 3","A: 3"],
    'D':  ["Pattern: 2 2 2 0","G C E A","     ─┬─","G: 2","C: 2","E: 2","A: 0"],
    'Dm': ["Pattern: 2 2 1 0","G C E A","     ─┬─","G: 2","C: 2","E: 1","A: 0"],
    'E':  ["Pattern: 1 4 0 2","G C E A","     ─┬─","G: 1","C: 4","E: 0","A: 2"],
    'Em': ["Pattern: 0 4 3 2","G C E A","     ─┬─","G: 0","C: 4","E: 3","A: 2"],
    'F':  ["Pattern: 2 0 1 0","G C E A","     ─┬─","G: 2","C: 0","E: 1","A: 0"],
    'Fm': ["Pattern: 1 0 1 3","G C E A","     ─┬─","G: 1","C: 0","E: 1","A: 3"],
    'G':  ["Pattern: 0 2 3 2","G C E A","     ─┬─","G: 0","C: 2","E: 3","A: 2"],
    'Gm': ["Pattern: 0 2 3 1","G C E A","     ─┬─","G: 0","C: 2","E: 3","A: 1"],
    'A':  ["Pattern: 2 1 0 0","G C E A","     ─┬─","G: 2","C: 1","E: 0","A: 0"],
    'Am': ["Pattern: 2 0 0 0","G C E A","     ─┬─","G: 2","C: 0","E: 0","A: 0"],
    'B':  ["Pattern: 4 3 2 2","G C E A","     ─┬─","G: 4","C: 3","E: 2","A: 2"],
    'Bm': ["Pattern: 4 2 2 2","G C E A","     ─┬─","G: 4","C: 2","E: 2","A: 2"],
}

# Extensions 7th / maj7 / m7 / sus4 (common ukulele shapes)
DIAGRAMS.update({
    # C family
    'C7':    ["Pattern: 0 0 0 1","G C E A","     ─┬─","G: 0","C: 0","E: 0","A: 1"],
    'Cmaj7': ["Pattern: 0 0 0 2","G C E A","     ─┬─","G: 0","C: 0","E: 0","A: 2"],
    'Cm7':   ["Pattern: 3 3 3 3","G C E A","     ─┬─","G: 3","C: 3","E: 3","A: 3"],
    'Csus4': ["Pattern: 0 0 1 3","G C E A","     ─┬─","G: 0","C: 0","E: 1","A: 3"],
    # D family
    'D7':    ["Pattern: 2 2 2 3","G C E A","     ─┬─","G: 2","C: 2","E: 2","A: 3"],
    'Dmaj7': ["Pattern: 2 2 2 4","G C E A","     ─┬─","G: 2","C: 2","E: 2","A: 4"],
    'Dm7':   ["Pattern: 2 2 1 3","G C E A","     ─┬─","G: 2","C: 2","E: 1","A: 3"],
    'Dsus4': ["Pattern: 2 2 3 0","G C E A","     ─┬─","G: 2","C: 2","E: 3","A: 0"],
    # E family
    'E7':    ["Pattern: 1 2 0 2","G C E A","     ─┬─","G: 1","C: 2","E: 0","A: 2"],
    'Emaj7': ["Pattern: 1 3 0 2","G C E A","     ─┬─","G: 1","C: 3","E: 0","A: 2"],
    'Em7':   ["Pattern: 0 2 0 2","G C E A","     ─┬─","G: 0","C: 2","E: 0","A: 2"],
    'Esus4': ["Pattern: 2 4 0 2","G C E A","     ─┬─","G: 2","C: 4","E: 0","A: 2"],
    # F family
    'F7':    ["Pattern: 2 3 1 3","G C E A","     ─┬─","G: 2","C: 3","E: 1","A: 3"],
    'Fmaj7': ["Pattern: 2 0 1 2","G C E A","     ─┬─","G: 2","C: 0","E: 1","A: 2"],
    'Fm7':   ["Pattern: 1 3 1 3","G C E A","     ─┬─","G: 1","C: 3","E: 1","A: 3"],
    'Fsus4': ["Pattern: 3 0 1 1","G C E A","     ─┬─","G: 3","C: 0","E: 1","A: 1"],
    # G family
    'G7':    ["Pattern: 0 2 1 2","G C E A","     ─┬─","G: 0","C: 2","E: 1","A: 2"],
    'Gmaj7': ["Pattern: 0 2 2 2","G C E A","     ─┬─","G: 0","C: 2","E: 2","A: 2"],
    'Gm7':   ["Pattern: 0 2 1 1","G C E A","     ─┬─","G: 0","C: 2","E: 1","A: 1"],
    'Gsus4': ["Pattern: 0 2 3 3","G C E A","     ─┬─","G: 0","C: 2","E: 3","A: 3"],
    # A family
    'A7':    ["Pattern: 0 1 0 0","G C E A","     ─┬─","G: 0","C: 1","E: 0","A: 0"],
    'Amaj7': ["Pattern: 1 1 0 0","G C E A","     ─┬─","G: 1","C: 1","E: 0","A: 0"],
    'Am7':   ["Pattern: 0 0 0 0","G C E A","     ─┬─","G: 0","C: 0","E: 0","A: 0"],
    'Asus4': ["Pattern: 2 2 0 0","G C E A","     ─┬─","G: 2","C: 2","E: 0","A: 0"],
    # B family
    'B7':    ["Pattern: 2 3 2 3","G C E A","     ─┬─","G: 2","C: 3","E: 2","A: 3"],
    'Bmaj7': ["Pattern: 3 3 2 2","G C E A","     ─┬─","G: 3","C: 3","E: 2","A: 2"],
    'Bm7':   ["Pattern: 2 2 2 2","G C E A","     ─┬─","G: 2","C: 2","E: 2","A: 2"],
    'Bsus4': ["Pattern: 4 4 2 2","G C E A","     ─┬─","G: 4","C: 4","E: 2","A: 2"],
})

# Chord definitions (Fret positions: G C E A)
# 0 = open string, -1 = muted (not used in uke usually), >0 = fret number
CHORD_DEFINITIONS = {
    # Major
    'C': [0, 0, 0, 3],
    'D': [2, 2, 2, 0],
    'E': [4, 4, 4, 7], # More common and easier (barre fret 4)
    'F': [2, 0, 1, 0],
    'G': [0, 2, 3, 2],
    'A': [2, 1, 0, 0],
    'B': [4, 3, 2, 2],
    
    # Minor
    'Cm': [0, 3, 3, 3],
    'Dm': [2, 2, 1, 0],
    'Em': [0, 4, 3, 2],
    'Fm': [1, 0, 1, 3],
    'Gm': [0, 2, 3, 1],
    'Am': [2, 0, 0, 0],
    'Bm': [4, 2, 2, 2],
    
    # 7th
    'C7': [0, 0, 0, 1],
    'D7': [2, 2, 2, 3],
    'E7': [1, 2, 0, 2],
    'F7': [2, 3, 1, 3],
    'G7': [0, 2, 1, 2],
    'A7': [0, 1, 0, 0],
    'B7': [2, 3, 2, 3],
    
    # Minor 7th
    'Cm7': [3, 3, 3, 3],
    'Dm7': [2, 2, 1, 3],
    'Em7': [0, 2, 0, 2],
    'Fm7': [1, 3, 1, 3],
    'Gm7': [0, 2, 1, 1],
    'Am7': [0, 0, 0, 0], # Open
    'Bm7': [2, 2, 2, 2], # Barre 2
    
    # Major 7th
    'Cmaj7': [0, 0, 0, 2],
    'Dmaj7': [2, 2, 2, 4],
    'Emaj7': [1, 3, 0, 2],
    'Fmaj7': [2, 4, 1, 3], # or 5500
    'Gmaj7': [0, 2, 2, 2],
    'Amaj7': [1, 1, 0, 0],
    'Bmaj7': [3, 3, 2, 2],
    
    # Sharp/Flat equivalents (handled by logic, but some explicit)
    'Bb': [3, 2, 1, 1],
    'Bbm': [3, 1, 1, 1],
    'Eb': [0, 3, 3, 1], # or 3336
    'Ab': [5, 3, 4, 3], # or 1343? 5343 is common G shape moved up
    'Db': [1, 1, 1, 4],
    'Gb': [3, 1, 2, 1],
    
    # Sharp equivalents
    'C#': [1, 1, 1, 4], # Db
    'D#': [0, 3, 3, 1], # Eb
    'F#': [3, 1, 2, 1], # Gb
    'G#': [5, 3, 4, 3], # Ab
    'A#': [3, 2, 1, 1], # Bb
    
    # Sharp Minor
    'C#m': [1, 1, 0, 4], # Dbm (approx) or 6444
    'D#m': [3, 3, 2, 1], # Ebm
    'F#m': [2, 1, 2, 0],
    'G#m': [4, 3, 4, 2], # Abm
    'A#m': [3, 1, 1, 1], # Bbm
}

def get_chord_notes(chord_name: str) -> List[str]:
    """
    Get component notes of a chord using PyChord library.
    Returns list of note names (e.g., ['C', 'E', 'G'] for C major).
    """
    try:
        chord = Chord(chord_name)
        return chord.components()
    except:
        # Fallback: return empty list for unknown chords
        return []

def get_chord_definition(chord_name: str) -> List[int]:
    """Returns the chord definition (frets G C E A) for ukulele."""
    # 1. Direct lookup
    if chord_name in CHORD_DEFINITIONS:
        return CHORD_DEFINITIONS[chord_name]
        
    # 2. Try PyChord's enharmonic equivalents (C# == Db handled automatically)
    # PyChord normalizes chord names, so we can try to find synonyms
    try:
        chord = Chord(chord_name)
        # Try the normalized version
        normalized = str(chord)
        if normalized in CHORD_DEFINITIONS and normalized != chord_name:
            return CHORD_DEFINITIONS[normalized]
    except:
        pass
    
    # 3. Fallback heuristics for complex chords
    # Strip 7th extensions if not found
    if '7' in chord_name:
        simple_name = chord_name.replace('7', '').replace('maj', '').replace('min', 'm')
        if simple_name in CHORD_DEFINITIONS:
            return CHORD_DEFINITIONS[simple_name]
            
    return CHORD_DEFINITIONS.get('C', [0,0,0,3]) # Default to C to avoid crash

def frequency_to_note(f: float) -> str:
    if f <= 0: return 'Silence'
    n = int(round(12 * math.log2(f / A4)))
    note = NOTES[(n + 9) % 12]
    octave = 4 + (n + 9) // 12
    return f"{note}{octave}"

def base(n: str) -> str:
    if n == 'Silence': return n
    return n[:2] if len(n) >= 2 and n[1] == '#' else n[0]
