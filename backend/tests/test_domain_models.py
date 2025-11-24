import pytest
from domain.models import DetectedChord, VideoAnalysis, AnalysisState

def test_detected_chord_definition():
    """Test that DetectedChord returns correct definitions."""
    chord = DetectedChord("C", 0, 1, [], 100)
    definition = chord.get_definition()
    # C Major uke definition: 0 0 0 3
    assert definition == [0, 0, 0, 3]

def test_detected_chord_diagram():
    """Test that DetectedChord returns diagram data."""
    chord = DetectedChord("C", 0, 1, [], 100)
    diagram = chord.get_diagram_data()
    assert len(diagram) > 0
    assert "G C E A" in diagram

def test_video_analysis_lifecycle(temp_dir):
    """Test VideoAnalysis state transitions."""
    import os
    input_path = os.path.join(temp_dir, "in.mp4")
    output_path = os.path.join(temp_dir, "out.mp4")
    
    # Create dummy files
    with open(input_path, "w") as f: f.write("dummy")
    
    analysis = VideoAnalysis(input_path=input_path)
    assert analysis.state == AnalysisState.CREATED
    
    # Set Audio
    audio_path = os.path.join(temp_dir, "audio.wav")
    with open(audio_path, "w") as f: f.write("dummy")
    analysis.set_audio(audio_path)
    assert analysis.state == AnalysisState.AUDIO_EXTRACTED
    assert analysis.audio_path == audio_path
    
    # Set Chords
    chords = [DetectedChord("C", 0, 1, [], 100)]
    analysis.set_chords(chords)
    assert analysis.state == AnalysisState.CHORDS_DETECTED
    assert len(analysis.chords) == 1
    
    # Complete (requires output file to exist)
    analysis.output_path = output_path
    with open(output_path, "w") as f: f.write("dummy")
    analysis.complete()
    assert analysis.state == AnalysisState.VIDEO_GENERATED

def test_video_analysis_fail():
    """Test VideoAnalysis failure state."""
    analysis = VideoAnalysis(input_path="dummy")
    analysis.fail("Something went wrong")
    assert analysis.state == AnalysisState.FAILED
    assert analysis.error == "Something went wrong"
