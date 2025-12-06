import tempfile
import os
import numpy as np
from typing import List, Tuple, Optional
import pathlib
import pathlib

from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ColorClip, ImageClip, VideoClip
from proglog import ProgressBarLogger

from domain.models import DetectedChord
from domain.music_theory import get_chord_definition
from services.font_manager import FontManager

class ChordRenderer:
    def __init__(self, font_manager: FontManager):
        self.width = 300
        self.height = 400
        self.padding = 20
        self.fret_count = 5
        self.string_count = 4
        
        # Colors
        self.bg_color = (255, 255, 255, 255) # White
        self.line_color = (50, 50, 50, 255) # Dark Grey
        self.dot_color = (255, 140, 0, 255) # Orange
        self.text_color = (0, 0, 0, 255) # Black
        self.muted_color = (200, 50, 50, 255) # Red X
        
        # Font loading via FontManager
        self.font_title = font_manager.get_pillow_font(40)
        self.font_fret = font_manager.get_pillow_font(20)



    def render_chord(self, chord_name: str, frets: List[int] = None) -> Image.Image:
        if frets is None:
            frets = get_chord_definition(chord_name)
            
        img = Image.new('RGBA', (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Title
        bbox = draw.textbbox((0, 0), chord_name, font=self.font_title)
        text_w = bbox[2] - bbox[0]
        draw.text(((self.width - text_w) / 2, 10), chord_name, font=self.font_title, fill=self.text_color)
        
        # Grid dimensions
        grid_top = 80
        grid_bottom = self.height - 40
        grid_left = 40
        grid_right = self.width - 40
        
        fret_spacing = (grid_bottom - grid_top) / self.fret_count
        string_spacing = (grid_right - grid_left) / (self.string_count - 1)
        
        # Draw Frets (Horizontal)
        for i in range(self.fret_count + 1):
            y = grid_top + i * fret_spacing
            width = 3 if i == 0 else 1 # Nut is thicker
            draw.line([(grid_left, y), (grid_right, y)], fill=self.line_color, width=width)
            
        # Draw Strings (Vertical)
        for i in range(self.string_count):
            x = grid_left + i * string_spacing
            draw.line([(x, grid_top), (x, grid_bottom)], fill=self.line_color, width=1)
            
        # Draw Dots
        dot_radius = 12
        
        for i, fret in enumerate(frets):
            x = grid_left + i * string_spacing
            
            if fret > 0:
                # Calculate y position (middle of fret)
                y = grid_top + (fret - 0.5) * fret_spacing
                
                # Draw circle
                draw.ellipse([(x - dot_radius, y - dot_radius), 
                              (x + dot_radius, y + dot_radius)], 
                             fill=self.dot_color, outline=None)
                
            elif fret == 0:
                # Open string circle at top
                y = grid_top - 15
                draw.ellipse([(x - 5, y - 5), (x + 5, y + 5)], 
                             outline=self.line_color, width=1)
                             
        return img

    def save_chord(self, chord_name: str, path: str):
        img = self.render_chord(chord_name)
        img.save(path)

# Global renderer instance REMOVED to enforce dependency injection
# renderer = ChordRenderer()



def create_chord_clip(text: str, start: float, duration: float, 
                     font_manager: FontManager,
                     fontsize: int = 80, color: str = 'yellow', 
                     pos: Tuple = ('center', 'top'), 
                     bg_color: Optional[str] = None) -> CompositeVideoClip:
    """
    Creates a text clip for the chord using the unified styled helper.
    """
    font_path = font_manager.get_font_path()
    
    # Use the unified helper with glow effect
    # Main chord gets a nice cyan/gold glow or similar. Let's use cyan for contrast with yellow.
    # Or maybe orange to match the dots? Let's stick to the requested 'glow' generic or a specific one.
    # User asked for "bloom or glow".
    
    composite = create_styled_text_clip(
        text=text,
        fontsize=fontsize,
        color=color,
        font_path=font_path,
        duration=duration,
        bg_opacity=0.6,
        glow_color='orange' # Orange glow for the main chord (yellow text) looks fiery/cool
    )
    
    composite = composite.with_start(start).with_position(pos)
    
    return composite

def create_diagram_clip(chord_name: str, start: float, duration: float,
                       pos: Tuple = ('left', 'top'), width: int = 200,
                       file_manager=None, work_dir=None,
                       renderer: ChordRenderer = None) -> ImageClip:
    """Creates a clip with the graphical diagram of the chord."""
    
    if not renderer:
        raise ValueError("renderer instance is required")

    # Generate temporary image
    if file_manager:
        path = file_manager.create_temp_file(suffix='.png', directory=work_dir)
    else:
        fd, path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
    
    renderer.save_chord(chord_name, path)
    
    # Create ImageClip
    img_clip = ImageClip(path).with_start(start).with_duration(duration)
    
    # Resize
    img_clip = img_clip.resized(width=width)
    
    img_clip = img_clip.with_position(pos)
    
    return img_clip
    
# ... (TimelineClip and MoviePyProgressLogger remain unchanged) ...

def create_styled_text_clip(text: str, fontsize: int, color: str, font_path: str, duration: float, 
                           bg_opacity: float = 0.6, glow_color: Optional[str] = None) -> CompositeVideoClip:
    """
    Unified helper for creating styled text clips with background, centering, and optional glow.
    
    Args:
        text: The text to render.
        fontsize: Font size in pixels.
        color: Main text color.
        font_path: Path to the font file.
        duration: Duration of the clip in seconds.
        bg_opacity: Opacity of the black background box (0.0 to 1.0).
        glow_color: Color of the glow effect (e.g., 'cyan', 'gold'). If None, no glow is applied.
        
    Returns:
        CompositeVideoClip: The composed text clip.
    """
    # 1. Calculate exact text size using PIL
    try:
        pil_font = ImageFont.truetype(font_path, fontsize)
    except:
        pil_font = ImageFont.load_default()
            
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = dummy_draw.textbbox((0, 0), text, font=pil_font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Padding for the TextClip itself (to avoid clipping)
    # Restoring generous padding logic that worked previously
    text_padding = int(fontsize * 0.8) # Increased padding for safety
    clip_w = text_w + text_padding * 2
    clip_h = text_h + text_padding * 2
    
    # Background dimensions (can be slightly tighter than the full clip if desired, 
    # but let's make it wrap the text comfortably)
    bg_padding_x = int(fontsize * 0.4)
    bg_padding_y = int(fontsize * 0.2)
    bg_w = text_w + bg_padding_x * 2
    bg_h = text_h + bg_padding_y * 2
    
    
    # Let's stick to a safe background size that looks good
    bg_w = max(bg_w, clip_w * 0.8) # Ensure it's not too small
    bg_h = max(bg_h, clip_h * 0.6)

    # 2. Create Background
    # bg_clip = ColorClip(size=(int(bg_w), int(bg_h)), color=(0,0,0), duration=duration)
    # bg_clip = bg_clip.with_opacity(bg_opacity)
    
    # layers = [bg_clip]
    layers = []
    
    # 3. Create Glow Layer (Optional)
    if glow_color:
        txt_glow = TextClip(
            text=text,
            font_size=fontsize,
            color=glow_color,
            stroke_color=glow_color,
            stroke_width=5, # Thick stroke for glow
            font=font_path,
            method='caption',
            size=(int(clip_w), int(clip_h)), # Use full padded size
            text_align='center'
        )
        txt_glow = txt_glow.with_opacity(0.4) 
        txt_glow = txt_glow.with_position('center')
        layers.append(txt_glow)
        
    # 4. Create Main Text (Sharp)
    txt_main = TextClip(
        text=text,
        font_size=fontsize,
        color=color,
        font=font_path,
        method='caption',
        size=(int(clip_w), int(clip_h)), # Use full padded size
        text_align='center'
    )
    txt_main = txt_main.with_position('center')
    layers.append(txt_main)
    
    # 5. Compose
    # The composite size determines the final "box" size. 
    # If we want the background to be the boundary, we use bg_w/bg_h.
    # The large text clips will be centered on it. 
    # Note: If TextClip is larger than Composite size, it might get cropped by Composite?
    # Yes, CompositeVideoClip clips to its size.
    # So Composite size MUST be >= TextClip size if we want to see everything.
    
    # Correction: The background box should be the visual container. 
    # If we want the text to not clip, the Composite must be large enough.
    # But if we make the Composite huge, the background (ColorClip) needs to be centered or sized to that?
    # No, we want the result to be a "box" with text.
    
    # Let's make the Composite size equal to the TextClip size (safe size), 
    # and size the ColorClip (background) to be smaller (visual box) and center it.
    
    final_w = clip_w
    final_h = clip_h
    
    # Re-create background with visual size, but centered in the large safe area
    bg_clip = ColorClip(size=(int(bg_w), int(bg_h)), color=(0,0,0), duration=duration)
    bg_clip = bg_clip.with_opacity(bg_opacity)
    bg_clip = bg_clip.with_position('center')
    
    # Re-build layers
    layers = [bg_clip]
    if glow_color:
        layers.append(txt_glow) # txt_glow is size=final_w/h
    layers.append(txt_main)     # txt_main is size=final_w/h
    
    composite = CompositeVideoClip(layers, size=(int(final_w), int(final_h)))
    composite = composite.with_duration(duration)
    
    return composite


class TimelineClip(VideoClip):
    def __init__(self, size, duration, chords, speed=200, font_manager: FontManager = None):
        """
        Timeline scrolling clip.
        speed: pixels per second scrolling speed.
        """
        VideoClip.__init__(self)
        self.size = size
        self.duration = duration
        self.chords = chords
        self.speed = speed
        self.center_x = size[0] // 2
        
        # Explicitly set frame_function for MoviePy to work with subclassing
        self.frame_function = self.make_frame
        
        # Pre-render chord names to images to avoid re-rendering text every frame
        self.chord_images = {}
        
        # Font loading via FontManager
        if font_manager:
            font = font_manager.get_pillow_font(30)
        else:
            font = ImageFont.load_default()

        for ac in chords:
            if ac.symbol not in self.chord_images:
                img = Image.new('RGBA', (100, 50), (0,0,0,0))
                draw = ImageDraw.Draw(img)
                # Draw circle background
                draw.ellipse([(25, 0), (75, 50)], fill=(0, 100, 255, 200))
                # Draw text
                bbox = draw.textbbox((0,0), ac.symbol, font=font)
                tw = bbox[2]-bbox[0]
                th = bbox[3]-bbox[1]
                draw.text((50 - tw/2, 25 - th/2 - 5), ac.symbol, font=font, fill='white')
                self.chord_images[ac.symbol] = np.array(img)

    def make_frame(self, t):
        """Draw the timeline at time t."""
        # Create base transparent image
        # Using numpy for speed
        w, h = self.size
        frame = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Draw horizontal line
        line_y = h // 2
        frame[line_y-2:line_y+2, :, 0:3] = 255 # White line
        frame[line_y-2:line_y+2, :, 3] = 200   # Alpha
        
        # Draw hit marker at center
        frame[line_y-20:line_y+20, self.center_x-2:self.center_x+2, 0] = 0   # Red marker
        frame[line_y-20:line_y+20, self.center_x-2:self.center_x+2, 1] = 255 # Green marker (Yellowish)
        frame[line_y-20:line_y+20, self.center_x-2:self.center_x+2, 2] = 0
        frame[line_y-20:line_y+20, self.center_x-2:self.center_x+2, 3] = 255
        
        # Draw chords
        # Visible window: t +/- (width/2)/speed
        window_half_dur = (w / 2) / self.speed
        visible_start = t - window_half_dur
        visible_end = t + window_half_dur
        
        for ac in self.chords:
            if ac.start > visible_end:
                break 
            if ac.start < visible_start:
                continue
                
            rel_time = ac.start - t
            x = int(self.center_x + rel_time * self.speed)
            
            # Draw chord image centered at x
            c_img = self.chord_images[ac.symbol]
            h_img, w_img = c_img.shape[:2]
            
            x_img = x - w_img // 2
            y_img = line_y - h_img // 2
            
            # Boundary checks
            if x_img >= w or x_img + w_img <= 0:
                continue
                
            # Blending (Manual alpha blending for numpy)
            x1 = max(0, x_img)
            y1 = max(0, y_img)
            x2 = min(w, x_img + w_img)
            y2 = min(h, y_img + h_img)
            
            img_x1 = x1 - x_img
            img_y1 = y1 - y_img
            img_x2 = img_x1 + (x2 - x1)
            img_y2 = img_y1 + (y2 - y1)
            
            if x2 > x1 and y2 > y1:
                overlay = c_img[img_y1:img_y2, img_x1:img_x2]
                alpha_s = overlay[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                
                for c in range(3):
                    frame[y1:y2, x1:x2, c] = (alpha_s * overlay[:, :, c] +
                                              alpha_l * frame[y1:y2, x1:x2, c])
                frame[y1:y2, x1:x2, 3] = np.maximum(frame[y1:y2, x1:x2, 3], overlay[:, :, 3])

        return frame

class MoviePyProgressLogger(ProgressBarLogger):
    """
    Custom logger wrapper for MoviePy that reports progress via callback.
    """
    def __init__(self, progress_callback=None, start_pct=0.0, end_pct=1.0, init_state=None, **kwargs):
        super().__init__(init_state=init_state, **kwargs)
        self.progress_callback = progress_callback
        self.start_pct = start_pct
        self.end_pct = end_pct
        self.messages = []
        print(f"🎬 MoviePy Logger initialized (Range: {start_pct}-{end_pct})")

    def callback(self, **changes):
        """Called every time a simple log message is updated."""
        for parameter, value in changes.items():
            if parameter == 'message':
                self.messages.append(value)
                print(f"📝 MoviePy: {value}")

    def bars_callback(self, bar, attr, value, old_value=None):
        """
        Called every time a progress bar is updated.
        """
        super().bars_callback(bar, attr, value, old_value)
        
        if self.progress_callback and attr == 'index' and bar == 'frame_index':
            if bar in self.bars and 'total' in self.bars[bar]:
                index = self.bars[bar]['index']
                total = self.bars[bar]['total']
                
                if total > 0:
                    percentage = (index / total) * 100
                    # Call the progress callback with percentage and bar name
                    
                    """
                    Callback invoked by MoviePyProgressLogger.
                    Maps MoviePy progress (0-100%) to Gradio progress range (start_pct to end_pct).
                    """
                    pct = percentage / 100.0
                    overall = self.start_pct + (pct * (self.end_pct - self.start_pct))
                    
                    # Create descriptive message
                    msg = f"Rendering video: {int(percentage)}%"
                    if bar == 'frame_index':  # frame_index is the video frame bar
                        msg += f" (Frame {index}/{total})"
                    
                    # Report to Gradio
                    self.progress_callback(overall, desc=msg)

from domain.models import VideoAnalysis

def render_video_with_overlays(analysis: VideoAnalysis, progress=None, start_pct=0.6, end_pct=1.0, file_manager=None, work_dir=None, font_manager: FontManager = None):
    """
    Generates the final video with chord overlays using VideoAnalysis aggregate.
    """
    if not font_manager:
        raise ValueError("font_manager is required")

    # Instantiate renderer with injected font manager
    renderer = ChordRenderer(font_manager)

    try:
        video = VideoFileClip(analysis.input_path)
        w, h = video.w, video.h
        
        clips = [video]
        
        # Timeline Clip (Bottom layer overlay)
        timeline_h = 100
        timeline = TimelineClip(size=(w, timeline_h), duration=video.duration, chords=analysis.chords, speed=300, font_manager=font_manager)
        timeline = timeline.with_duration(video.duration)
        timeline = timeline.with_position(('center', h - 150)) 
        clips.append(timeline)
        
        chords = analysis.chords
        # Iterate chords
        for i, ac in enumerate(chords):
            start = ac.start
            if i < len(chords) - 1:
                end = chords[i+1].start
            else:
                end = video.duration
                
            duration = end - start
            if duration <= 0: continue

            # Current Chord
            clip_main = create_chord_clip(
                ac.symbol, start, duration, 
                font_manager,
                fontsize=int(h * 0.15),  # Reduced from 0.2 to prevent clipping
                pos=('center', 'center'), 
                color='yellow'
            )
            clips.append(clip_main)

            # Diagram
            try:
                clip_diag = create_diagram_clip(
                    ac.symbol, start, duration,
                    pos=(0.05, 0.1), width=200,
                    file_manager=file_manager,
                    work_dir=work_dir,
                    renderer=renderer
                )
                clips.append(clip_diag)
            except Exception as e:
                print(f"⚠️ Error generating diagram for {ac.symbol}: {e}")

            # Next Chord
            if i < len(chords) - 1:
                next_ac = chords[i+1]
                try:
                    # Adjusted position: Horizontal Layout [Next:] [Diagram] at Top Right
                    
                    # Dimensions
                    diag_width = 100 # Slightly smaller for compact layout
                    margin_right = 20
                    margin_top = 20
                    gap = 10
                    
                    # 1. Chord Diagram (Rightmost)
                    clip_next = create_diagram_clip(
                        next_ac.symbol, start, duration,
                        pos=('right', 'top'), 
                        width=diag_width,
                        file_manager=file_manager,
                        work_dir=work_dir,
                        renderer=renderer
                    )
                    # Position: Top Right
                    diag_x = w - diag_width - margin_right
                    clip_next = clip_next.with_position((diag_x, margin_top))
                    
                    # 2. Text Label "Next:" (Left of Diagram)
                    # Ensure font_path is available
                    font_path = font_manager.get_font_path()
                    
                    txt_next = create_styled_text_clip(
                        text="Next:", 
                        fontsize=30, 
                        color='white', 
                        font_path=font_path, 
                        duration=duration,
                        glow_color='cyan' # Subtle cyan glow for the label
                    )
                    
                    # Calculate text position relative to diagram
                    # We want it centered vertically relative to diagram, or aligned top
                    # Let's align centers roughly. Diagram is usually square-ish or tall.
                    # Text clip height is roughly fontsize + padding.
                    
                    # Position: Left of diagram
                    # We don't know exact text width here easily without checking clip.w, 
                    # but create_styled_text_clip returns a CompositeVideoClip where .w might work if rendered,
                    # but usually we align by right edge of text to left edge of diagram.
                    
                    # Since we can't easily get the width of the composite before rendering in some versions,
                    # we rely on the fact that we position it manually.
                    # Better: Position text at (diag_x - gap - text_width, margin_top)
                    # But we need text width. 
                    # create_styled_text_clip returns a clip, we can check its .w property (usually available if size was set)
                    
                    txt_x = diag_x - txt_next.w - gap
                    txt_y = margin_top + (diag_width - txt_next.h) // 2 # Vertically center relative to diagram width (approx height)
                    
                    txt_next = txt_next.with_position((txt_x, txt_y)).with_start(start)
                    
                    clips.append(txt_next)
                    clips.append(clip_next)
                except Exception as e:
                    print(f"⚠️ Error rendering next chord: {e}")
                    pass
        
        # Final Composite
        final_video = CompositeVideoClip(clips, size=(w, h))
        final_video = final_video.with_duration(video.duration)
        
        # Logger
        logger = MoviePyProgressLogger(progress, start_pct, end_pct)
        
        # Write file
        # Write file with conditional GPU support
        use_gpu = os.getenv("MOVIEPY_USE_GPU", "false").lower() == "true"
        
        if use_gpu:
            codec = os.getenv("MOVIEPY_FFMPEG_CODEC", "h264_nvenc")
            print(f"🚀 Rendering with GPU acceleration ({codec})...")
            
            # Build ffmpeg_params
            ffmpeg_params = []
            
            # Preset (default to 'fast' if not specified, but allow empty to skip)
            preset = os.getenv("MOVIEPY_FFMPEG_PRESET", "fast")
            if preset and preset.lower() != "none":
                ffmpeg_params.extend(['-preset', preset])
                
            # Extra params via env var (space separated)
            # e.g. "-global_quality 25 -look_ahead 1"
            extra_params = os.getenv("MOVIEPY_FFMPEG_PARAMS", "")
            if extra_params:
                ffmpeg_params.extend(extra_params.split())

            try:
                final_video.write_videofile(
                    analysis.output_path, 
                    codec=codec, 
                    audio_codec='aac', 
                    logger=logger,
                    threads=4, 
                    ffmpeg_params=ffmpeg_params
                )
            except Exception as e:
                print(f"⚠️ GPU processing failed ({codec}): {e}")
                print("🔄 Falling back to CPU rendering (libx264)...")
                final_video.write_videofile(
                    analysis.output_path, 
                    codec='libx264', 
                    audio_codec='aac', 
                    logger=logger,
                    threads=4,
                    preset='ultrafast'
                )
        else:
            print("🐌 Rendering with CPU (libx264)...")
            final_video.write_videofile(
                analysis.output_path, 
                codec='libx264', 
                audio_codec='aac', 
                logger=logger,
                threads=4,
                preset='ultrafast'
            )
        
        # Cleanup video objects
        video.close()
        final_video.close()

    except Exception as e:
        print(f"❌ Error rendering video: {e}")
        raise e
