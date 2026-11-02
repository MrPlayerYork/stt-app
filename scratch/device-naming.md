# PortAudio (Python) vs cpal device names (same machine)

PortAudio (sounddevice) list: see scratch/portaudio.txt. The Python app used the JACK host-API entries
(pipewire-jack), so saved `audio.device_name` values look like "Blue Snowball Analog Stereo"
(sounddevice's `name` field has no ", JACK Audio Connection Kit" suffix; that is only print formatting).

cpal 0.18 hosts here (no libpipewire/libjack dev headers, so the pipewire/jack features can't build):
- PulseAudio host (pure Rust, talks to pipewire-pulse): "Blue Snowball Analog Stereo",
  "HD Pro Webcam C920 Analog Stereo", "Astro A50 Chat", "Starship/Matisse HD Audio Controller Analog Stereo",
  plus "Monitor of ..." entries. Names are IDENTICAL to PortAudio's JACK names (both are PipeWire node descriptions).
- ALSA host: "Blue Snowball, USB Audio" (hw/plughw/front/sysdefault duplicates, most busy because PipeWire owns
  them), "PipeWire Sound Server", "Default ... Output". Different from both PortAudio lists.
Only PortAudio-JACK-only entries (app streams like "Brave", "Astro A50 Game") have no cpal equivalent.
