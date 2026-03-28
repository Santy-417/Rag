"""
voice_avatar.py — Robot avatar animado para la pestaña de Voz.
Uso:
    from voice_avatar import render_avatar
    render_avatar("idle")   # idle | listening | processing | speaking
"""

import streamlit.components.v1 as components


def get_avatar_html(state: str = "idle") -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: transparent; }}
.scene {{
  display: flex; flex-direction: column; align-items: center;
  padding: 28px 16px 20px; gap: 18px;
  background: #0d0f14; border-radius: 16px;
}}
.robot-wrap {{ position: relative; width: 200px; }}
svg.robot {{ overflow: visible; }}
.state-label {{
  font-family: monospace; font-size: 11px; letter-spacing: 0.12em;
  text-transform: uppercase; color: #3a4a6a; text-align: center;
}}
.state-label span {{ color: #4a7fd4; }}
.waveform {{
  display: flex; align-items: center; gap: 3px;
  height: 32px; opacity: 0; transition: opacity 0.4s;
}}
.waveform.show {{ opacity: 1; }}
.wbar {{
  width: 3px; border-radius: 2px; background: #2a5db0;
  min-height: 3px; transition: height 0.08s ease;
}}
@keyframes antenna-blink {{
  0%,40%,100% {{ opacity: 1; }} 20% {{ opacity: 0.15; }}
}}
@keyframes eye-pulse {{
  0%,100% {{ opacity: 0.7; }} 50% {{ opacity: 1; }}
}}
@keyframes breathe-body {{
  0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-2px); }}
}}
@keyframes listen-bob {{
  0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-5px); }}
}}
@keyframes proc-spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes proc-spin-rev {{ to {{ transform: rotate(-360deg); }} }}
@keyframes speak-bounce {{
  0%,100% {{ transform: translateY(0) scaleY(1); }}
  50% {{ transform: translateY(-3px) scaleY(1.02); }}
}}
@keyframes ring-expand {{
  0% {{ r: 0; opacity: 0.6; stroke-width: 2; }}
  100% {{ r: 90; opacity: 0; stroke-width: 0.5; }}
}}
.robot-group {{ animation: breathe-body 4s ease-in-out infinite; }}
.antenna-led {{ animation: antenna-blink 2s ease-in-out infinite; }}
.eye-l, .eye-r {{ animation: eye-pulse 3s ease-in-out infinite; }}
.proc-ring-outer, .proc-ring-inner {{ animation: none; }}
.sound-ring-1, .sound-ring-2 {{ opacity: 0; }}
.state-listening .robot-group {{ animation: listen-bob 0.9s ease-in-out infinite; }}
.state-listening .antenna-led {{ animation: antenna-blink 0.6s ease-in-out infinite; fill: #4af0a0 !important; }}
.state-listening .eye-l, .state-listening .eye-r {{ fill: #4af0a0 !important; animation: eye-pulse 0.7s ease-in-out infinite; }}
.state-listening .sound-ring-1 {{ animation: ring-expand 1.4s ease-out infinite; }}
.state-listening .sound-ring-2 {{ animation: ring-expand 1.4s ease-out 0.4s infinite; }}
.state-processing .robot-group {{ animation: none; }}
.state-processing .antenna-led {{ animation: antenna-blink 0.3s linear infinite; fill: #f0a040 !important; }}
.state-processing .eye-l, .state-processing .eye-r {{ fill: #f0a040 !important; animation: eye-pulse 0.3s linear infinite; }}
.state-processing .proc-ring-outer {{ transform-origin: 100px 148px; animation: proc-spin 2.5s linear infinite; }}
.state-processing .proc-ring-inner {{ transform-origin: 100px 148px; animation: proc-spin-rev 1.8s linear infinite; }}
.state-speaking .robot-group {{ animation: speak-bounce 0.5s ease-in-out infinite; }}
.state-speaking .antenna-led {{ animation: antenna-blink 0.4s ease-in-out infinite; fill: #4ab0ff !important; }}
.state-speaking .eye-l, .state-speaking .eye-r {{ fill: #4ab0ff !important; animation: eye-pulse 0.45s ease-in-out infinite; }}
.state-speaking .sound-ring-1 {{ animation: ring-expand 0.8s ease-out infinite; }}
.state-speaking .sound-ring-2 {{ animation: ring-expand 0.8s ease-out 0.2s infinite; }}
</style>
</head>
<body>
<div class="scene">
  <div class="robot-wrap state-{state}" id="robotWrap">
    <svg class="robot" viewBox="0 0 200 300" width="200" height="300">
      <defs>
        <radialGradient id="eyeGlow" cx="50%" cy="40%" r="60%">
          <stop offset="0%" stop-color="#ffffff" stop-opacity="0.9"/>
          <stop offset="100%" stop-color="#2266cc" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="bodyGrad" cx="40%" cy="30%" r="70%">
          <stop offset="0%" stop-color="#2a3248"/><stop offset="100%" stop-color="#151820"/>
        </radialGradient>
        <radialGradient id="headGrad" cx="40%" cy="30%" r="70%">
          <stop offset="0%" stop-color="#303a52"/><stop offset="100%" stop-color="#1a2030"/>
        </radialGradient>
      </defs>
      <circle class="sound-ring-1" cx="100" cy="148" r="0" fill="none" stroke="#2a6ad4" stroke-width="2"/>
      <circle class="sound-ring-2" cx="100" cy="148" r="0" fill="none" stroke="#2a6ad4" stroke-width="1.5"/>
      <g class="robot-group">
        <line x1="100" y1="20" x2="100" y2="44" stroke="#2a3248" stroke-width="3" stroke-linecap="round"/>
        <circle cx="100" cy="15" r="5" fill="#1a2030" stroke="#2a3248" stroke-width="1.5"/>
        <circle class="antenna-led" cx="100" cy="15" r="3" fill="#4a7fd4"/>
        <rect x="36" y="44" width="128" height="90" rx="14" fill="url(#headGrad)" stroke="#2a3550" stroke-width="1.5"/>
        <rect x="44" y="52" width="112" height="74" rx="10" fill="#1a2236" stroke="#222d44" stroke-width="1"/>
        <rect x="52" y="62" width="38" height="26" rx="8" fill="#0d1220" stroke="#1e2d50" stroke-width="1.2"/>
        <ellipse class="eye-l" cx="71" cy="75" rx="11" ry="9" fill="#2a5db0"/>
        <ellipse cx="71" cy="72" rx="5" ry="3" fill="url(#eyeGlow)" opacity="0.5"/>
        <circle cx="71" cy="75" r="4" fill="#0a0e18"/>
        <circle cx="69" cy="73" r="1.5" fill="#ffffff" opacity="0.7"/>
        <rect x="110" y="62" width="38" height="26" rx="8" fill="#0d1220" stroke="#1e2d50" stroke-width="1.2"/>
        <ellipse class="eye-r" cx="129" cy="75" rx="11" ry="9" fill="#2a5db0"/>
        <ellipse cx="129" cy="72" rx="5" ry="3" fill="url(#eyeGlow)" opacity="0.5"/>
        <circle cx="129" cy="75" r="4" fill="#0a0e18"/>
        <circle cx="127" cy="73" r="1.5" fill="#ffffff" opacity="0.7"/>
        <rect x="62" y="98" width="76" height="10" rx="5" fill="#0d1220" stroke="#1e2d50" stroke-width="1"/>
        <rect id="mouthBar" x="65" y="100.5" width="20" height="5" rx="2.5" fill="#2a5db0"/>
        <rect x="89" y="100.5" width="30" height="5" rx="2.5" fill="#1e2d50"/>
        <rect x="123" y="100.5" width="12" height="5" rx="2.5" fill="#1e2d50"/>
        <rect x="55" y="57" width="8" height="5" rx="2" fill="#1e2d50" stroke="#2a3a5a" stroke-width="0.8"/>
        <rect x="137" y="57" width="8" height="5" rx="2" fill="#1e2d50" stroke="#2a3a5a" stroke-width="0.8"/>
        <circle cx="59" cy="59.5" r="1.5" fill="#3a6ad4" opacity="0.8"/>
        <circle cx="141" cy="59.5" r="1.5" fill="#3a6ad4" opacity="0.8"/>
        <rect x="52" y="134" width="96" height="88" rx="10" fill="url(#bodyGrad)" stroke="#2a3550" stroke-width="1.5"/>
        <rect x="60" y="142" width="80" height="60" rx="7" fill="#0d1220" stroke="#1a2640" stroke-width="1"/>
        <rect x="68" y="150" width="28" height="20" rx="4" fill="#111828" stroke="#1e3050" stroke-width="1"/>
        <rect x="72" y="154" width="6" height="3" rx="1.5" fill="#2a5db0" opacity="0.7"/>
        <rect x="80" y="154" width="10" height="3" rx="1.5" fill="#1e2d50"/>
        <rect x="72" y="159" width="16" height="3" rx="1.5" fill="#1e2d50"/>
        <rect x="72" y="164" width="8" height="3" rx="1.5" fill="#2a5db0" opacity="0.5"/>
        <rect x="104" y="150" width="28" height="20" rx="4" fill="#111828" stroke="#1e3050" stroke-width="1"/>
        <circle cx="110" cy="156" r="2.5" fill="#1e2d50"/>
        <circle cx="118" cy="156" r="2.5" fill="#2a5db0" opacity="0.8"/>
        <circle cx="126" cy="156" r="2.5" fill="#1e2d50"/>
        <circle cx="114" cy="163" r="2.5" fill="#1e2d50"/>
        <circle cx="122" cy="163" r="2.5" fill="#1e2d50"/>
        <rect x="64" y="178" width="72" height="16" rx="4" fill="#111828" stroke="#1e3050" stroke-width="1"/>
        <rect x="68" y="182" width="12" height="8" rx="2" fill="#1a3a70"/>
        <rect x="84" y="184" width="24" height="4" rx="2" fill="#1e2d50"/>
        <rect x="112" y="184" width="8" height="4" rx="2" fill="#2a5db0" opacity="0.6"/>
        <rect x="124" y="184" width="8" height="4" rx="2" fill="#1e2d50"/>
        <g class="proc-ring-outer">
          <circle cx="100" cy="148" r="52" fill="none" stroke="#1e3a70" stroke-width="1" stroke-dasharray="6 4" opacity="0.6"/>
        </g>
        <g class="proc-ring-inner">
          <circle cx="100" cy="148" r="44" fill="none" stroke="#2a5db0" stroke-width="1" stroke-dasharray="3 6" opacity="0.5"/>
        </g>
        <rect x="22" y="136" width="28" height="56" rx="8" fill="#1a2236" stroke="#2a3550" stroke-width="1.2"/>
        <rect x="28" y="144" width="16" height="8" rx="3" fill="#0d1220" stroke="#1e2d50" stroke-width="0.8"/>
        <circle cx="36" cy="166" r="6" fill="#0d1220" stroke="#1e2d50" stroke-width="0.8"/>
        <circle cx="36" cy="166" r="3" fill="#1e3060" opacity="0.8"/>
        <rect x="14" y="148" width="8" height="3" rx="1.5" fill="#1a2236" stroke="#2a3550" stroke-width="0.8"/>
        <rect x="14" y="160" width="8" height="3" rx="1.5" fill="#1a2236" stroke="#2a3550" stroke-width="0.8"/>
        <rect x="150" y="136" width="28" height="56" rx="8" fill="#1a2236" stroke="#2a3550" stroke-width="1.2"/>
        <rect x="156" y="144" width="16" height="8" rx="3" fill="#0d1220" stroke="#1e2d50" stroke-width="0.8"/>
        <circle cx="164" cy="166" r="6" fill="#0d1220" stroke="#1e2d50" stroke-width="0.8"/>
        <circle cx="164" cy="166" r="3" fill="#1e3060" opacity="0.8"/>
        <rect x="178" y="148" width="8" height="3" rx="1.5" fill="#1a2236" stroke="#2a3550" stroke-width="0.8"/>
        <rect x="178" y="160" width="8" height="3" rx="1.5" fill="#1a2236" stroke="#2a3550" stroke-width="0.8"/>
        <rect x="60" y="222" width="32" height="60" rx="8" fill="#1a2236" stroke="#2a3550" stroke-width="1.2"/>
        <rect x="66" y="240" width="20" height="6" rx="2" fill="#0d1220" stroke="#1a2640" stroke-width="0.8"/>
        <rect x="64" y="278" width="28" height="6" rx="3" fill="#151c2e" stroke="#2a3550" stroke-width="0.8"/>
        <rect x="108" y="222" width="32" height="60" rx="8" fill="#1a2236" stroke="#2a3550" stroke-width="1.2"/>
        <rect x="114" y="240" width="20" height="6" rx="2" fill="#0d1220" stroke="#1a2640" stroke-width="0.8"/>
        <rect x="108" y="278" width="28" height="6" rx="3" fill="#151c2e" stroke="#2a3550" stroke-width="0.8"/>
        <line x1="68" y1="144" x2="92" y2="144" stroke="#1e2d50" stroke-width="0.5" stroke-dasharray="2 2"/>
        <line x1="108" y1="144" x2="132" y2="144" stroke="#1e2d50" stroke-width="0.5" stroke-dasharray="2 2"/>
        <line x1="100" y1="200" x2="68" y2="222" stroke="#2a3550" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="100" y1="200" x2="132" y2="222" stroke="#2a3550" stroke-width="1.5" stroke-linecap="round"/>
      </g>
    </svg>
  </div>
  <div class="waveform" id="waveform">
    <div class="wbar" id="wb1"  style="height:6px"></div>
    <div class="wbar" id="wb2"  style="height:10px"></div>
    <div class="wbar" id="wb3"  style="height:16px"></div>
    <div class="wbar" id="wb4"  style="height:8px"></div>
    <div class="wbar" id="wb5"  style="height:20px"></div>
    <div class="wbar" id="wb6"  style="height:12px"></div>
    <div class="wbar" id="wb7"  style="height:18px"></div>
    <div class="wbar" id="wb8"  style="height:7px"></div>
    <div class="wbar" id="wb9"  style="height:14px"></div>
    <div class="wbar" id="wb10" style="height:9px"></div>
    <div class="wbar" id="wb11" style="height:22px"></div>
    <div class="wbar" id="wb12" style="height:6px"></div>
  </div>
  <div class="state-label">estado: <span id="stateLabel">cargando</span></div>
</div>
<script>
(function() {{
  const LABELS = {{
    idle: 'en espera', listening: 'escuchando...',
    processing: 'procesando...', speaking: 'respondiendo...'
  }};
  const wbars = Array.from({{length:12}}, (_,i) => document.getElementById('wb'+(i+1)));
  const waveform   = document.getElementById('waveform');
  const stateLabel = document.getElementById('stateLabel');
  const mouthBar   = document.getElementById('mouthBar');
  let barInterval = null, mouthInterval = null;

  function stopAll() {{
    if(barInterval)   {{ clearInterval(barInterval);   barInterval   = null; }}
    if(mouthInterval) {{ clearInterval(mouthInterval); mouthInterval = null; }}
    wbars.forEach(b => b.style.height = '4px');
    if(mouthBar) mouthBar.setAttribute('width','20');
    waveform.className = 'waveform';
  }}

  const state = '{state}';
  stateLabel.textContent = LABELS[state] || state;

  if(state === 'listening') {{
    waveform.className = 'waveform show';
    wbars.forEach(b => b.style.background = '#2a8a60');
    barInterval = setInterval(() => {{
      wbars.forEach(b => b.style.height = (Math.random()*20+3)+'px');
    }}, 120);
  }} else if(state === 'speaking') {{
    waveform.className = 'waveform show';
    wbars.forEach(b => b.style.background = '#2a5db0');
    barInterval = setInterval(() => {{
      wbars.forEach(b => b.style.height = (Math.random()*24+3)+'px');
    }}, 180);
    mouthInterval = setInterval(() => {{
      if(mouthBar) mouthBar.setAttribute('width', Math.floor(Math.random()*42+8)+'');
    }}, 200);
  }}
}})();
</script>
</body>
</html>"""


def render_avatar(state: str = "idle", height: int = 420) -> None:
    """
    Renderiza el robot avatar en la pestaña activa de Streamlit.

    Args:
        state:  "idle" | "listening" | "processing" | "speaking"
        height: Alto del iframe en píxeles (default 420)
    """
    components.html(get_avatar_html(state), height=height)
