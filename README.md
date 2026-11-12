# Soy-wax thermal buffer for compact AI hardware (simulation prototype)

The idea: put soy wax next to a small AI chip's heatsink. During a short burst of heavy load the
wax melts and absorbs heat. During idle time it re-solidifies and hands that heat to the fan.

**In the default scenario, 20 g of soy wax cast into aluminum foam lowers the chip's peak
temperature by 9 °C, removes all throttling in three 90-second AI jobs (stock heatsink: 131 s), and
holds full speed 3.8× longer in an 8-minute job.**

This repo holds the physics prototype, a narrated demo video built from it, and a small website
that presents both.

| File | What it is |
|---|---|
| `site/` | Static website (open `site/index.html`), deployed on Render |
| `site/assets/soywax_demo.mp4` | Narrated demo video, 1920x1080, 2:09 |
| `out/soywax_demo_narration.txt` | Narration script with timestamps |
| `out/advantages.png` | Chart of the headline results |
| `out/summary.png`, `out/results.json` | Full 36-minute run and numbers for every build |
| `soywax_sim.py` | The simulation |
| `make_demo_video.py` | Renders the video from the simulation |
| `render.yaml` | Render Blueprint for the website |

## Website

The site is plain HTML and CSS in `site/`, with no build step. To deploy your own copy on Render:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/AarushAgarwal-dev/soywax-thermal-buffer)

Render reads `render.yaml`, publishes `site/` as a static site, and redeploys on every push to `main`.

## Run it

```
python soywax_sim.py                   # design checks, results table, charts and results.json in out/
python make_demo_video.py              # narrated video (the voice needs internet on the first run)
python make_demo_video.py --no-voice   # captions only, works offline
python make_demo_video.py --stills     # preview frames only
```

Scenario options work with both scripts: `--wax-g`, `--latent` (J/g), `--t-melt`, `--k-foam`,
`--t-amb`, `--p-full`, `--p-idle`, `--r-ha`, `--c-hs`. For example, `--wax-g 40 --t-amb 35`.

Requirements: Python 3.10+, numpy, matplotlib, ffmpeg. The voice needs `edge-tts`.

## What is simulated

```
P(t) -> [chip + package] --0.7 K/W--> [heatsink, 25 J/K] --2.4 K/W--> 25 °C air (fan)
                                           |
                                    contact, 3000 W/m²K
                                           |
                                [wax layer 1] ... [wax layer 12]   (far face insulated)
```

- The chip is a 30 W boost / 4 W idle edge-AI module. Above 85 °C a governor lowers the clock.
  Power scales with clock³, and each job needs a fixed amount of work, so throttled jobs take longer.
- The wax pack is a 50 × 50 mm slab split into 12 layers. Each layer tracks its enthalpy, so
  melting and re-solidifying over the wax's melting range come out of the energy balance.
- The workload is three 90-second jobs, 7 minutes apart, then an 8-minute job.
- The video compares three builds: **A** the stock heatsink, **B** +20 g of soy wax (8.9 mm slab),
  and **C** +20 g of soy wax cast into aluminum foam (92 % wax by volume, 9.7 mm).
- The simulation also runs a fourth build, **D**, which adds 20 g of plain aluminum. It checks how
  much of any effect comes from added mass alone. Its numbers are in `out/results.json` and
  `out/summary.png`.

## Results (default scenario)

| Build | Peak junction, short jobs | Throttled, short jobs | Full speed in 8-min job | Time to finish 8-min job |
|---|---|---|---|---|
| A · Stock heatsink | 86.7 °C | 131 s | 0:52 | 9:16 |
| B · +20 g soy wax | 85.9 °C | 47 s | 1:11 | 8:53 |
| C · +20 g soy wax in aluminum foam | 77.8 °C | 0 s | 3:16 | 8:38 |

Highlights:

- **Wax in aluminum foam (C):** 9 °C lower peak temperature, no throttling in the short jobs, full
  speed held 3.8× longer in the long job, and the long job finishes 38 s sooner.
- **The simple wax pack (B)** alone cuts throttling by 64 % and extends full speed from 0:52 to 1:11.
- **Passive:** nothing to power or control. The foam build adds about 25 g (20 g wax, 5 g aluminum).

Design notes:

- **Put the wax in metal.** In a 90 s burst, heat reaches about 1.7 mm into plain wax. Foam, fins or a
  honeycomb keep all the wax within about 2 mm of metal.
- **Match the melting point to the hardware.** It should sit above the idle heatsink temperature
  (35 °C here) and below the point where throttling starts (64 °C at the heatsink). Soy wax at
  43–55 °C fits this device.
- **Size the pack for the burst.** 20 g stores about 2.2 kJ, against about 1.8 kJ of burst heat the
  fan cannot remove in 90 s.
- **Plan the recharge.** The wax re-solidifies in about 5 minutes of lighter load, and the chip idles
  2–8 °C warmer while it releases the stored heat.
- **Plan the mass.** Soy wax stores about 110 J/g, against about 200 J/g for commercial paraffin
  phase-change materials. Soy wax is plant-based. A paraffin PCM needs roughly half the mass.

## Assumptions and sources

| Parameter | Value used | Basis |
|---|---|---|
| Soy wax latent heat | 110 J/g | T-history tests gave 109 ± 13 J/g for soy wax, against 198–207 J/g for RT45HC/RT35HC paraffin ([MDPI Electronics 15(20), 4785](https://www.mdpi.com/2079-9292/15/20/4785)) |
| Melting range | 43–55 °C (centred on 49 °C) | Golden Wax 464 SDS lists 113–119 °F (45–48 °C); soy blends melt at roughly 115–130 °F (46–54 °C) ([Lone Star Candle Supply](https://lonestarcandlesupply.com/choosing-a-soy-wax/)) |
| Wax conductivity | 0.17 W/m·K solid, 0.15 liquid | Typical for waxes and fats. Soy wax with graphene or MAXene additives reached at most 0.89 W/m·K ([Trisnadewi et al. 2023](https://hal.science/hal-04150497)) |
| Wax in aluminum foam | 3 W/m·K, 92 % wax by volume | Aluminum-foam/paraffin composites measure 3–7 W/m·K, 30–60× pure paraffin ([Chen et al.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5560106); [MDPI Materials 17(17), 4332](https://www.mdpi.com/1996-1944/17/17/4332)) |
| Chip and heatsink | values in the diagram above | An illustrative compact edge-AI module, not a specific product |

## Model limits

- Conduction is 1-D and there is no convection in molten wax. That is conservative for pure wax.
- Melting and freezing follow the same curve. Real soy wax can supercool and freezes over a broad
  range, so the model may understate the refreeze time.
- Chip and heatsink are single lumps with no spreading resistance. The governor is a simple
  integral controller.
- Wax properties vary by blend. Measure yours (DSC or T-history) before trusting the numbers.
- Not modelled: molten wax leaks (it needs a sealed container), wax expands when it melts (the
  container needs headroom), vegetable wax oxidizes over many cycles, and wax is flammable.

These numbers are a reason to build a bench rig: heater + thermocouples + the same fan, with and
without the wax pack.

## Voice

The narration uses Microsoft's online Edge neural voice `en-US-AndrewNeural` via the `edge-tts`
package. Check Microsoft's terms before using it commercially. To change it, use
`--voice en-US-AvaNeural` or `--no-voice`.
