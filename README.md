# Soy-wax thermal buffer for compact AI hardware (simulation prototype)

The idea: put soy wax next to a small AI chip's heatsink. During a short burst of heavy load the
wax melts and absorbs heat. During idle time it re-solidifies and hands that heat to the fan.

This folder holds a physics prototype of that idea, a narrated demo video built from it, and a
small website that presents both.

| File | What it is |
|---|---|
| `site/` | Static website (open `site/index.html`), deployed on Render |
| `site/assets/soywax_demo.mp4` | Narrated demo video, 1920x1080, 2:28 |
| `out/soywax_demo_narration.txt` | Narration script with timestamps |
| `out/summary.png`, `out/results.json` | Static plot and numbers for all four builds |
| `soywax_sim.py` | The simulation |
| `make_demo_video.py` | Renders the video from the simulation |
| `render.yaml` | Render Blueprint for the website |

In the video, "Al foam" means aluminum foam (Al is the chemical symbol for aluminum).

## Website

The site is plain HTML and CSS in `site/`, with no build step. To deploy your own copy on Render:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/AarushAgarwal-dev/soywax-thermal-buffer)

Render reads `render.yaml`, publishes `site/` as a static site, and redeploys on every push to `main`.

## Run it

```
python soywax_sim.py                   # design checks, results table, out/summary.png, out/results.json
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
- The four builds are the stock heatsink, +20 g of pure soy wax (8.9 mm slab), +20 g of soy wax
  cast into aluminum foam (92 % wax by volume, 9.7 mm), and a control with +20 g of plain aluminum.

## Results (default scenario)

| Build | Peak junction, short jobs | Throttled, short jobs | Full speed in 8-min job |
|---|---|---|---|
| Stock heatsink | 86.7 °C | 131 s | 0:52 |
| +20 g soy wax | 85.9 °C | 47 s | 1:11 |
| +20 g soy wax in aluminum foam | 77.8 °C | 0 s | 3:16 |
| +20 g aluminum (control) | 86.0 °C | 40 s | 1:16 |

What this says about the idea:

- **Wax in metal works, pure wax does not.** Pure wax conducts heat poorly, so a 90 s burst melts
  only about 1.7 mm of it. The pure slab performed about as well as the same mass of plain aluminum.
  Keep all the wax within about 2 mm of metal: foam, fins, or a honeycomb.
- **It buys time, not capacity.** Once the wax has melted, every build throttles to the same ~19 W.
  The fan and heatsink set that limit.
- **The melting point has to fit the hardware.** It must sit above the idle heatsink temperature
  (35 °C here) and below the point where throttling starts (64 °C at the heatsink). Soy wax at
  43–55 °C fits this device.
- **It needs idle time.** Re-solidifying takes about 5 minutes after each burst. During that time
  the chip idles 2–8 °C warmer.
- Soy wax stores about half as much heat per gram as commercial paraffin PCMs (~110 vs ~200 J/g).
  Expect roughly twice the mass of a paraffin PCM for the same buffer.

## Assumptions and sources

| Parameter | Value used | Basis |
|---|---|---|
| Soy wax latent heat | 110 J/g | T-history tests gave 109 ± 13 J/g for soy wax, against 198–207 J/g for RT45HC/RT35HC paraffin ([MDPI Electronics 15(20), 4785](https://www.mdpi.com/2079-9292/15/20/4785)) |
| Melting range | 43–55 °C (centred on 49 °C) | Golden Wax 464 SDS lists 113–119 °F (45–48 °C); soy blends melt at roughly 115–130 °F (46–54 °C) ([Lone Star Candle Supply](https://lonestarcandlesupply.com/choosing-a-soy-wax/)) |
| Wax conductivity | 0.17 W/m·K solid, 0.15 liquid | Typical for waxes and fats. Soy wax with graphene or MAXene additives reached at most 0.89 W/m·K ([Trisnadewi et al. 2023](https://hal.science/hal-04150497)) |
| Wax in aluminum foam | 3 W/m·K, 92 % wax by volume | Aluminum-foam/paraffin composites measure 3–7 W/m·K, 30–60× pure paraffin ([Chen et al.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5560106); [MDPI Materials 17(17), 4332](https://www.mdpi.com/1996-1944/17/17/4332)) |
| Chip and heatsink | values in the diagram above | An illustrative compact edge-AI module, not a specific product |

## Limitations

- Conduction is 1-D and there is no convection in molten wax. That is conservative for pure wax.
- Melting and freezing follow the same curve. Real soy wax can supercool and freezes over a broad
  range, so the model may understate the refreeze time.
- Chip and heatsink are single lumps with no spreading resistance. The governor is a simple
  integral controller.
- Wax properties vary by blend. Measure yours (DSC or T-history) before trusting the numbers.
- Not modelled: molten wax leaks (it needs a sealed container), wax expands when it melts (the
  container needs headroom), vegetable wax oxidizes over many cycles, and wax is flammable.

Treat these numbers as a reason to build a bench rig, not as a result: heater + thermocouples +
the same fan, with and without the wax pack.

## Voice

The narration uses Microsoft's online Edge neural voice `en-US-AndrewNeural` via the `edge-tts`
package. Check Microsoft's terms before using it commercially. To change it, use
`--voice en-US-AvaNeural` or `--no-voice`.
