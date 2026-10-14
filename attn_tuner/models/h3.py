"""MiniMax H3 (joint audio-video). Packed sequence: text, conditioning/references, audio, then the target video."""

from __future__ import annotations

from .base import Adapter



class H3(Adapter):
    name = "minimax_h3"
    title = "MiniMax H3"
    groups = ("video", "audio")
    # prefix keys (text, conditioning, audio) and the audio queries stay exact; H3 publishes both spans itself
    sinks = {"sink_keys": "prefix", "sink_queries": "audio"}
    # The two sets do not overlap. Both span what stresses a joint audio-video model: speech and lip sync, music, fast
    # motion, fine texture, text in frame, low light, and several shot types.
    eval = {
        "fox_snow": "A red fox trots through a snowy pine forest at dawn, breath steaming. Soft crunch of snow underfoot, "
        "distant birdsong.",
        "chef_talk": "A chef in a busy restaurant kitchen looks at the camera and says, 'Tonight we are making fresh "
        "pasta from scratch.' Sizzling pans and clattering plates behind her.",
        "street_rain": "Night city street in heavy rain, neon signs reflecting in puddles, a cyclist passes quickly. "
        "Rain hiss, tires on wet asphalt, a far-off siren.",
        "violin": "Close-up of an elderly man playing a slow melancholic melody on a violin in a wood-panelled room, "
        "warm lamp light, visible bow hair and rosin dust.",
        "surf_drone": "Drone shot tracking a surfer riding a large turquoise wave, spray catching the sun. Roaring "
        "surf and wind.",
        "chainmail": "Macro shot slowly panning across a knight's chainmail and engraved silver gauntlet, candlelight "
        "flickering. Faint metallic clinks.",
        "cafe_sign": "A barista writes 'OPEN 7AM' in chalk on a small sign outside a cafe, then flips it around. "
        "Chalk scratching, quiet street ambience.",
        "kids_park": "Two children laugh and chase each other around a playground on a sunny afternoon, a dog barking "
        "excitedly as it runs after them.",
    }
    calib = [
        "A hummingbird hovers at a red flower in slow motion, wings a soft blur. Rapid buzzing.",
        "A news anchor in a studio reads the headline, 'Markets closed higher today after a volatile week.'",
        "Time-lapse of clouds rolling over mountain peaks at sunset, the light shifting from gold to purple.",
        "A jazz trio plays in a dim club: upright bass, brushed drums, and piano. Glasses clink in the background.",
        "A cat knocks a glass of water off a kitchen counter; the glass shatters on the tile floor.",
        "First-person view of a mountain bike descending a rocky forest trail at high speed, gravel crunching.",
        "A woman whispers a secret to her friend on a park bench; leaves rustle in the breeze.",
        "Fireworks burst over a harbor at night, reflections on the water, booming echoes across the bay.",
    ]

    # ref2va prompts in the structured format H3's reference workflows use (the SMD Ref2VA card's example), for two
    # references: <Picture 1> a person, <Picture 2> a small object or toy. Scenes, sound and speech vary as above.
    ref2va_scenes = {
        "workbench": ("watches <Subject 2> come to life on a sunlit workbench and waves back.",
                      "A medium close shot in a small home workshop; <Subject 2> stands on the bench, turns its head with "
                      "a servo whirr and raises one arm. <Subject 1> sits behind it, smiles and says, <d>[English] Hello "
                      "there.</d> Slow push-in.", "Workshop room tone, servo whirrs, two electronic chirps."),
        "rain_window": ("sits by a rain-streaked window at night holding <Subject 2>.",
                        "Close shot lit by neon from the street; rain runs down the glass. <Subject 1> turns <Subject 2> "
                        "in his hands, then sets it on the sill. Handheld, slight drift.",
                        "Heavy rain on glass, distant traffic, a far-off siren."),
        "kitchen": ("cooks pasta while <Subject 2> watches from the counter.",
                    "Wide shot of a busy home kitchen; steam rises from a pot. <Subject 1> looks at the camera and says, "
                    "<d>[English] Tonight we are making fresh pasta.</d> Quick pan to <Subject 2> on the counter.",
                    "Sizzling pan, bubbling water, a clattering lid."),
        "beach": ("runs along a beach at sunset carrying <Subject 2>.",
                  "Tracking shot at low height; waves break behind him, spray catches the light. He lifts <Subject 2> "
                  "above his head and laughs.", "Roaring surf, wind, laughter, gulls."),
        "jazz_club": ("plays upright bass in a dim jazz club while <Subject 2> sits on the amp.",
                      "Medium shot, warm stage light and haze; fingers move fast on the strings; glasses glint in the "
                      "foreground.", "Upright bass, brushed drums and piano; glasses clink."),
        "snow_park": ("builds a snowman with <Subject 2> on a snowy morning.",
                      "Wide shot in a park; breath steams; <Subject 2> sits in the snow as <Subject 1> rolls a large "
                      "snowball past it and kneels to whisper, <d>[English] Almost done.</d>",
                      "Crunching snow, soft wind, distant children."),
        "chalk_sign": ("writes 'OPEN 7AM' in chalk on a cafe sign while <Subject 2> leans against it.",
                       "Close shot of the chalk letters forming, then a tilt up to his face; morning light.",
                       "Chalk scratching, quiet street ambience."),
        "fireworks": ("watches fireworks over a harbor with <Subject 2> on his shoulder.",
                      "Wide night shot; bursts reflect on the water; the camera cranes up past them.",
                      "Booming echoes across the bay, crowd cheering."),
    }

    @staticmethod
    def ref2va_prompt(summary, shot, sound):
        return ("subject_definitions:\n<Subject 1> is the adult man in <Picture 1>. The image supplies his identity and "
                "clothing.\n<Subject 2> is the toy in <Picture 2>. The image supplies its design and colors.\n\n"
                f"summary:\n[reference generation] <Subject 1> {summary}\n\n"
                "retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved.\n<Subject 2> (appears in "
                "[Shot 1]): fully_preserved.\n\n"
                f"detailed_description:\nLive-action photography.\n[Shot 1] {shot}\n\n"
                f"overall_soundscape:\n{sound}")

    def add_args(self, p):
        p.set_defaults(unet="minimax_h3_fl2va_pruned_int8_convrot.safetensors", steps=4, width=960, height=544,
                       sampler="res_multistep", scheduler="simple")
        p.add_argument("--length", type=int, default=124, help="frames")
        p.add_argument("--shift", default=None, help="MiniMaxH3SigmaShift video,audio, e.g. 12,2")
        p.add_argument("--ref-image", action="append", default=[],
                       help="ref2va: condition on this reference image (repeatable, in <Picture N> order)")
        p.add_argument("--ref-short-edge", type=int, default=0,
                       help="resize references to this short edge first, as a workflow's resize node does (0 = no)")
        p.add_argument("--text-encoder", default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
        p.add_argument("--vae", default="minimax_h3_video_vae_int8_convrot.safetensors",
                       help="video VAE, for encoding references")
        p.add_argument("--ref2va-prompts", action="store_true",
                       help="structured two-reference prompts (person + toy) instead of the plain prompt sets")

    def load(self, nodes_map, args):
        from ..runtime import call

        if args.ref2va_prompts:
            if len(args.ref_image) != 2:
                raise SystemExit("--ref2va-prompts describe two references: pass --ref-image twice (a person, then "
                                 "an object)")
            scenes = [self.ref2va_prompt(*v) for v in self.ref2va_scenes.values()]
            self.calib, self.eval = scenes[4:], dict(zip(list(self.ref2va_scenes)[:4], scenes[:4]))
        (model,) = call(nodes_map, "UNETLoader", unet_name=args.unet, weight_dtype="default")
        for lora in args.lora:  # LoRAs before the shift, the order of H3's reference workflows
            (model,) = call(nodes_map, "LoraLoaderModelOnly", model=model, lora_name=lora, strength_model=1.0)
        if args.shift:
            video, audio = map(float, args.shift.split(","))
            (model,) = call(nodes_map, "MiniMaxH3SigmaShift", model=model, shift_video=video, shift_audio=audio)
        (clip,) = call(nodes_map, "CLIPLoader", clip_name=args.text_encoder, type="minimax", device="default")
        (vae,) = call(nodes_map, "VAELoader", vae_name=args.vae)
        return model, (clip, vae)

    def conditioning(self, nodes_map, ctx, args, prompt):
        import comfy.model_management

        from ..runtime import call

        clip, vae = ctx
        # the DiT from the previous prompt would otherwise stay resident while the VAE encodes references at high res
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        if args.ref_image:
            # ref2va: the node then scales each reference (down only) to the generation's area
            import numpy as np
            import torch
            from PIL import Image

            refs = {}
            for i, path in enumerate(args.ref_image):
                im = Image.open(path).convert("RGB")
                if args.ref_short_edge:
                    s = args.ref_short_edge / min(im.size)
                    im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
                refs[f"ref_image_{i}"] = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255)[None]
            # plain prompts don't name the reference; structured ones already do
            text = prompt if "<Picture 1>" in prompt else f"<Picture 1> {prompt}"
            return call(nodes_map, "MiniMaxH3ReferenceToVideo", clip=clip, vae=vae, prompt=text, width=args.width,
                        height=args.height, length=args.length, ref_images=refs)[:2]
        return call(nodes_map, "MiniMaxH3ImageToVideo", clip=clip, vae=vae, prompt=prompt, width=args.width,
                    height=args.height, length=args.length)[:2]

    def rows(self, to):
        layout = to.get("minimax_h3_layout")
        if to.get("block_index") is None or layout is None:
            return None
        return {g: [(a, b) for a, b, kind in layout.segments if kind == g] for g in self.groups}

    def split_latent(self, samples):
        video, audio = samples.unbind()[:2]  # NestedTensor (video, audio)
        return {"video": video, "audio": audio}

    def split_output(self, x, c):
        import comfy.utils

        # ComfyUI packs [video, audio] into one flat latent; `latent_shapes` in the conditioning undoes it.
        shapes = c.get("latent_shapes")
        if shapes is None or len(shapes) < 2:
            raise RuntimeError("could not split the model output into video and audio")
        video, audio = comfy.utils.unpack_latents(x, shapes)[:2]
        return {"video": video, "audio": audio}
