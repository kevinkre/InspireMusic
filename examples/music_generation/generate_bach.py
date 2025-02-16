"""Bach-style music generation script"""

from inspiremusic.cli.inference import InspireMusicUnified
from inspiremusic.cli.inference import set_env_variables

if __name__ == "__main__":
    set_env_variables()
    model = InspireMusicUnified(model_name="InspireMusic-1.5B-Long")
    
    # Bach-style prompts
    prompts = [
        "Create a complex polyphonic piece in the style of Bach's Well-Tempered Clavier...",
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating Bach-style piece {i+1}...")
        model.inference(
            "text-to-music",
            prompt,
            chorus="intro",
            start_time=0.0,
            end_time=120.0,
            result_dir="bach_generations",
            output_format="wav"
        )