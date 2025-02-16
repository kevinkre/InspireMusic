from inspiremusic.cli.inference import InspireMusicUnified
from inspiremusic.cli.inference import set_env_variables

if __name__ == "__main__":
    set_env_variables()
    model = InspireMusicUnified(model_name="InspireMusic-1.5B-Long")
    
    # Bach-style prompts
    prompts = [
        "Create a complex polyphonic piece in the style of Bach's Well-Tempered Clavier, featuring intricate counterpoint and a fugal structure. The piece should maintain a steady rhythmic pulse while weaving multiple melodic lines together.",
        "Compose a Bach-inspired chorale with four distinct voices - soprano, alto, tenor, and bass. The piece should follow traditional baroque harmonic progressions with clear cadences and tasteful ornamentation.",
        "Generate a piece in the style of Bach's inventions, with two independent melodic lines engaging in imitative counterpoint. The music should be in a minor key with a contemplative, scholarly character."
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating Bach-style piece {i+1}...")
        # Generate with longer duration for complete musical development
        model.inference(
            "text-to-music",
            prompt,
            chorus="intro",  # Use intro section for more freedom in structure
            start_time=0.0,
            end_time=120.0,  # 2 minutes
            result_dir="bach_generations",
            output_format="wav"
        )
        print(f"Completed piece {i+1}")
