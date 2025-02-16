#!/bin/bash

# Generate Bach-style piece using the Well-Tempered Clavier prompt
python -m inspiremusic.cli.inference \
  --task text-to-music \
  -m "InspireMusic-1.5B-Long" \
  -g 0 \
  -t "Create a complex polyphonic piece in the style of Bach's Well-Tempered Clavier, featuring intricate counterpoint and a fugal structure. The piece should maintain a steady rhythmic pulse while weaving multiple melodic lines together." \
  -c intro \
  -s 0.0 \
  -e 120.0 \
  -r "bach_generations" \
  -o "well_tempered_clavier" \
  -f wav

# Generate Bach-style chorale
python -m inspiremusic.cli.inference \
  --task text-to-music \
  -m "InspireMusic-1.5B-Long" \
  -g 0 \
  -t "Compose a Bach-inspired chorale with four distinct voices - soprano, alto, tenor, and bass. The piece should follow traditional baroque harmonic progressions with clear cadences and tasteful ornamentation." \
  -c intro \
  -s 0.0 \
  -e 120.0 \
  -r "bach_generations" \
  -o "chorale" \
  -f wav

# Generate Bach-style invention
python -m inspiremusic.cli.inference \
  --task text-to-music \
  -m "InspireMusic-1.5B-Long" \
  -g 0 \
  -t "Generate a piece in the style of Bach's inventions, with two independent melodic lines engaging in imitative counterpoint. The music should be in a minor key with a contemplative, scholarly character." \
  -c intro \
  -s 0.0 \
  -e 120.0 \
  -r "bach_generations" \
  -o "invention" \
  -f wav