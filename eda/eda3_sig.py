import json, re, collections, sys, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
ATTR=set('brown white black yellow green red beige orange gray blue purple pink solid-colored striped polka-dotted checkered metal wood wooden plastic glass rubber fabric ceramic open closed standing sitting lying crouching kneeling one two three four single a an the'.split())
STY=set('photograph photographed oil painting watercolor gouache colored-pencil pencil sketch ink drawing flat illustration low-poly d render rendered paper-cut collage clay-style front front-view side side-view overhead oblique view viewed angle lighting lit direct sunlight soft diffuse warm indoor scene image shows shown shot uses is are in with from under and of to left right next above below on rests resting sits inside holds holding wears wearing rides riding which that who also present appears appear set setting contains there it both by alongside has made color pattern studio plain backdrop as'.split())
def objsig(t):
    w=re.findall(r"[a-z\-]+",t.lower())
    return frozenset(x for x in w if x not in ATTR and x not in STY)
