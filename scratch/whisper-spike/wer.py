"""Word error rate against the user's reference transcript. Usage: wer.py out_*.log ..."""
import re
import sys

REF = ("This is a test. I am making sure that the audio is works correctly on both models so we can "
       "compare them. I don't know if it will work, but I hope this audio helps in some way, shape, or form.")


def norm(s):
    return re.sub(r"[^a-z0-9' ]", " ", s.lower()).replace("'", "").split()


def wer(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / len(ref)


ref = norm(REF)
for path in sys.argv[1:]:
    text = next(l[6:] for l in open(path) if l.startswith("TEXT: "))
    print(f"{path}: WER {wer(ref, norm(text)):.3f} ({round(wer(ref, norm(text))*len(ref))} edits / {len(ref)} words)")
