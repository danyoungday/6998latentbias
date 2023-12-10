import difflib

def get_overlap_indices(pos, neg):
    """
    Uses difflib library to find substitutions in neg from pos.
    Returns masks where 1 is a difference and 0 is the same.
    """
    pos_split = pos.split()
    neg_split = neg.split()
    diff = list(difflib.unified_diff(pos_split, neg_split, n=max(len(pos_split), len(neg_split))))[3:]
    pos_mask = []
    neg_mask = []
    for d in diff:
        if d.startswith('-'):
            pos_mask.append(1)
        elif d.startswith('+'):
            neg_mask.append(1)
        else:
            pos_mask.append(0)
            neg_mask.append(0)

    return pos_mask, neg_mask


def get_masked_and_label(string, mask):
    """
    Masks string and produces label according to T5 format.
    """
    split = string.split()
    masked = []
    special_idx = 0
    for i in range(len(split)):
        if mask[i] == 1:
            if i > 0 and mask[i-1] == 1:
                continue
            masked.append(f"<extra_id_{special_idx}>")
            special_idx += 1
        else:
            masked.append(split[i])

    special_idx = 1
    label = ["<extra_id_0>"]
    for i in range(len(split)):
        if mask[i] == 1:
            if i > 0 and mask[i-1] == 0 and not label[-1].startswith("<extra_id_"):
                label.append(f"<extra_id_{special_idx}>")
            label.append(split[i])

    return " ".join(masked), " ".join(label)