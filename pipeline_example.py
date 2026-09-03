from pipeline import get_cue_and_scope, format_result

def main():
    examples = [
        # Negation. example from training.
        "They analyzed 146 prokaryotic genomes , but no likely tRNA of the "
        "novel amino acid was detected .",
        # Speculation ("should", "may") and negation ("not") in one sentence.
        "It may happen to be noted that the degree distribution is not maintained .",
        # No cue at all
        "The samples were incubated for thirty minutes .",
    ]

    text = "I'm not sure I handled that conversation well, and I probably should've apologized sooner."
    print(get_cue_and_scope(text))
    print()
    print(format_result(get_cue_and_scope(text)))
    

if __name__ == "__main__":
    main()
