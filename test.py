#!/usr/bin/env python
# coding: utf8

from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding



# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
)
def main(model=Path("L:\\BigData\\Env\\SavedModels"), output_dir=Path("L:\\BigData\\Env\\SavedModels")):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)

        ner = nlp.get_pipe("ner")

    ner.add_label("Email")# add new entity label to entity recognizer
    ner.add_label("Full Name")
    ner.add_label("First Name")
    ner.add_label("Updatedtime")
    ner.add_label("City")
    ner.add_label("Postcode")
    ner.add_label("Date")
    ner.add_label("Phone")
    ner.add_label("State")
    ner.add_label("Country")
    ner.add_label("Street")
    ner.add_label("Id")
    ner.add_label("Last Name")
    ner.add_label("username")
    ner.add_label("Transaction Id")

    move_names = list(ner.move_names)

    # test the trained model
    test_text = "Carlie@marguerite.ca , Glenna  Olson , Corbin , 1999-07-05T12:33:47.066Z , Kingshire 77787 1/8/2013 | (695)464-6165 x9467 , Iowa Indonesia 038 Schumm Walks 0 Ortiz Frida 362 "
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    plac.call(main)
