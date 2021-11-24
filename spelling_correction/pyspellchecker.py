from spellchecker import SpellChecker

spell = SpellChecker(language='de')

# find those words that may be misspelled
misspelled = spell.unknown(['Halloo', 'Wie', 'Psot', 'Kiwi'])

for word in misspelled:
    # Get the one `most likely` answer
    print(spell.correction(word))

    # Get a list of `likely` options
    print(spell.candidates(word))

    print(spell.word_probability(word))
